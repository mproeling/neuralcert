"""Pair-sharded distributed runtime for the gated discovery pipeline.

The activation-heavy pair axis is sharded explicitly.  This module contains no
objective or certification logic; it only owns process-group coordination and
pair assignment metadata.
"""
from __future__ import annotations

import os
import socket
import sys
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as tdist
import torch.nn as nn

# ===========================================================================
# DISTRIBUTED LAYER — pair-sharded data parallelism
# ===========================================================================
# WHY THIS AXIS, AND WHY NOT FSDP/ZeRO/TENSOR PARALLELISM
# ------------------------------------------------------
# The VRAM here is ACTIVATIONS, not parameters. The dominant term is the
# xi recursion:
#
#     mem ~ 2 * (#chain steps) * N * nq * P * 8 bytes,   P = m(m+1)/2
#
# ~1.1 MB per pair at N=128, nq=32, k=1000. The ChannelNet is ~10^4 doubles,
# i.e. noise. So parameter-sharding schemes (FSDP, ZeRO, tensor parallel)
# address the wrong resource entirely and buy nothing. What binds is P, which
# grows QUADRATICALLY in m — and P is a perfectly parallel axis.
#
# For pair p = (j,l), the whole chain — _xi_power, _log_inner,
# _outer_integral_log_shared — touches only channels j and l. There is no
# cross-pair coupling until the assembly of A and B. Each pair costs
# ~(#steps)*N*nq*(shared mesh) flops and produces exactly TWO SCALARS, so the
# compute-to-communication ratio is enormous: this runs fine over Ethernet
# with gloo, never mind NVLink.
#
# WHY DDP IS THE WRONG WRAPPER
# ----------------------------
# nn.parallel.DistributedDataParallel averages gradients, assuming a loss
# that is a MEAN over a sample axis. Here the reduction is a SUM over pairs
# INSIDE a nonlinear ratio, so DDP would silently deliver 1/world_size of a
# gradient that is also structurally wrong. We use a bare process group and
# reduce by hand.
#
# THE GRADIENT (exact, no differentiable collectives needed)
# ----------------------------------------------------------
# Hellmann-Feynman already detaches the Ritz vector v, so with
# c_p = (2 - delta_{jl}) v_j v_l the numerator and denominator are plain sums
# over pairs:
#
#     num = sum_p c_p B_p,     den = sum_p c_p A_p,     R = k num / den.
#
# Hence
#
#     dR/dtheta = sum_ranks [ (k/den) dnum_r/dtheta
#                             - (k num/den^2) dden_r/dtheta ],
#
# so each rank all-reduces the two scalar VALUES, then backprops the local
# surrogate with num and den held constant, then all-reduces parameter
# gradients with op=SUM. That is the EXACT gradient — it is just the quotient
# rule — and it needs only two scalar all-reduces forward and one gradient
# all-reduce backward. No torch.distributed.nn autograd-aware collectives.
#
# The surrogate's own value is identically zero (it is a directional
# derivative, not the function), so rayleigh() adds back the true R via the
# standard  surr - surr.detach() + R_val  trick: correct value, correct
# gradient.
#
# THE THREE HAZARDS, AND WHERE THEY ARE HANDLED
# ---------------------------------------------
#  (1) COLLECTIVE DIVERGENCE. Every rank must execute the same collectives in
#      the same order or the job deadlocks. report() raises ValidationError
#      on gate failures, so a gate that fires on one rank and not another
#      hangs the job. Dist.agree() makes every such verdict unanimous.
#  (2) EIGENSOLVE DRIFT. Independent eigh() on near-degenerate spectra can
#      return different Ritz vectors on different ranks (different vendor
#      LAPACK, or merely different reduction order), after which the ranks
#      optimise different objectives and the run silently diverges. Rank 0
#      solves; v is broadcast. m doubles, once per iteration.
#  (3) PARAMETER DRIFT. Identical seeds are not enough — grow() consumes RNG,
#      and a warm start may not. Dist.sync_module() is called after init,
#      after every grow, and after every load_state_dict of a snapshot.
# ===========================================================================

class Dist:
    """Process-group wrapper. world == 1 makes every method a no-op, so the
    single-GPU path is byte-for-byte the original code path."""

    def __init__(self) -> None:
        self.rank, self.world, self.local_rank = 0, 1, 0
        self.backend = None
        self.enabled = False

    # ---- lifecycle ---------------------------------------------------------
    def init(self, backend: str = "auto", device_hint: str = "auto",
             oversubscribe: bool = False, timeout_minutes: float = 30.0) -> None:
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
            return                      # launched plainly: stay single-process
        world = int(os.environ["WORLD_SIZE"])
        if world <= 1:
            return
        self.rank = int(os.environ["RANK"])
        self.world = world
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank))
        want_cuda = torch.cuda.is_available() and device_hint != "cpu"
        if backend == "auto":
            # NCCL does not exist on Windows; gloo does, and it handles CUDA
            # tensors. Our payloads are O(m^2) doubles per iteration, so the
            # gloo penalty is irrelevant here.
            if (want_cuda and sys.platform != "win32"
                    and tdist.is_nccl_available()):
                backend = "nccl"
            else:
                backend = "gloo"

        # ---- device-count preflight, BEFORE init_process_group -------------
        # Asking torchrun for more processes than there are GPUs used to fail
        # deep inside torch.cuda.set_device with a bare "invalid device
        # ordinal", on the surplus ranks only. The ranks that DID have a GPU
        # then ran on — through the banner and the whole g==1 preflight, which
        # touches no collective because m=1 puts the plan in replica mode —
        # and only blocked at the first real collective, by which point
        # torchrun had SIGTERM'd them and the traceback that mattered was
        # buried under two pages of launcher stack. Check it up front, on
        # every rank, and say what to do about it.
        if want_cuda:
            ndev = torch.cuda.device_count()
            if ndev == 0:
                raise RuntimeError(
                    "distributed launch requested a CUDA backend but this "
                    "process sees no CUDA devices. Use --device cpu, or check "
                    "CUDA_VISIBLE_DEVICES.")
            if self.local_rank >= ndev:
                if not oversubscribe:
                    raise RuntimeError(
                        f"--nproc_per_node is larger than the number of "
                        f"visible GPUs: local_rank {self.local_rank} needs "
                        f"cuda:{self.local_rank} but only {ndev} device(s) "
                        f"exist ({', '.join(torch.cuda.get_device_name(i) for i in range(ndev))}).\n"
                        f"  Relaunch with --nproc_per_node={ndev}.\n"
                        f"  Note that oversubscribing GPUs would defeat the "
                        f"purpose anyway: sharding pairs is a VRAM measure, "
                        f"and two ranks on one card hold the same total "
                        f"activation memory as one rank holding every pair, "
                        f"plus the communication. If you only want to smoke-"
                        f"test the distributed path on one card, pass "
                        f"--dist-oversubscribe (or --device cpu).")
                self.local_rank = self.local_rank % ndev
        if backend == "nccl":
            torch.cuda.set_device(self.local_rank)
        tdist.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=float(timeout_minutes)),
        )
        self.backend = backend
        self.enabled = True

    def shutdown(self, *, graceful: bool = True) -> None:
        """Tear down the process group.

        A barrier is safe only on the normal path. During exception handling
        another rank may still be blocked in a different collective; entering
        a fresh barrier there can deadlock permanently.
        """
        if not self.enabled or not tdist.is_initialized():
            return
        try:
            if graceful:
                tdist.barrier()
        finally:
            try:
                tdist.destroy_process_group()
            finally:
                self.enabled = False

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.enabled:
            tdist.barrier()

    # ---- collectives -------------------------------------------------------
    def _dev(self, t: torch.Tensor) -> torch.Tensor:
        """gloo cannot reduce on some CUDA builds; bounce through CPU when the
        backend is gloo and the tensor is not already on CPU."""
        return t

    def all_reduce_(self, t: torch.Tensor, op="sum") -> torch.Tensor:
        if not self.enabled:
            return t
        o = {"sum": tdist.ReduceOp.SUM, "max": tdist.ReduceOp.MAX,
             "min": tdist.ReduceOp.MIN}[op]
        if self.backend == "gloo" and t.is_cuda:
            c = t.cpu()
            tdist.all_reduce(c, op=o)
            t.copy_(c)
        else:
            tdist.all_reduce(t, op=o)
        return t

    def broadcast_(self, t: torch.Tensor, src: int = 0) -> torch.Tensor:
        if not self.enabled:
            return t
        if self.backend == "gloo" and t.is_cuda:
            c = t.cpu()
            tdist.broadcast(c, src=src)
            t.copy_(c)
        else:
            tdist.broadcast(t, src=src)
        return t

    def all_gather_objects(self, obj):
        """Gather small Python metadata objects on every rank."""
        if not self.enabled:
            return [obj]
        out = [None for _ in range(self.world)]
        tdist.all_gather_object(out, obj)
        return out

    def sum_floats(self, vals, device, dtype=torch.float64):
        """All-reduce a short list of scalars; returns Python floats that are
        bitwise identical on every rank."""
        t = torch.stack([v.detach().reshape(()) if torch.is_tensor(v)
                         else torch.tensor(float(v), device=device, dtype=dtype)
                         for v in vals]).to(device=device, dtype=dtype)
        self.all_reduce_(t, "sum")
        return [float(x) for x in t]

    def agree(self, ok: bool, device) -> bool:
        """Unanimous verdict. Any rank voting False makes every rank False.
        This is what stops a per-rank ValidationError from deadlocking the
        job at the next collective."""
        if not self.enabled:
            return ok
        t = torch.tensor([0.0 if ok else 1.0], device=device,
                         dtype=torch.float64)
        self.all_reduce_(t, "max")
        return float(t[0]) == 0.0

    def sync_module(self, module: nn.Module, src: int = 0) -> None:
        """Force bitwise-identical parameters. Cheap insurance against RNG
        divergence in grow() / warm start / snapshot restore."""
        if not self.enabled:
            return
        with torch.no_grad():
            for p in module.parameters():
                self.broadcast_(p.data, src)
            for b in module.buffers():
                self.broadcast_(b.data, src)

    def sync_grads(self, module: nn.Module, mode: str) -> None:
        """One flattened all-reduce of dL/dtheta.

        mode 'shard'   : each rank held a DISJOINT subset of pairs, so the
                         true gradient is the SUM of the local ones.
        mode 'replica' : every rank computed the whole thing redundantly
                         (m too small to shard); averaging is the identity
                         but keeps the ranks bitwise locked together.
        """
        if not self.enabled:
            return
        params = [p for p in module.parameters() if p.requires_grad]
        if not params:
            return
        for p in params:
            if p.grad is None:            # keep the flat layout identical on
                p.grad = torch.zeros_like(p)   # every rank, always
        flat = torch.cat([p.grad.reshape(-1) for p in params])
        self.all_reduce_(flat, "sum")
        if mode == "replica":
            flat /= self.world
        off = 0
        with torch.no_grad():
            for p in params:
                n = p.grad.numel()
                p.grad.copy_(flat[off:off + n].view_as(p.grad))
                off += n


DIST = Dist()

_builtin_print = print


def print(*a, **kw):                                        # noqa: A001
    """Module-wide shadow: only rank 0 writes to stdout. Shadowing the builtin
    at module scope keeps all ~120 existing call sites untouched."""
    if DIST.is_main:
        _builtin_print(*a, **kw)


def print_all(*a, **kw):
    """Rank-tagged print, for the rare message that must come from every rank
    (fatal errors, self-test output). Serialised by a barrier-and-order pass
    so the ranks do not interleave mid-line on a shared terminal."""
    for r in range(DIST.world):
        if r == DIST.rank:
            _builtin_print(f"[rank {DIST.rank}]", *a, flush=True, **kw)
        DIST.barrier()


@dataclass
class PairPlan:
    """Which pairs this rank owns for a given m, and how to combine.

    mode 'shard'   : disjoint round-robin subsets, results must be reduced.
    mode 'replica' : every rank owns every pair. Used when P < world (the
                     preflight g==1 has P = 1, the dictionary baseline P = 78)
                     so no rank is ever left with an empty pair set, which
                     would make the local tensors degenerate and the reduces
                     ill-defined.
    """
    mode: str
    m: int
    P_global: int
    idx: torch.Tensor        # (P_loc,) global pair index
    iu0: torch.Tensor        # (P_loc,) first channel of each local pair
    iu1: torch.Tensor        # (P_loc,) second channel


# ---------------------------------------------------------------------------
