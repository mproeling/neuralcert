"""Problem-agnostic deterministic multimodular (CRT) certification.

THE PATTERN THIS ABSTRACTS
==========================
A large family of variational-bound certificates has the same shape:

  * the certificate value is a finite set of RATIONALS  V_f = S_f / D_f,
    f ranging over a few named "functionals" (for Maynard: the two quadratic
    forms c^T A c and c^T B c);
  * each S_f is an INTEGER that is astronomically large but never needs to
    be formed over Z -- it is a fixed algebraic expression in integer input
    data, so it can be evaluated in Z/pZ for word-size p;
  * each D_f is a known integer, computable directly;
  * a RIGOROUS A PRIORI BOUND  |S_f| <= BND_f  is available from
    submultiplicativity of the inputs.

Then S_f is recovered exactly by centered CRT over enough primes, with no
floats, no rational reconstruction, and no probabilistic step anywhere.

To port a new problem you implement `ModularProblem` -- five methods, none
of which knows anything about primes, parallelism, checkpointing, or
reconstruction.  Everything below that line is shared.

WHAT THE ENGINE GUARANTEES
==========================
1. prod(reconstruction primes) > 4 * max_f BND_f  (checked on the actual
   product, not estimated);
2. centered representatives are re-checked against BND_f, so an arithmetic
   bug surfaces as an exception rather than a wrong theorem;
3. held-out primes, excluded from the reconstruction, independently re-check
   every reconstructed integer;
4. checkpoints are bound to a structural fingerprint of the payload, so
   residues computed for different inputs can never be silently reused.

Guarantees 2-4 are implementation sanity tests.  Guarantee 1 is the
mathematics: given it, the centered representative EQUALS the true integer.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from fractions import Fraction

from .fingerprint import Fingerprint
from .primes import gen_primes_for_bound


# ---------------------------------------------------------------------------
# The plug-in interface
# ---------------------------------------------------------------------------
class ModularProblem(ABC):
    """One certification problem, expressed only in Z/pZ.

    Implementations must be PICKLABLE (they are shipped to worker processes
    once, via the pool initializer) and must be pure: `residues(p)` may not
    mutate observable state, because primes are evaluated out of order and
    possibly concurrently.
    """

    #: short identifier, written into checkpoints and certificates
    name: str = "unnamed"

    @abstractmethod
    def functionals(self) -> tuple[str, ...]:
        """Names of the integers to reconstruct, in a fixed order."""

    @abstractmethod
    def a_priori_bounds(self) -> Mapping[str, int]:
        """Rigorous |S_f| <= BND_f for every functional.

        These must be PROVEN bounds derived from the integer inputs (l1
        submultiplicativity, factorial ratios, ...), never measured or
        guessed.  The engine's correctness rests entirely on them.
        """

    @abstractmethod
    def denominators(self) -> Mapping[str, int]:
        """Known positive integer D_f with V_f = S_f / D_f."""

    @abstractmethod
    def residues(self, p: int) -> Mapping[str, int]:
        """S_f mod p for every functional, computed entirely in Z/pZ.

        This is where all the per-problem work lives.  Any per-prime tables
        that are shared across sub-terms should be built once here and
        reused, not rebuilt per sub-term.
        """

    @abstractmethod
    def fingerprint_payload(self):
        """Every datum that affects the residues, as ints/strs/sequences.

        Anything omitted here can silently invalidate a resumed run.  When
        in doubt, include it.
        """

    # -- optional hooks ---------------------------------------------------
    def prepare_worker(self) -> None:
        """Called once per worker process before any `residues` call.

        Use for derived state that is expensive to build but cheap to
        rebuild, so it need not be serialized.
        """

    def describe(self) -> str:
        return self.name


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------
def crt_combine(residues: list[int], primes: list[int], bound: int) -> int:
    """Deterministic centered CRT with an a priori bound check.

    Incremental Garner-style lifting; asserts prod(primes) > 2*bound so the
    centered representative is forced to be the true integer, then verifies
    the result against the bound as an arithmetic sanity check.
    """
    if len(residues) != len(primes):
        raise ValueError("residue/prime count mismatch")
    x, M = 0, 1
    for r, p in zip(residues, primes):
        inv = pow(M % p, p - 2, p)
        t = (r - x) % p * inv % p
        x += M * t
        M *= p
    if M <= 2 * bound:
        raise RuntimeError("CRT modulus does not exceed 2*bound -- "
                           "prime budget insufficient (bug)")
    if x > M // 2:
        x -= M
    if abs(x) > bound:
        raise RuntimeError("reconstructed integer violates the a priori "
                           "bound -- arithmetic bug")
    return x


# ---------------------------------------------------------------------------
# Worker plumbing (module-level for multiprocessing picklability)
# ---------------------------------------------------------------------------
_G_PROBLEM: ModularProblem | None = None
_G_KEYS: tuple[str, ...] = ()


def _worker_init(problem: ModularProblem, keys: tuple[str, ...]) -> None:
    global _G_PROBLEM, _G_KEYS
    _G_PROBLEM, _G_KEYS = problem, keys
    problem.prepare_worker()


def _worker_chunk(primes: list[int]) -> list[tuple[int, tuple[int, ...]]]:
    prob, keys = _G_PROBLEM, _G_KEYS
    out = []
    for p in primes:
        res = prob.residues(p)
        out.append((p, tuple(int(res[k]) % p for k in keys)))
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
class MultiModularCertifier:
    """Exact reconstruction of a ModularProblem's functionals.

    Usage:
        cert = MultiModularCertifier(problem, verify_primes=2)
        ints = cert.run(jobs=16, resume_path="run.residues")
        vals = cert.exact_values()          # dict[str, Fraction]
    """

    def __init__(self, problem: ModularProblem, verify_primes: int = 2
                 ) -> None:
        self.problem = problem
        self.keys = tuple(problem.functionals())
        if not self.keys:
            raise ValueError("problem declares no functionals")
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("functional names must be distinct")

        self.bounds = {k: int(problem.a_priori_bounds()[k]) for k in self.keys}
        self.dens = {k: int(problem.denominators()[k]) for k in self.keys}
        for k in self.keys:
            if self.bounds[k] < 0:
                raise ValueError(f"a priori bound for {k!r} is negative")
            if self.dens[k] == 0:
                raise ValueError(f"denominator for {k!r} is zero")

        self.verify_primes = max(0, int(verify_primes))
        self.primes, self.n_rec = gen_primes_for_bound(
            max(self.bounds.values()), extra=self.verify_primes)
        self._ints: dict[str, int] | None = None

    # -- fingerprint ------------------------------------------------------
    def fingerprint(self, chars: int = 32) -> str:
        fp = Fingerprint("certkit.crt.v1")
        fp.update(self.problem.name)
        fp.update(list(self.keys))
        fp.update([self.bounds[k] for k in self.keys])
        fp.update([self.dens[k] for k in self.keys])
        fp.update(self.problem.fingerprint_payload())
        return fp.hexdigest(chars)

    # -- checkpoint I/O ---------------------------------------------------
    def _header(self) -> str:
        return (f"# certkit-residues v1 {self.problem.name} "
                f"{self.fingerprint()} {' '.join(self.keys)}")

    def _load_resume(self, path: str, verbose: bool
                     ) -> dict[int, tuple[int, ...]]:
        results: dict[int, tuple[int, ...]] = {}
        if not os.path.exists(path):
            return results
        want = self._header()
        with open(path) as fh:
            first = fh.readline().strip()
            if first != want:
                raise SystemExit(
                    f"    resume file {path} was written for a DIFFERENT "
                    f"payload; its residues are invalid here. Delete or "
                    f"rename it.\n      expected: {want}\n      found:    "
                    f"{first}")
            for line in fh:
                parts = line.split()
                if len(parts) == 1 + len(self.keys):
                    results[int(parts[0])] = tuple(int(v) for v in parts[1:])
        pool = set(self.primes)
        results = {p: v for p, v in results.items() if p in pool}
        if verbose and results:
            print(f"    resume: {len(results)}/{len(self.primes)} prime "
                  f"residues loaded from {path}")
        return results

    # -- main sweep -------------------------------------------------------
    def run(self, jobs: int = 1, chunk: int = 4, verbose: bool = True,
            resume_path: str | None = None) -> dict[str, int]:
        t0 = time.time()
        results: dict[int, tuple[int, ...]] = {}
        fh = None
        if resume_path:
            results = self._load_resume(resume_path, verbose)
            fresh = not os.path.exists(resume_path)
            fh = open(resume_path, "a", buffering=1)
            if fresh:
                fh.write(self._header() + "\n")

        todo = [p for p in self.primes if p not in results]
        chunks = [todo[i:i + chunk] for i in range(0, len(todo), chunk)]

        def absorb(rows):
            for p, vals in rows:
                results[p] = vals
                if fh is not None:
                    fh.write(f"{p} {' '.join(str(v) for v in vals)}\n")
            if verbose and len(results) % 32 < chunk:
                self._progress(len(results), t0)

        try:
            if jobs <= 1:
                _worker_init(self.problem, self.keys)
                for rows in map(_worker_chunk, chunks):
                    absorb(rows)
            else:
                import multiprocessing as mp
                with mp.Pool(jobs, initializer=_worker_init,
                             initargs=(self.problem, self.keys)) as pool:
                    for rows in pool.imap_unordered(_worker_chunk, chunks):
                        absorb(rows)
        finally:
            if fh is not None:
                fh.close()
        if verbose:
            self._progress(len(results), t0, final=True)

        rec_p = self.primes[:self.n_rec]
        ints: dict[str, int] = {}
        for i, key in enumerate(self.keys):
            ints[key] = crt_combine([results[p][i] for p in rec_p], rec_p,
                                    self.bounds[key])
        for p in self.primes[self.n_rec:]:
            for i, key in enumerate(self.keys):
                if ints[key] % p != results[p][i]:
                    raise RuntimeError(
                        f"held-out prime {p} disagrees with the "
                        f"reconstruction of {key!r} -- arithmetic bug")
        self._ints = ints
        return ints

    def _progress(self, done: int, t0: float, final: bool = False) -> None:
        print(f"    CRT primes {done:6d}/{len(self.primes)} "
              f"[{time.time() - t0:8.1f}s]" + ("  (done)" if final else ""),
              flush=True)

    # -- results ----------------------------------------------------------
    def exact_values(self) -> dict[str, Fraction]:
        if self._ints is None:
            raise RuntimeError("run() has not completed")
        return {k: Fraction(self._ints[k], self.dens[k]) for k in self.keys}

    def summary(self) -> str:
        n_bits = max(b.bit_length() for b in self.bounds.values())
        return (f"{self.problem.describe()}: {len(self.keys)} functionals, "
                f"a priori bound {n_bits} bits, "
                f"{self.n_rec} reconstruction primes "
                f"+ {len(self.primes) - self.n_rec} held out")
