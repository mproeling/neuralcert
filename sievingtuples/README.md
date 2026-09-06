# Admissible tuples for the M_k thresholds

Three programs, covering the three regimes where k values fall into.

```
g++ -O3 -march=native -fopenmp -std=c++17 admissible_hybrid.cpp -o admissible_hybrid
g++ -O3 -march=native -fopenmp -std=c++17 verify_tuple.cpp     -o verify_tuple
g++ -O3 -march=native            -std=c++17 zhang_bound.cpp    -o zhang_bound
```

`-fopenmp` is optional; without it everything still runs correctly, single-threaded.

```
In Mac use homebrew: brew install llvm libomp 
Followed by:
/opt/homebrew/opt/llvm/bin/clang++ \
  -O3 -std=c++17 \
  -fopenmp \
  admissible_hybrid.cpp \ 
  -o admissible_hybrid 
```
---

## Validation performed

Everything below was run on **one core, `-O2`, no OpenMP**, so your home machine
will be several times faster.

| check | expected | got |
|---|---|---|
| `zhang_bound --k 3500000` vs Polymath8 "first k primes past k" row | 59,874,594 | **59,874,594** exact match |
| `zhang_bound --k 35265` vs independent Python sieve | 432,230 | **432,230** exact match |
| Dusart closed form vs exact enumeration at k=12.5M | — | agree to **0.016 %** |
| `admissible_hybrid --k 632` vs published greedy–Schinzel row (4,710) | ≤4,710 | **4,700** (record is 4,680) |
| `admissible_hybrid --k 5000` vs published greedy–Schinzel row (46,968) | ≈46,968 | 47,070 (0.22 % behind; coarse sweep) |
| `verify_tuple` on a gap permutation that preserves the diameter | reject | rejected, 8 witness failures |
| `FastMod` Barrett reduction vs hardware `%`, 200k random cases | match | match (runs at every startup) |

---

## k = 220,000 — `admissible_hybrid`

Already done, certificate included: **H(220000) ≤ 2,879,306**
(ratio to k log k + k = 0.9839; verified independently).

Reproduce, or push further:

```bash
# 1. coarse shift sweep over [-k log k, 2 k log k], no verification
./admissible_hybrid --k 220000 --scan -2700000:5400000:405000     # ~67 s / 1 core

# 2. refine around the winner
./admissible_hybrid --k 220000 --scan -2100000:-1700000:25000     # ~61 s

# 3. full run at the best shift, with local optimization
./admissible_hybrid --k 220000 --shift -1775000 --local-rounds 2000 \
                    --cert tuple_220000.cert

# 4. independent check
./verify_tuple tuple_220000.cert
```

## k = 12,500,000 — either tool

`zhang_bound` provides a rigorous number in under a second:

```bash
./zhang_bound --k 12500000 --mode both
```
```
dusart : H(12500000) <= 230,572,609     (closed form, no enumeration)
exact  : H(12500000) <= 230,535,386     p in [12500003, 243035389]
         ratio to k log k + k : 1.0635
```

# longer more competitive run 
```bash
./admissible_hybrid --k 12500000 --x 228000000 --shift 0 \
                    --local-rounds 0 --threads 8 --cert tuple_12500000.cert
```
Expect greedy phase to dominate: π(k) \approx 809,000 primes, survivor vector
\approx 13 M `u32`, and the adaptive stop should land near p \approx 10^6. Extrapolating
the k \approx 220,000 timing, expect several hours single-core. 
Perform `--scan` at coarse step first — a single scan point is much
cheaper than a full run, since it skips verification.

If it reports `FAIL: only N survivors`, raise `--x` by the suggested factor;
the window is the one parameter that can make it fail outright.

---

## k = 1,200,000,000 — `zhang_bound` only

Window sieving is out of reach here (the window alone is 2.9·10^10 integers).
The Zhang construction is not:

```bash
./zhang_bound --k 1200000000 --mode both       # 59 s on one core
```
```
dusart : H(1200000000) <= 27,849,469,966
exact  : H(1200000000) <= 27,846,420,672      p in [1200000041, 29046420713]
         ratio to k log k + k : 1.0593
```

The certificate here is structural rather than enumerated:
every element is a prime exceeding k, so residue class 0 mod p is
empty for every p ≤ k, and for p > k a k-element set cannot meet all p classes.

---

## Certificate format

```
format=admissible-tuple-certificate/1
k=220000
base=-1582432                 # absolute value of h_1
diameter=2879306
prime_count=19618
BEGIN_GAPS                    # h[i]-h[i-1], h[0]=0, k-1 values
2 6 4 2 4 ...
END_GAPS
BEGIN_WITNESSES               # one line per prime p <= k
2 1
3 2
...
END_WITNESSES
```

Witnesses are derived from the **final tuple**, not from the sieve trajectory,
so the certificate is independent of how the tuple was found and survives any
future local-optimization or discovery step you bolt on the front.

`verify_tuple` shares no code path with the constructor: it regenerates the
primes itself, rebuilds the tuple from the gaps, confirms the witness list is
*exactly* the primes ≤ k (so a certificate cannot pass by omitting an
inconvenient prime), and re-derives each residue by direct enumeration.

---

## Notes on the implementation, relative to `greedy_admissible_v3`

- **Schinzel presieve before the greedy phase.** Delete n ≡ 1 (mod p) for
  p ≤ y (y = 2), then n ≡ 0 (mod p) for y < p ≤ z = √x. Greedy runs only for
  p > z. This is where most of the gain over pure greedy comes from.
- **The shift is swept**, not fixed. `--scan LO:HI:STEP` over
  [−k log k, 2 k log k].
- **Adaptive stop.** Once a prime already has an empty residue class, deleting
  it is a no-op and every later greedy decision is unchanged, so the loop skips
  it. After `--zero-run` consecutive no-ops the greedy phase stops entirely.
  Correctness does not depend on where it stops: step [4] rechecks every prime
  p ≤ k from scratch, and any prime that fails is forced back into the greedy
  set and the pipeline reruns. That repair loop fired zero times in every test.
- **Two exact verification strategies**, selected per prime by cost model:
  counting (O(k)) for small p, and residue probing against a tuple bitmap with
  early exit for large p, where empty classes are plentiful. The crossover sits
  near p ≈ k/13. Without this, verification at k = 12.5 M is a 12-hour job.
- **Barrett reduction** instead of hardware `%`, self-tested against `%` at
  every startup.
- **`double`, not `long double`**, and the end-weight table is precomputed once
  over window positions rather than recomputed per prime per survivor.
- **Buffers hoisted** with touched-list resets, so the per-prime cost is O(N)
  and not O(N + p).
