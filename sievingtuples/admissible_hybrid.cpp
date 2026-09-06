// admissible_hybrid.cpp
//
// Narrow admissible k-tuple construction: shifted-Schinzel presieve, greedy
// phase, exact full verification, local optimization, compact certificate.
//
// Pipeline
//   [0] window W = [s, s+x] of integers.
//   [1] Schinzel presieve with parameters (y, z):
//         delete n = 1 (mod p) for p <= y            (y = 2 by default)
//         delete n = 0 (mod p) for y < p <= z        (z = sqrt(x) by default)
//       Shifting the WINDOW is what the "shifted" in shifted Schinzel means;
//       the sieved residue classes are stated in absolute integers, so no
//       remapping is needed -- s is the free parameter.
//   [2] greedy phase for z < p <= P: delete a least-populated residue class
//       among current survivors.  P is adaptive: we stop after --zero-run
//       consecutive primes for which some class is already empty (deleting
//       nothing).  Correctness does not depend on where we stop -- step [4]
//       checks every prime p <= k from scratch.
//   [3] extract the NARROWEST block of k consecutive survivors.
//   [4] exact verification for every prime p <= k: find a residue class
//       missed by the k-tuple.  Two exact strategies, chosen per prime:
//         count : tally all k residues, O(k) per prime.  Used for small p.
//         probe : scan residues r = 0,1,2,... testing class emptiness against
//                 a bitmap of the tuple, exiting a class at its first hit.
//                 Used for large p, where empty classes are plentiful.
//       The crossover is computed from the cost models, see pick_strategy().
//       If some prime has NO empty class the tuple is inadmissible: that
//       prime is added to a forced list and [2]-[4] rerun.  (In practice
//       this fires zero or one time.)
//   [5] local optimization: repeat
//         contract : find v strictly inside (h_1, h_k), v not in H, with
//                    H + {v} admissible; add it and drop the endpoint that
//                    shrinks the diameter more.  Strictly narrows.
//         shift    : drop h_k, add the largest legal v < h_1 (or mirror).
//                    Diameter may grow; kept only when it does not.
//       Soundness of contract: if H + {v} is admissible then so is every
//       subset, in particular (H \ {endpoint}) + {v}.  So testing against
//       the critical classes of H (primes missing exactly one class) is a
//       sufficient condition.  It is conservative, never wrong.
//   [6] certificate: normalized offsets + one witness residue per prime
//       p <= k, derived from the final tuple, independent of how it was
//       found.  Verified by verify_tuple.
//
// Build:
//   g++ -O3 -march=native -fopenmp -std=c++17 admissible_hybrid.cpp -o admissible_hybrid
//   (drop -fopenmp if unavailable; single-threaded still correct)

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using u64 = std::uint64_t;
using u32 = std::uint32_t;
using i64 = std::int64_t;

// ---------------------------------------------------------------- fast mod --
// Barrett-style reduction for a 32-bit modulus applied to values < 2^40.
// Validated against hardware % at startup (self_test_fastmod).
struct FastMod {
    u64 p, m;
    FastMod() : p(1), m(0) {}
    explicit FastMod(u64 p_) : p(p_) {
        m = (u64)((((__uint128_t)1) << 64) / p);
    }
    inline u64 mod(u64 v) const {
        u64 q = (u64)(((__uint128_t)m * v) >> 64);
        u64 r = v - q * p;
        if (r >= p) r -= p;
        if (r >= p) r -= p;
        return r;
    }
};

static void self_test_fastmod() {
    std::mt19937_64 rng(12345);
    for (int t = 0; t < 200000; ++t) {
        u64 p = (rng() % 4000000000ULL) + 2;
        FastMod fm(p);
        u64 v = rng() % (1ULL << 40);
        if (fm.mod(v) != v % p) {
            std::fprintf(stderr, "FATAL: FastMod self-test failed p=%llu v=%llu\n",
                         (unsigned long long)p, (unsigned long long)v);
            std::exit(3);
        }
    }
}

// ------------------------------------------------------------------ primes --
static std::vector<u32> primes_upto(u64 n) {
    std::vector<u32> ps;
    if (n < 2) return ps;
    ps.push_back(2);
    if (n < 3) return ps;
    u64 m = (n - 1) / 2;                       // odd numbers 3,5,7,...
    std::vector<unsigned char> comp(m, 0);
    for (u64 i = 0; i < m; ++i) {
        u64 p = 2 * i + 3;
        if (comp[i]) continue;
        ps.push_back((u32)p);
        if (p > n / p) continue;
        for (u64 j = (p * p - 3) / 2; j < m; j += p) comp[j] = 1;
    }
    return ps;
}

// ------------------------------------------------------------------- args ---
struct Args {
    u64 k = 0;
    i64 shift = 0;
    u64 x = 0;                 // window length; 0 => auto
    u64 y = 2;                 // Schinzel small-prime parameter
    u64 z = 0;                 // Schinzel presieve bound; 0 => sqrt(x)
    std::string tie = "down";  // down | up | center | ends
    u64 zero_run = 400;        // stop greedy after this many consecutive no-ops
    int  local_rounds = 200;
    bool scan_only = false;    // shift sweep: skip verification and local opt
    i64  scan_lo = 0, scan_hi = 0, scan_step = 0;
    std::string cert = "tuple.cert";
    int threads = 0;
};

static u64 U(const std::string& s) {
    size_t z = 0; auto v = std::stoull(s, &z);
    if (z != s.size()) throw std::runtime_error("bad unsigned integer: " + s);
    return (u64)v;
}
static i64 I(const std::string& s) {
    size_t z = 0; auto v = std::stoll(s, &z);
    if (z != s.size()) throw std::runtime_error("bad integer: " + s);
    return (i64)v;
}

static void usage() {
    std::cout <<
    "Usage: admissible_hybrid --k K [options]\n"
    "  --k K              tuple size (required)\n"
    "  --x LEN            window length (default: 1.05*(k log k + k))\n"
    "  --shift S          window start, may be negative (default 0)\n"
    "  --y Y              Schinzel small-prime bound (default 2)\n"
    "  --z Z              Schinzel presieve bound (default floor(sqrt(x)))\n"
    "  --tie down|up|center|ends   tie-break among least-populated classes\n"
    "  --zero-run N       stop greedy after N consecutive no-op primes (400)\n"
    "  --local-rounds N   local optimization rounds (200; 0 disables)\n"
    "  --scan LO:HI:STEP  sweep shift, report diameters only, no cert\n"
    "  --cert FILE        certificate path (tuple.cert)\n"
    "  --threads N        OpenMP threads (default: all)\n";
}

static Args parse(int ac, char** av) {
    Args a;
    for (int i = 1; i < ac; ++i) {
        std::string t = av[i];
        auto need = [&](const char* n) {
            if (i + 1 >= ac) throw std::runtime_error(std::string("missing value for ") + n);
            return std::string(av[++i]);
        };
        if (t == "--k") a.k = U(need("--k"));
        else if (t == "--x") a.x = U(need("--x"));
        else if (t == "--shift") a.shift = I(need("--shift"));
        else if (t == "--y") a.y = U(need("--y"));
        else if (t == "--z") a.z = U(need("--z"));
        else if (t == "--tie") a.tie = need("--tie");
        else if (t == "--zero-run") a.zero_run = U(need("--zero-run"));
        else if (t == "--local-rounds") a.local_rounds = (int)U(need("--local-rounds"));
        else if (t == "--cert") a.cert = need("--cert");
        else if (t == "--threads") a.threads = (int)U(need("--threads"));
        else if (t == "--scan") {
            std::string v = need("--scan");
            size_t c1 = v.find(':'), c2 = v.rfind(':');
            if (c1 == std::string::npos || c1 == c2) throw std::runtime_error("--scan needs LO:HI:STEP");
            a.scan_lo = I(v.substr(0, c1));
            a.scan_hi = I(v.substr(c1 + 1, c2 - c1 - 1));
            a.scan_step = I(v.substr(c2 + 1));
            if (a.scan_step <= 0) throw std::runtime_error("--scan STEP must be positive");
            a.scan_only = true;
        }
        else if (t == "-h" || t == "--help") { usage(); std::exit(0); }
        else throw std::runtime_error("unknown argument " + t);
    }
    if (a.k < 2) throw std::runtime_error("--k must be >= 2");
    if (a.tie != "down" && a.tie != "up" && a.tie != "center" && a.tie != "ends")
        throw std::runtime_error("--tie must be down|up|center|ends");
    if (a.x == 0) {
        double kk = (double)a.k;
        a.x = (u64)(1.05 * (kk * std::log(kk) + kk)) + 64;
    }
    if (a.z == 0) a.z = (u64)std::sqrt((double)a.x);
    if (a.z < a.y) a.z = a.y;
    return a;
}

// ------------------------------------------------------------- bit helpers --
struct Bitmap {
    std::vector<u64> w;
    u64 n = 0;
    void reset(u64 bits, bool one) {
        n = bits;
        w.assign((bits + 63) / 64, one ? ~0ULL : 0ULL);
        if (one && (bits & 63)) w.back() = (~0ULL) >> (64 - (bits & 63));
    }
    inline bool get(u64 i) const { return (w[i >> 6] >> (i & 63)) & 1ULL; }
    inline void clr(u64 i) { w[i >> 6] &= ~(1ULL << (i & 63)); }
    inline void set(u64 i) { w[i >> 6] |= (1ULL << (i & 63)); }
};

// nonneg residue of a signed value mod p
static inline u64 nmod(i64 v, u64 p) {
    i64 r = v % (i64)p;
    if (r < 0) r += (i64)p;
    return (u64)r;
}

// ---------------------------------------------------------------- presieve --
// Delete n = 1 (mod p) for p <= y, n = 0 (mod p) for y < p <= z, from
// the window [s, s+x].  Returns survivor offsets relative to s.
static std::vector<u32> presieve(i64 s, u64 x, u64 y, u64 z,
                                 const std::vector<u32>& small_primes) {
    Bitmap alive; alive.reset(x + 1, true);
    for (u32 p : small_primes) {
        if (p > z) break;
        u64 target = (p <= y) ? 1 : 0;         // absolute residue class to delete
        // first index i with (s + i) = target (mod p)
        u64 i0 = nmod((i64)target - s, p);
        for (u64 i = i0; i <= x; i += p) alive.clr(i);
    }
    std::vector<u32> out;
    out.reserve((size_t)(x / 8));
    for (u64 i = 0; i <= x; ++i) if (alive.get(i)) out.push_back((u32)i);
    return out;
}

// ------------------------------------------------------------------ greedy --
struct GreedyStats { u64 last_prime = 0; u64 deleted = 0; };

// Greedy over primes in (z, ...], plus any primes in `forced` regardless of
// where the adaptive stop would have landed.  Survivors are offsets from s.
static GreedyStats greedy_phase(i64 s, std::vector<u32>& surv,
                                const std::vector<u32>& primes,
                                u64 z, u64 k, u64 zero_run,
                                const std::vector<u32>& forced,
                                const std::string& tie, u64 x,
                                bool verbose) {
    GreedyStats st;
    u64 maxp = primes.empty() ? 2 : primes.back();
    std::vector<u32> cnt((size_t)maxp + 1, 0);
    std::vector<u32> touched;
    touched.reserve(1 << 20);

    std::vector<double> wt;                    // only for --tie ends
    if (tie == "ends") {
        wt.resize((size_t)x + 1);
        for (u64 i = 0; i <= x; ++i) {
            u64 d = std::min(i, x - i);
            wt[i] = 1.0 / (1.0 + (double)d);
        }
    }
    std::vector<double> score;
    if (tie == "ends") score.assign((size_t)maxp + 1, 0.0);

    u64 zeros = 0;
    size_t fi = 0;
    for (u32 p : primes) {
        if (p <= z) continue;
        if (p > k) break;
        bool is_forced = false;
        while (fi < forced.size() && forced[fi] < p) ++fi;
        if (fi < forced.size() && forced[fi] == p) is_forced = true;

        if (zeros >= zero_run && !is_forced) continue;   // adaptive stop

        FastMod fm(p);
        u64 base = nmod(s, p);
        touched.clear();
        for (u32 off : surv) {
            u64 r = fm.mod(base + (u64)off);
            if (cnt[r]++ == 0) touched.push_back((u32)r);
        }

        if (touched.size() < (size_t)p) {
            // Some class is already empty: deleting it removes nothing.  The
            // survivor set, and hence every later greedy decision, is
            // unchanged.  Skip.  Step [4] supplies the witness.
            for (u32 r : touched) cnt[r] = 0;
            ++zeros;
            st.last_prime = p;
            continue;
        }
        zeros = 0;

        // every class occupied: find a least-populated one
        u32 mn = cnt[0];
        for (u32 r = 1; r < p; ++r) mn = std::min(mn, cnt[r]);

        u32 best = 0; bool have = false;
        if (tie == "down") {
            for (u32 r = 0; r < p; ++r) if (cnt[r] == mn) { best = r; break; }
        } else if (tie == "up") {
            for (u32 r = p; r-- > 0;) if (cnt[r] == mn) { best = r; break; }
        } else if (tie == "center") {
            i64 mid = (i64)p / 2, bd = -1;
            for (u32 r = 0; r < p; ++r) if (cnt[r] == mn) {
                i64 d = std::llabs((i64)r - mid);
                if (!have || d < bd) { have = true; bd = d; best = r; }
            }
        } else { // ends: among minima, prefer the class most concentrated at
                 // the window ends, i.e. the one whose loss costs the middle
                 // the least.  Weight table is precomputed, one add per element.
            for (u32 r : touched) score[r] = 0.0;
            for (u32 off : surv) score[fm.mod(base + (u64)off)] += wt[off];
            double bs = -1.0;
            for (u32 r = 0; r < p; ++r) if (cnt[r] == mn) {
                if (!have || score[r] > bs) { have = true; bs = score[r]; best = r; }
            }
        }

        st.deleted += mn;
        st.last_prime = p;
        for (u32 r : touched) cnt[r] = 0;

        if (mn) {
            std::vector<u32> next;
            next.reserve(surv.size() - mn);
            for (u32 off : surv)
                if (fm.mod(base + (u64)off) != best) next.push_back(off);
            surv.swap(next);
        }
        if (verbose && (p < 200 || surv.size() < k + k / 10))
            std::fprintf(stderr, "  p=%u del=%u alive=%zu\n", p, mn, surv.size());
    }
    return st;
}

// ------------------------------------------------------- narrowest k-block --
struct Block { size_t idx; u64 diam; };
static Block narrowest(const std::vector<u32>& surv, u64 k) {
    Block b{0, ~0ULL};
    if (surv.size() < k) return b;
    for (size_t i = 0; i + k <= surv.size(); ++i) {
        u64 d = (u64)surv[i + k - 1] - (u64)surv[i];
        if (d < b.diam) { b.diam = d; b.idx = i; }
    }
    return b;
}

// ------------------------------------------------------------ verification --
// For the k-tuple H (offsets, sorted, h[0] == 0 after normalization) and every
// prime p <= k, find a residue class mod p containing no element of H.
//
// witness[i]  = missed residue for primes[i], or p (sentinel) if none exists
// n_missed[i] = 1 if exactly one class is missed, 2 if >= 2, 0 if none.
//               (needed for local optimization: only "exactly one" constrains)
struct Verdict {
    std::vector<u32> witness;
    std::vector<unsigned char> missed_cat;   // 0 none, 1 exactly one, 2 two-or-more
    std::vector<u32> failures;
};

static Verdict verify_tuple(const std::vector<u32>& H, i64 base,
                            const std::vector<u32>& primes, u64 k) {
    Verdict V;
    V.witness.assign(primes.size(), 0);
    V.missed_cat.assign(primes.size(), 0);

    const u64 D = H.back();                    // span
    Bitmap in; in.reset(D + 1, false);
    for (u32 h : H) in.set(h);

    // Cost model.  count: O(k).  probe: ~C*exp(k/p) + D/p, where C is the mean
    // number of bitmap hits before a class is shown occupied, ~ D/k.
    const double dens = (double)H.size() / (double)(D + 1);
    const double C = 1.0 / dens;
    auto use_probe = [&](u64 p) -> bool {
        double e = (double)k / (double)p;
        if (e > 40.0) return false;            // exp overflow-ish: counting wins
        double probe_cost = C * std::exp(e) + (double)D / (double)p;
        return probe_cost < (double)H.size();
    };

    std::vector<std::vector<u32>> local_fail;
    int nth = 1;
#ifdef _OPENMP
    nth = omp_get_max_threads();
#endif
    local_fail.resize(nth);

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        std::vector<u32> cnt;
        std::vector<u32> touched;
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 64)
#endif
        for (long long ii = 0; ii < (long long)primes.size(); ++ii) {
            u64 p = primes[ii];
            if (p > k) { V.witness[ii] = 0; V.missed_cat[ii] = 2; continue; }
            FastMod fm(p);
            u64 b0 = nmod(base, p);

            if (!use_probe(p)) {
                if (cnt.size() < p) cnt.assign(p, 0);
                touched.clear();
                for (u32 h : H) {
                    u64 r = fm.mod(b0 + (u64)h);
                    if (cnt[r]++ == 0) touched.push_back((u32)r);
                }
                for (u32 r : touched) cnt[r] = 0;
                u64 nempty = p - touched.size();
                if (nempty == 0) {
                    V.missed_cat[ii] = 0;
                    local_fail[tid].push_back((u32)p);
                } else {
                    // recover the smallest empty class
                    std::sort(touched.begin(), touched.end());
                    u32 w = 0;
                    for (u32 t : touched) { if (t != w) break; ++w; }
                    V.witness[ii] = w;
                    V.missed_cat[ii] = (nempty == 1) ? 1 : 2;
                }
            } else {
                // probe: scan classes, exit each at its first occupied slot
                u32 found1 = p, found2 = p;
                for (u64 r = 0; r < p; ++r) {
                    // class r of the ABSOLUTE integers = offsets o with
                    // (base + o) = r (mod p)
                    u64 o0 = nmod((i64)r - (i64)nmod(base, p), p);
                    bool occupied = false;
                    for (u64 o = o0; o <= D; o += p)
                        if (in.get(o)) { occupied = true; break; }
                    if (!occupied) {
                        if (found1 == p) found1 = (u32)r;
                        else { found2 = (u32)r; break; }
                    }
                }
                if (found1 == p) { V.missed_cat[ii] = 0; local_fail[tid].push_back((u32)p); }
                else { V.witness[ii] = found1; V.missed_cat[ii] = (found2 == p) ? 1 : 2; }
            }
        }
    }
    for (auto& v : local_fail) for (u32 p : v) V.failures.push_back(p);
    std::sort(V.failures.begin(), V.failures.end());
    return V;
}

// ------------------------------------------------------ local optimization --
// A position v (offset from base) may be added to H iff for every prime whose
// missed-class count is exactly 1, v avoids that class.
static bool legal_positions(const std::vector<u32>& primes, const Verdict& V,
                            i64 base, i64 lo, i64 hi, std::vector<i64>& out) {
    if (hi < lo) return false;
    u64 len = (u64)(hi - lo) + 1;
    Bitmap ok; ok.reset(len, true);
    for (size_t i = 0; i < primes.size(); ++i) {
        if (V.missed_cat[i] != 1) continue;
        u64 p = primes[i], r = V.witness[i];
        // absolute n with n = r (mod p); first such n >= base+lo
        i64 start = base + lo;
        u64 i0 = nmod((i64)r - start, p);
        for (u64 j = i0; j < len; j += p) ok.clr(j);
    }
    out.clear();
    for (u64 j = 0; j < len; ++j) if (ok.get(j)) out.push_back(lo + (i64)j);
    return !out.empty();
}

// --------------------------------------------------------------------- main --
int main(int ac, char** av) {
    try {
        Args a = parse(ac, av);
#ifdef _OPENMP
        if (a.threads > 0) omp_set_num_threads(a.threads);
#endif
        self_test_fastmod();

        std::fprintf(stderr, "k=%llu  x=%llu  y=%llu  z=%llu  tie=%s\n",
                     (unsigned long long)a.k, (unsigned long long)a.x,
                     (unsigned long long)a.y, (unsigned long long)a.z, a.tie.c_str());

        std::vector<u32> P = primes_upto(a.k);
        std::fprintf(stderr, "primes <= k: %zu\n", P.size());

        // ---------------- shift sweep mode -------------------------------
        if (a.scan_only) {
            std::printf("# shift  survivors  diameter\n");
            i64 bests = 0; u64 bestd = ~0ULL;
            for (i64 s = a.scan_lo; s <= a.scan_hi; s += a.scan_step) {
                std::vector<u32> surv = presieve(s, a.x, a.y, a.z, P);
                std::vector<u32> none;
                greedy_phase(s, surv, P, a.z, a.k, a.zero_run, none, a.tie, a.x, false);
                Block b = narrowest(surv, a.k);
                if (surv.size() < a.k) {
                    std::printf("%lld  %zu  FAIL(short)\n", (long long)s, surv.size());
                } else {
                    std::printf("%lld  %zu  %llu\n", (long long)s, surv.size(),
                                (unsigned long long)b.diam);
                    if (b.diam < bestd) { bestd = b.diam; bests = s; }
                }
                std::fflush(stdout);
            }
            std::fprintf(stderr, "\nbest shift = %lld  diameter = %llu\n",
                         (long long)bests, (unsigned long long)bestd);
            std::fprintf(stderr, "rerun without --scan using --shift %lld\n", (long long)bests);
            return 0;
        }

        // ---------------- construct --------------------------------------
        std::vector<u32> forced;
        std::vector<u32> H;
        i64 base = 0;
        Verdict V;

        for (int attempt = 0; attempt < 6; ++attempt) {
            std::vector<u32> surv = presieve(a.shift, a.x, a.y, a.z, P);
            std::fprintf(stderr, "presieve survivors: %zu (density %.4f)\n",
                         surv.size(), (double)surv.size() / (double)(a.x + 1));

            GreedyStats gs = greedy_phase(a.shift, surv, P, a.z, a.k, a.zero_run,
                                          forced, a.tie, a.x, false);
            std::fprintf(stderr, "greedy: deleted %llu, last active prime %llu, alive %zu\n",
                         (unsigned long long)gs.deleted,
                         (unsigned long long)gs.last_prime, surv.size());

            if (surv.size() < a.k) {
                std::fprintf(stderr,
                    "FAIL: only %zu survivors, need %llu.  Increase --x "
                    "(try %.0f) or lower --z.\n",
                    surv.size(), (unsigned long long)a.k,
                    (double)a.x * 1.08);
                return 2;
            }

            Block b = narrowest(surv, a.k);
            base = a.shift + (i64)surv[b.idx];
            H.assign(surv.begin() + b.idx, surv.begin() + b.idx + a.k);
            u32 z0 = H[0];
            for (auto& h : H) h -= z0;
            std::fprintf(stderr, "sieve diameter: %llu  (k log k + k = %.0f)\n",
                         (unsigned long long)H.back(),
                         (double)a.k * std::log((double)a.k) + (double)a.k);

            V = verify_tuple(H, base, P, a.k);
            if (V.failures.empty()) break;

            std::fprintf(stderr, "verification: %zu inadmissible primes "
                         "(largest %u); forcing and retrying\n",
                         V.failures.size(), V.failures.back());
            for (u32 p : V.failures) forced.push_back(p);
            std::sort(forced.begin(), forced.end());
            forced.erase(std::unique(forced.begin(), forced.end()), forced.end());
            if (attempt == 5) throw std::runtime_error("repair loop did not converge");
        }
        std::fprintf(stderr, "verified admissible, diameter %llu\n",
                     (unsigned long long)H.back());

        // ---------------- local optimization -----------------------------
        int improved = 0;
        for (int round = 0; round < a.local_rounds; ++round) {
            bool progress = false;

            // contract: insert an interior point, then drop an endpoint
            std::vector<i64> cand;
            Bitmap inH; inH.reset(H.back() + 1, false);
            for (u32 h : H) inH.set(h);
            if (legal_positions(P, V, base, 1, (i64)H.back() - 1, cand)) {
                for (i64 v : cand) {
                    if (inH.get((u64)v)) continue;
                    // H + {v}, drop the endpoint that shrinks more
                    std::vector<u32> H2;
                    H2.reserve(H.size() + 1);
                    for (u32 h : H) H2.push_back(h);
                    H2.push_back((u32)v);
                    std::sort(H2.begin(), H2.end());
                    u64 dl = H2[H2.size() - 1] - H2[1];        // drop left end
                    u64 dr = H2[H2.size() - 2] - H2[0];        // drop right end
                    std::vector<u32> H3;
                    if (dl <= dr) H3.assign(H2.begin() + 1, H2.end());
                    else          H3.assign(H2.begin(), H2.end() - 1);
                    u32 z0 = H3[0];
                    i64 nb = base + (i64)z0;
                    for (auto& h : H3) h -= z0;
                    if (H3.back() >= H.back()) continue;
                    H = H3; base = nb;
                    V = verify_tuple(H, base, P, a.k);
                    if (!V.failures.empty())
                        throw std::runtime_error("internal: contract produced inadmissible tuple");
                    progress = true; ++improved;
                    break;
                }
            }
            if (progress) {
                std::fprintf(stderr, "  contract -> diameter %llu\n",
                             (unsigned long long)H.back());
                continue;
            }

            // shift: drop an endpoint, extend on the other side
            bool moved = false;
            for (int dir = 0; dir < 2 && !moved; ++dir) {
                std::vector<i64> ext;
                i64 lo, hi;
                if (dir == 0) { lo = -(i64)H.back() / 8 - 2; hi = -1; }
                else          { lo = (i64)H.back() + 1; hi = (i64)H.back() + (i64)H.back() / 8 + 2; }
                if (!legal_positions(P, V, base, lo, hi, ext)) continue;
                i64 v = (dir == 0) ? ext.back() : ext.front();
                std::vector<u32> H2;
                if (dir == 0) {
                    H2.push_back(0);
                    for (size_t i = 0; i + 1 < H.size(); ++i) H2.push_back((u32)((i64)H[i] - v));
                } else {
                    for (size_t i = 1; i < H.size(); ++i) H2.push_back(H[i]);
                    H2.push_back((u32)v);
                }
                std::sort(H2.begin(), H2.end());
                u32 z0 = H2[0];
                i64 nb = base + ((dir == 0) ? v : (i64)z0);
                for (auto& h : H2) h -= z0;
                if (H2.size() != H.size()) continue;
                if (H2.back() > H.back()) continue;             // keep only non-worsening
                Verdict V2 = verify_tuple(H2, nb, P, a.k);
                if (!V2.failures.empty()) continue;
                bool strict = H2.back() < H.back();
                H = H2; base = nb; V = V2;
                if (strict) { ++improved; moved = true; progress = true;
                    std::fprintf(stderr, "  shift -> diameter %llu\n",
                                 (unsigned long long)H.back()); }
                else moved = true;
            }
            if (!progress) break;
        }
        std::fprintf(stderr, "local optimization: %d improvements, final diameter %llu\n",
                     improved, (unsigned long long)H.back());

        // ---------------- certificate ------------------------------------
        V = verify_tuple(H, base, P, a.k);
        if (!V.failures.empty()) throw std::runtime_error("final verification failed");

        std::ofstream o(a.cert);
        if (!o) throw std::runtime_error("cannot write " + a.cert);
        o << "format=admissible-tuple-certificate/1\n";
        o << "k=" << a.k << "\n";
        o << "base=" << base << "\n";
        o << "diameter=" << H.back() << "\n";
        o << "prime_count=" << P.size() << "\n";
        o << "note=offsets are gaps h[i]-h[i-1] with h[0]=0; witness r_p is a\n";
        o << "note=residue class mod p containing no element; primes p>k need\n";
        o << "note=no witness since k elements cannot cover p>k classes.\n";
        o << "BEGIN_GAPS\n";
        for (size_t i = 1; i < H.size(); ++i) {
            o << (H[i] - H[i - 1]) << (((i % 20) == 0) ? '\n' : ' ');
        }
        o << "\nEND_GAPS\n";
        o << "BEGIN_WITNESSES\n";
        for (size_t i = 0; i < P.size(); ++i) o << P[i] << " " << V.witness[i] << "\n";
        o << "END_WITNESSES\n";
        o.close();

        double pred = (double)a.k * std::log((double)a.k) + (double)a.k;
        std::printf("PASS\n");
        std::printf("k          = %llu\n", (unsigned long long)a.k);
        std::printf("shift      = %lld\n", (long long)a.shift);
        std::printf("window     = %llu\n", (unsigned long long)a.x);
        std::printf("base       = %lld\n", (long long)base);
        std::printf("diameter   = %llu\n", (unsigned long long)H.back());
        std::printf("k log k+k  = %.0f   (ratio %.4f)\n", pred, (double)H.back() / pred);
        std::printf("certificate= %s\n", a.cert.c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
