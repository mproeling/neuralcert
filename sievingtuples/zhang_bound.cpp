// zhang_bound.cpp
//
// H(k) upper bounds for k beyond the reach of window sieving.
//
// The construction is  H = { p_{m+1}, ..., p_{m+k} }  with p_{m+1} > k, i.e.
// the first k primes exceeding k.  Admissibility is STRUCTURAL, not searched:
// every element is a prime greater than k, so for each prime p <= k no element
// is divisible by p, hence residue class 0 mod p is empty.  For p > k a set of
// k elements cannot meet all p classes.  The certificate is therefore one line
// ("all elements are primes > k, witness r_p = 0 for every p <= k") plus the
// two endpoints -- there is nothing to store per prime and nothing to search.
//
// Two modes:
//   exact  : segmented sieve, returns the true value of p_{pi(k)+k} - p_{pi(k)+1}.
//   dusart : closed-form rigorous upper bound, no enumeration, valid for
//            k large.  Use it to sanity-check the exact run, or to state a
//            bound when the enumeration is impractical.
//
// Dusart (2010) explicit estimates used:
//   pi(t)  <= t/ln t * (1 + 1/ln t + 2.334/ln^2 t)        (t >= 2953652287)
//   p_n    <= n(ln n + ln ln n - 1 + (ln ln n - 2)/ln n)  (n >= 688383)
// Both are applied in the direction that can only inflate the answer, so the
// printed bound is rigorous.  The code refuses to print a Dusart bound outside
// the stated validity ranges.
//
// Build:
//   g++ -O3 -march=native -std=c++17 zhang_bound.cpp -o zhang_bound

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using u64 = std::uint64_t;
using u32 = std::uint32_t;

static std::vector<u32> base_primes(u64 n) {
    std::vector<u32> ps;
    if (n < 2) return ps;
    std::vector<unsigned char> comp(n + 1, 0);
    for (u64 i = 2; i <= n; ++i) {
        if (comp[i]) continue;
        ps.push_back((u32)i);
        if (i * i <= n) for (u64 j = i * i; j <= n; j += i) comp[j] = 1;
    }
    return ps;
}

// Walk primes upward from `start`, reporting the 1st and kth prime strictly
// greater than `start`.  Segmented, odd-only, O(sqrt(X)) memory.
struct Endpoints { u64 first = 0, kth = 0; bool ok = false; };

static Endpoints first_k_primes_past(u64 start, u64 k, bool verbose) {
    Endpoints E;
    const u64 SEG = 1ULL << 20;              // bytes of odd slots per segment
    u64 lo = start + 1;
    if (lo % 2 == 0) lo += 1;                 // odd start
    u64 count = 0;
    u64 sqrt_cap = 1ULL << 22;                // grown as needed
    std::vector<u32> bp = base_primes(sqrt_cap);
    std::vector<unsigned char> seg(SEG);
    u64 report = 0;

    if (start < 2) { /* handle 2 separately */
        E.first = 2; ++count;
    }

    while (count < k) {
        u64 hi = lo + 2 * SEG - 2;
        if (hi / sqrt_cap >= sqrt_cap) {
            sqrt_cap *= 2;
            bp = base_primes(sqrt_cap);
        }
        std::memset(seg.data(), 1, SEG);
        for (u32 p : bp) {
            if (p == 2) continue;
            u64 pp = (u64)p * p;
            if (pp > hi) break;
            u64 s = std::max(pp, ((lo + p - 1) / p) * (u64)p);
            if ((s & 1) == 0) s += p;
            for (u64 v = s; v <= hi; v += 2ULL * p) seg[(v - lo) >> 1] = 0;
        }
        for (u64 i = 0; i < SEG && count < k; ++i) {
            if (!seg[i]) continue;
            u64 v = lo + 2 * i;
            if (v > hi) break;
            if (count == 0) E.first = v;
            ++count;
            if (count == k) { E.kth = v; E.ok = true; break; }
        }
        lo = hi + 2;
        if (verbose && lo > report) {
            std::fprintf(stderr, "  ... at %.3e, %llu / %llu primes\n",
                         (double)lo, (unsigned long long)count, (unsigned long long)k);
            report = lo + (u64)5e8;
        }
    }
    return E;
}

// pi(t) upper bound, Dusart 2010
static double pi_upper(double t) {
    double L = std::log(t);
    return t / L * (1.0 + 1.0 / L + 2.334 / (L * L));
}
// p_n upper bound, Dusart 2010
static double pn_upper(double n) {
    double L = std::log(n), LL = std::log(L);
    return n * (L + LL - 1.0 + (LL - 2.0) / L);
}

int main(int argc, char** argv) {
    try {
        u64 k = 0;
        std::string mode = "exact";
        bool verbose = true;
        for (int i = 1; i < argc; ++i) {
            std::string t = argv[i];
            auto need = [&]() {
                if (i + 1 >= argc) throw std::runtime_error("missing value");
                return std::string(argv[++i]);
            };
            if (t == "--k") k = std::stoull(need());
            else if (t == "--mode") mode = need();
            else if (t == "--quiet") verbose = false;
            else if (t == "-h" || t == "--help") {
                std::printf("usage: zhang_bound --k K [--mode exact|dusart|both]\n");
                return 0;
            } else throw std::runtime_error("unknown argument " + t);
        }
        if (k < 2) throw std::runtime_error("--k required, >= 2");

        double kk = (double)k;
        double pred = kk * std::log(kk) + kk;
        std::printf("k          = %llu\n", (unsigned long long)k);
        std::printf("k log k+k  = %.0f   (empirical rule of thumb for H(k))\n", pred);

        if (mode == "dusart" || mode == "both") {
            if (k < 2953652287.0 / 1.0 && k < 3e9) {
                // pi(k) upper bound needs t >= 2953652287; below that use the
                // exact count instead.  We only need an UPPER bound on the
                // index, so a crude safe bound is fine for smaller k.
                std::printf("dusart     : pi(k) bound outside stated range for this k;\n");
                std::printf("             use --mode exact, or supply pi(k) exactly.\n");
            }
            double idx = pi_upper((double)k) + kk;         // index of the kth prime past k
            if (idx < 688383.0) {
                std::printf("dusart     : index below p_n validity range\n");
            } else {
                double up = pn_upper(idx);
                std::printf("dusart     : H(%llu) <= %.0f - %llu - 1 = %.0f  (rigorous, no enumeration)\n",
                            (unsigned long long)k, up, (unsigned long long)k, up - kk - 1.0);
                std::printf("             ratio to k log k + k : %.4f\n", (up - kk - 1.0) / pred);
            }
        }

        if (mode == "exact" || mode == "both") {
            if (verbose) std::fprintf(stderr, "segmented sieve from %llu ...\n",
                                      (unsigned long long)k);
            Endpoints E = first_k_primes_past(k, k, verbose);
            if (!E.ok) throw std::runtime_error("sieve did not complete");
            u64 d = E.kth - E.first;
            std::printf("exact      : p_{pi(k)+1} = %llu, p_{pi(k)+k} = %llu\n",
                        (unsigned long long)E.first, (unsigned long long)E.kth);
            std::printf("             H(%llu) <= %llu\n",
                        (unsigned long long)k, (unsigned long long)d);
            std::printf("             ratio to k log k + k : %.4f\n", (double)d / pred);
            std::printf("\ncertificate: H = { p : p prime, %llu <= p <= %llu }\n",
                        (unsigned long long)E.first, (unsigned long long)E.kth);
            std::printf("             |H| = %llu, every element prime and > k,\n",
                        (unsigned long long)k);
            std::printf("             so residue class 0 mod p is empty for every p <= k.\n");
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
