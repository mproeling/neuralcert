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
//   dusart : rigorous upper bound.  Below the validity threshold of the
//            analytic pi(k) estimate, pi(k) is counted exactly first.
//
// Dusart (2010) explicit estimates used:
//   pi(t)  <= t/ln t * (1 + 1/ln t + 2.334/ln^2 t)        (t >= 2953652287)
//   p_n    <= n(ln n + ln ln n - 1 + (ln ln n - 2)/ln n)  (n >= 688383)
// MPFR directed rounding is used throughout the analytic calculation.  A
// plain double evaluation is not sufficient to label an integer bound rigorous.
//
// Build:
//   g++ -O3 -march=native -std=c++17 zhang_bound.cpp -lmpfr -lgmp -o zhang_bound

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <limits>
#include <mpfr.h>
#include <stdexcept>
#include <string>
#include <vector>

using u64 = std::uint64_t;
using u32 = std::uint32_t;

static constexpr u64 PI_DUSART_MIN = 2953652287ULL;
static constexpr u64 PN_DUSART_MIN = 688383ULL;
static bool pi_formula_valid(u64 x) { return x >= PI_DUSART_MIN; }
static bool pn_formula_valid(u64 n) { return n >= PN_DUSART_MIN; }

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

// Exact pi(n), with O(sqrt(n)) setup memory and a fixed-size segmented sieve.
static u64 prime_count_exact(u64 n) {
    if (n < 2) return 0;
    u64 root = (u64)std::sqrt((long double)n);
    while ((root + 1) <= n / (root + 1)) ++root;
    while (root > n / root) --root;
    std::vector<u32> bp = base_primes(root);
    const u64 SEG = 1ULL << 20;
    std::vector<unsigned char> composite(SEG);
    u64 count = 1; // prime 2
    for (u64 lo = 3; lo <= n;) {
        u64 slots = std::min(SEG, (n - lo) / 2 + 1);
        u64 hi = lo + 2 * (slots - 1);
        std::memset(composite.data(), 0, (size_t)slots);
        for (u32 p : bp) {
            if (p == 2) continue;
            u64 pp = (u64)p * p;
            if (pp > hi) break;
            u64 first = std::max(pp, ((lo + p - 1) / p) * (u64)p);
            if ((first & 1) == 0) first += p;
            for (u64 v = first; v <= hi; v += 2ULL * p)
                composite[(v - lo) / 2] = 1;
        }
        for (u64 i = 0; i < slots; ++i) count += !composite[i];
        if (hi >= n - 1) break;
        lo = hi + 2;
    }
    return count;
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

// Rigorous floor of the Dusart upper expression for pi(k).  Since pi(k) is an
// integer and pi(k) <= expression, floor(expression_upper) is an upper bound.
static u64 pi_upper_index(u64 k) {
    if (!pi_formula_valid(k))
        throw std::runtime_error("internal error: pi(k) formula outside validity range");
    mpfr_t x, log_lo, term, sum;
    mpfr_inits2(256, x, log_lo, term, sum, (mpfr_ptr)0);
    mpfr_set_uj(x, k, MPFR_RNDN);
    mpfr_log(log_lo, x, MPFR_RNDD);
    mpfr_ui_div(sum, 1, log_lo, MPFR_RNDU);
    mpfr_mul(term, log_lo, log_lo, MPFR_RNDD);
    mpfr_set_str(x, "2.334", 10, MPFR_RNDU);
    mpfr_div(term, x, term, MPFR_RNDU);
    mpfr_add(sum, sum, term, MPFR_RNDU);
    mpfr_add_ui(sum, sum, 1, MPFR_RNDU);
    mpfr_set_uj(x, k, MPFR_RNDN);
    mpfr_div(x, x, log_lo, MPFR_RNDU);
    mpfr_mul(sum, x, sum, MPFR_RNDU);
    mpfr_set_uj(x, std::numeric_limits<u64>::max(), MPFR_RNDN);
    if (mpfr_cmp(sum, x) > 0) {
        mpfr_clears(x, log_lo, term, sum, (mpfr_ptr)0);
        throw std::overflow_error("pi(k) upper index does not fit uint64");
    }
    u64 result = (u64)mpfr_get_uj(sum, MPFR_RNDD);
    mpfr_clears(x, log_lo, term, sum, (mpfr_ptr)0);
    return result;
}

// Rigorous ceil of Dusart's p_n upper expression, using interval endpoints
// for log(n) and log(log(n)).  Here n >= 688383, so log(log(n))-2 is positive.
static void pn_upper_integer(mpz_t out, u64 n) {
    if (!pn_formula_valid(n))
        throw std::runtime_error("p_n bound outside validity range (n < 688383)");
    mpfr_t x, l_lo, l_hi, ll_hi, q, sum;
    mpfr_inits2(256, x, l_lo, l_hi, ll_hi, q, sum, (mpfr_ptr)0);
    mpfr_set_uj(x, n, MPFR_RNDN);
    mpfr_log(l_lo, x, MPFR_RNDD);
    mpfr_log(l_hi, x, MPFR_RNDU);
    mpfr_log(ll_hi, l_hi, MPFR_RNDU);
    mpfr_sub_ui(q, ll_hi, 2, MPFR_RNDU);
    mpfr_div(q, q, l_lo, MPFR_RNDU);
    mpfr_add(sum, l_hi, ll_hi, MPFR_RNDU);
    mpfr_sub_ui(sum, sum, 1, MPFR_RNDU);
    mpfr_add(sum, sum, q, MPFR_RNDU);
    mpfr_set_uj(x, n, MPFR_RNDN);
    mpfr_mul(sum, sum, x, MPFR_RNDU);
    mpfr_get_z(out, sum, MPFR_RNDU);
    mpfr_clears(x, l_lo, l_hi, ll_hi, q, sum, (mpfr_ptr)0);
}

int main(int argc, char** argv) {
    try {
        u64 k = 0;
        std::string mode = "exact";
        bool verbose = true;
        bool self_test = false;
        for (int i = 1; i < argc; ++i) {
            std::string t = argv[i];
            auto need = [&]() {
                if (i + 1 >= argc) throw std::runtime_error("missing value");
                return std::string(argv[++i]);
            };
            if (t == "--k") k = std::stoull(need());
            else if (t == "--mode") mode = need();
            else if (t == "--quiet") verbose = false;
            else if (t == "--self-test") self_test = true;
            else if (t == "-h" || t == "--help") {
                std::printf("usage: zhang_bound --k K [--mode exact|dusart|both] [--self-test]\n");
                return 0;
            } else throw std::runtime_error("unknown argument " + t);
        }
        if (self_test) {
            if (pi_formula_valid(PI_DUSART_MIN - 1) ||
                !pi_formula_valid(PI_DUSART_MIN) ||
                pn_formula_valid(PN_DUSART_MIN - 1) ||
                !pn_formula_valid(PN_DUSART_MIN))
                throw std::runtime_error("bound-domain self-test failed");
            if (prime_count_exact(10) != 4 || prime_count_exact(100) != 25)
                throw std::runtime_error("exact pi(k) self-test failed");
            mpz_t boundary;
            mpz_init(boundary);
            pn_upper_integer(boundary, PN_DUSART_MIN);
            if (mpz_sgn(boundary) <= 0) {
                mpz_clear(boundary);
                throw std::runtime_error("p_n boundary self-test failed");
            }
            mpz_clear(boundary);
            std::printf("PASS: domain boundaries pi=%llu, p_n=%llu\n",
                        (unsigned long long)PI_DUSART_MIN,
                        (unsigned long long)PN_DUSART_MIN);
            return 0;
        }
        if (k < 2) throw std::runtime_error("--k required, >= 2");
        if (mode != "exact" && mode != "dusart" && mode != "both")
            throw std::runtime_error("--mode must be exact, dusart, or both");

        double kk = (double)k;
        double pred = kk * std::log(kk) + kk;
        std::printf("k          = %llu\n", (unsigned long long)k);
        std::printf("k log k+k  = %.0f   (empirical rule of thumb for H(k))\n", pred);

        if (mode == "dusart" || mode == "both") {
            u64 pik_upper;
            const char* provenance;
            if (!pi_formula_valid(k)) {
                pik_upper = prime_count_exact(k);
                provenance = "exact pi(k) + Dusart p_n";
            } else {
                pik_upper = pi_upper_index(k);
                provenance = "Dusart pi(k) + Dusart p_n";
            }
            if (pik_upper > std::numeric_limits<u64>::max() - k)
                throw std::overflow_error("prime index k + pi(k) overflows uint64");
            u64 idx = k + pik_upper;
            if (!pn_formula_valid(idx)) {
                std::fprintf(stderr,
                    "dusart     : unavailable: index k + pi(k) = %llu is below "
                    "the proven p_n domain %llu; no rigorous analytic bound printed\n",
                    (unsigned long long)idx, (unsigned long long)PN_DUSART_MIN);
                if (mode == "dusart") return 2;
            } else {
                mpz_t up, diameter;
                mpz_inits(up, diameter, (mpz_ptr)0);
                pn_upper_integer(up, idx);
                mpz_sub_ui(diameter, up, k);
                mpz_sub_ui(diameter, diameter, 1);
                std::printf("dusart     : pi(k) <= %llu (%s)\n",
                            (unsigned long long)pik_upper, provenance);
                gmp_printf("             H(%llu) <= %Zd - %llu - 1 = %Zd"
                           "  (rigorous, MPFR outward rounding)\n",
                           (unsigned long long)k, up, (unsigned long long)k, diameter);
                mpz_clears(up, diameter, (mpz_ptr)0);
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
