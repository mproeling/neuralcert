// verify_tuple.cpp
//
// Independent verifier for admissible-tuple-certificate/1.
//
// Shares NO code path with the constructor: it re-derives the primes, re-reads
// the gaps, and re-checks each claimed witness residue by direct enumeration.
// It also independently confirms that the witness set is complete, i.e. that
// the certificate lists exactly the primes p <= k.
//
// Lemma used (stated in the certificate, checked here only for p <= k):
//   a k-element set cannot occupy all p residue classes when p > k, so
//   admissibility is a finite condition over p <= k.
//
// Build:
//   g++ -O3 -march=native -fopenmp -std=c++17 verify_tuple.cpp -o verify_tuple

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using u64 = std::uint64_t;
using u32 = std::uint32_t;
using i64 = std::int64_t;

static std::vector<u32> primes_upto(u64 n) {
    std::vector<u32> ps;
    if (n < 2) return ps;
    ps.push_back(2);
    if (n < 3) return ps;
    u64 m = (n - 1) / 2;
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

int main(int argc, char** argv) {
    try {
        if (argc < 2) { std::fprintf(stderr, "usage: verify_tuple FILE.cert\n"); return 1; }
        std::ifstream f(argv[1]);
        if (!f) throw std::runtime_error("cannot open certificate");

        u64 k = 0, diameter = 0, prime_count = 0;
        i64 base = 0;
        std::string fmt;
        std::vector<u64> gaps;
        std::vector<std::pair<u32, u32>> wit;

        std::string line;
        int section = 0;   // 0 header, 1 gaps, 2 witnesses
        while (std::getline(f, line)) {
            if (line == "BEGIN_GAPS") { section = 1; continue; }
            if (line == "END_GAPS") { section = 0; continue; }
            if (line == "BEGIN_WITNESSES") { section = 2; continue; }
            if (line == "END_WITNESSES") { section = 0; continue; }
            if (section == 1) {
                std::istringstream is(line); u64 g;
                while (is >> g) gaps.push_back(g);
            } else if (section == 2) {
                std::istringstream is(line); u64 p, r;
                if (is >> p >> r) wit.push_back({(u32)p, (u32)r});
            } else {
                auto eq = line.find('=');
                if (eq == std::string::npos) continue;
                std::string key = line.substr(0, eq), val = line.substr(eq + 1);
                if (key == "format") fmt = val;
                else if (key == "k") k = std::stoull(val);
                else if (key == "base") base = std::stoll(val);
                else if (key == "diameter") diameter = std::stoull(val);
                else if (key == "prime_count") prime_count = std::stoull(val);
            }
        }
        if (fmt != "admissible-tuple-certificate/1")
            throw std::runtime_error("unrecognized format: " + fmt);

        // ---- rebuild the tuple from gaps -------------------------------
        if (gaps.size() + 1 != k)
            throw std::runtime_error("gap count " + std::to_string(gaps.size()) +
                                     " inconsistent with k=" + std::to_string(k));
        std::vector<u64> H;
        H.reserve(k);
        H.push_back(0);
        for (u64 g : gaps) {
            if (g == 0) throw std::runtime_error("zero gap: repeated element");
            H.push_back(H.back() + g);
        }
        if (H.back() != diameter)
            throw std::runtime_error("claimed diameter " + std::to_string(diameter) +
                                     " != reconstructed " + std::to_string(H.back()));
        std::printf("[ok] k = %llu, %llu distinct sorted offsets, diameter %llu\n",
                    (unsigned long long)k, (unsigned long long)H.size(),
                    (unsigned long long)H.back());

        // ---- independently regenerate the primes ------------------------
        std::vector<u32> P = primes_upto(k);
        if (P.size() != wit.size())
            throw std::runtime_error("witness count " + std::to_string(wit.size()) +
                                     " != pi(k) = " + std::to_string(P.size()));
        if (prime_count && prime_count != P.size())
            throw std::runtime_error("declared prime_count disagrees with pi(k)");
        for (size_t i = 0; i < P.size(); ++i)
            if (wit[i].first != P[i])
                throw std::runtime_error("witness list is not exactly the primes <= k "
                                         "(mismatch at index " + std::to_string(i) + ")");
        std::printf("[ok] witness list is exactly the %zu primes p <= k\n", P.size());

        // ---- check every witness ---------------------------------------
        long long bad = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64) reduction(+:bad)
#endif
        for (long long i = 0; i < (long long)P.size(); ++i) {
            u64 p = P[i], r = wit[i].second;
            if (r >= p) { ++bad; continue; }
            i64 b = base % (i64)p; if (b < 0) b += (i64)p;
            bool hit = false;
            for (u64 h : H) {
                if ((u64)((b + (i64)(h % p)) % (i64)p) == r) { hit = true; break; }
            }
            if (hit) {
                std::fprintf(stderr, "FAIL: class %llu mod %llu is occupied\n",
                             (unsigned long long)r, (unsigned long long)p);
                ++bad;
            }
        }
        if (bad) throw std::runtime_error(std::to_string(bad) + " witness failures");

        std::printf("[ok] every prime p <= k misses its claimed residue class\n");
        std::printf("VERIFIED: admissible %llu-tuple of diameter %llu\n",
                    (unsigned long long)k, (unsigned long long)H.back());
        std::printf("          H(%llu) <= %llu\n",
                    (unsigned long long)k, (unsigned long long)H.back());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "VERIFICATION FAILED: %s\n", e.what());
        return 2;
    }
}
