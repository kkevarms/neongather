#include <cstdio>
#include <chrono>
#include <random>
#include <vector>

#include "../src/gather_select.h"

using namespace neongather;

int main()
{
#if !defined(NEONGATHER_SELECT_ARM64)
    printf("SKIP: not built for ARM64\n");
    return 0;
#else
    const size_t numColumns = 27;
    const size_t n = 1024;
    const int iterations = 200000; // same as bench_gather.cpp, for a real apples-to-apples read

    std::mt19937_64 rng(54321);
    std::vector<std::vector<uint8_t>> columnStorage(numColumns, std::vector<uint8_t>(n));
    std::vector<const uint8_t*> columns(numColumns);
    for (size_t k = 0; k < numColumns; ++k)
    {
        for (size_t i = 0; i < n; ++i) columnStorage[k][i] = (uint8_t)(rng() % 3);
        columns[k] = columnStorage[k].data();
    }
    std::vector<uint8_t> indices(n);
    for (auto& idx : indices) idx = (uint8_t)(rng() % numColumns);

    std::vector<uint8_t> out(n, 0);
    volatile uint64_t sink = 0;

    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iterations; ++it)
    {
        indices[0] = (uint8_t)(it % numColumns);
        selectScalar(columns.data(), numColumns, indices.data(), out.data(), n);
        sink += out[0] + out[n - 1];
    }
    auto t1 = std::chrono::steady_clock::now();
    double scalarMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("scalar:       %.2f ms total, %.2f ns/gather-batch, %.3f ns/element  (sink=%llu)\n",
           scalarMs, scalarMs * 1e6 / iterations, scalarMs * 1e6 / iterations / n, (unsigned long long)sink);

    sink = 0;
    auto t2 = std::chrono::steady_clock::now();
    for (int it = 0; it < iterations; ++it)
    {
        indices[0] = (uint8_t)(it % numColumns);
        selectChain(columns.data(), numColumns, indices.data(), out.data(), n);
        sink += out[0] + out[n - 1];
    }
    auto t3 = std::chrono::steady_clock::now();
    double selMs = std::chrono::duration<double, std::milli>(t3 - t2).count();
    printf("selectChain:  %.2f ms total, %.2f ns/gather-batch, %.3f ns/element  (%.2fx vs scalar, sink=%llu)\n",
           selMs, selMs * 1e6 / iterations, selMs * 1e6 / iterations / n, scalarMs / selMs, (unsigned long long)sink);
    return 0;
#endif
}
