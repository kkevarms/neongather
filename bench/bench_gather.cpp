#include <cstdio>
#include <cstring>
#include <chrono>
#include <random>
#include <vector>

#include "../src/gather.h"

using namespace neongather;

int main()
{
    const size_t tableSize = 27;
    const size_t tablesPerBlock = 2;
    const size_t numTables = 1024;
    const int iterations = 200000; // matches roughly one bpp9000 score() pass's neuron-tick count

    std::mt19937_64 rng(12345);
    std::vector<uint8_t> tables(numTables * tableSize);
    std::vector<uint8_t> localIndices(numTables);
    for (auto& b : tables) b = (uint8_t)(rng() % 3);
    for (auto& idx : localIndices) idx = (uint8_t)(rng() % tableSize);

    const size_t numBlocks = (numTables + tablesPerBlock - 1) / tablesPerBlock;
    std::vector<uint8_t> packed(numBlocks * 64, 0);
    for (size_t t = 0; t < numTables; ++t)
    {
        const size_t block = t / tablesPerBlock;
        const size_t slot = t % tablesPerBlock;
        memcpy(&packed[block * 64 + slot * tableSize], &tables[t * tableSize], tableSize);
    }

    std::vector<const uint8_t*> tablePtrs(numTables);
    for (size_t t = 0; t < numTables; ++t) tablePtrs[t] = &tables[t * tableSize];

    std::vector<uint8_t> out(numTables, 0);
    volatile uint64_t sink = 0; // forces the compiler to actually observe `out` each iteration

    // Scalar baseline. Vary localIndices[0] per iteration so the compiler can't hoist/cache the
    // whole computation out of the loop either.
    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iterations; ++it)
    {
        localIndices[0] = (uint8_t)(it % tableSize);
        gatherScalar(tablePtrs.data(), localIndices.data(), out.data(), numTables);
        sink += out[0] + out[numTables - 1];
    }
    auto t1 = std::chrono::steady_clock::now();
    double scalarMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("scalar:  %.2f ms total, %.2f ns/gather-batch, %.3f ns/element  (sink=%llu)\n",
           scalarMs, scalarMs * 1e6 / iterations, scalarMs * 1e6 / iterations / numTables,
           (unsigned long long)sink);

#if defined(NEONGATHER_ARM64)
    sink = 0;
    auto t2 = std::chrono::steady_clock::now();
    for (int it = 0; it < iterations; ++it)
    {
        localIndices[0] = (uint8_t)(it % tableSize);
        gatherPackedTBL4(packed.data(), tableSize, tablesPerBlock, localIndices.data(), out.data(), numTables);
        sink += out[0] + out[numTables - 1];
    }
    auto t3 = std::chrono::steady_clock::now();
    double tblMs = std::chrono::duration<double, std::milli>(t3 - t2).count();
    printf("TBL4:    %.2f ms total, %.2f ns/gather-batch, %.3f ns/element  (%.2fx vs scalar, sink=%llu)\n",
           tblMs, tblMs * 1e6 / iterations, tblMs * 1e6 / iterations / numTables, scalarMs / tblMs,
           (unsigned long long)sink);
#else
    printf("(no ARM64 TBL4 path built)\n");
#endif
    return 0;
}
