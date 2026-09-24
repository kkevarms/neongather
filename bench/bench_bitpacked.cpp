#include <cstdio>
#include <cstring>
#include <chrono>
#include <random>
#include <vector>

#include "../src/gather_bitpacked.h"

using namespace neongather;

int main()
{
#if !defined(NEONGATHER_BITPACKED_ARM64)
    printf("SKIP: not built for ARM64\n");
    return 0;
#else
    const size_t tableSize = 27;
    const size_t packedTableBytes = (tableSize + 3) / 4; // 7
    const size_t tablesPerBlock = 64 / packedTableBytes; // 9
    const size_t numTables = 1024;
    const int iterations = 200000; // same as bench_gather.cpp, apples-to-apples

    std::mt19937_64 rng(99999);
    std::vector<uint8_t> tables(numTables * tableSize);
    std::vector<uint8_t> localIndices(numTables);
    for (auto& b : tables) b = (uint8_t)(rng() % 3);
    for (auto& idx : localIndices) idx = (uint8_t)(rng() % tableSize);

    std::vector<uint8_t> packedTables(numTables * packedTableBytes);
    for (size_t t = 0; t < numTables; ++t)
        packTrits(&tables[t * tableSize], tableSize, &packedTables[t * packedTableBytes]);

    const size_t numBlocks = (numTables + tablesPerBlock - 1) / tablesPerBlock;
    std::vector<uint8_t> packedBlocks(numBlocks * 64, 0);
    for (size_t t = 0; t < numTables; ++t)
    {
        const size_t block = t / tablesPerBlock;
        const size_t slot = t % tablesPerBlock;
        memcpy(&packedBlocks[block * 64 + slot * packedTableBytes], &packedTables[t * packedTableBytes], packedTableBytes);
    }

    std::vector<const uint8_t*> tablePtrs(numTables);
    for (size_t t = 0; t < numTables; ++t) tablePtrs[t] = &tables[t * tableSize];

    std::vector<uint8_t> out(numTables, 0);
    volatile uint64_t sink = 0;

    // Scalar baseline (plain, unpacked -- the real comparison point, same as bench_gather.cpp).
    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iterations; ++it)
    {
        localIndices[0] = (uint8_t)(it % tableSize);
        for (size_t t = 0; t < numTables; ++t) out[t] = tablePtrs[t][localIndices[t]];
        sink += out[0] + out[numTables - 1];
    }
    auto t1 = std::chrono::steady_clock::now();
    double scalarMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("scalar:           %.2f ms total, %.2f ns/gather-batch, %.3f ns/element  (sink=%llu)\n",
           scalarMs, scalarMs * 1e6 / iterations, scalarMs * 1e6 / iterations / numTables, (unsigned long long)sink);

    sink = 0;
    auto t2 = std::chrono::steady_clock::now();
    for (int it = 0; it < iterations; ++it)
    {
        localIndices[0] = (uint8_t)(it % tableSize);
        gatherPackedBitsTBL4(packedBlocks.data(), packedTableBytes, tablesPerBlock, localIndices.data(), out.data(), numTables);
        sink += out[0] + out[numTables - 1];
    }
    auto t3 = std::chrono::steady_clock::now();
    double bitMs = std::chrono::duration<double, std::milli>(t3 - t2).count();
    printf("bitpacked TBL4:   %.2f ms total, %.2f ns/gather-batch, %.3f ns/element  (%.2fx vs scalar, sink=%llu)\n",
           bitMs, bitMs * 1e6 / iterations, bitMs * 1e6 / iterations / numTables, scalarMs / bitMs, (unsigned long long)sink);
    return 0;
#endif
}
