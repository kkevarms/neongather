#include <cstdio>
#include <cstring>
#include <chrono>
#include <random>
#include <thread>
#include <vector>
#include <atomic>

#include "../src/gather_bitpacked.h"

using namespace neongather;

// Mac-only multithreaded scaling test: does the bit-packed TBL4 gather actually scale across
// real cores, or is it memory-bandwidth-bound (shared across cores) rather than compute-bound
// (which would scale near-linearly)? Each thread runs fully independent data (own tables,
// indices, output buffer) to avoid any false sharing/contention that isn't real to the actual
// use case (each mining thread has its own Miner instance and own LUT data).

struct ThreadWork
{
    size_t tableSize = 27;
    size_t packedTableBytes = (27 + 3) / 4;
    size_t tablesPerBlock = 64 / ((27 + 3) / 4);
    size_t numTables = 1024;
    std::vector<uint8_t> packedBlocks;
    std::vector<uint8_t> localIndices;
    std::vector<uint8_t> out;

    void init(uint64_t seed)
    {
        std::mt19937_64 rng(seed);
        std::vector<uint8_t> tables(numTables * tableSize);
        localIndices.resize(numTables);
        for (auto& b : tables) b = (uint8_t)(rng() % 3);
        for (auto& idx : localIndices) idx = (uint8_t)(rng() % tableSize);

        std::vector<uint8_t> packedTables(numTables * packedTableBytes);
        for (size_t t = 0; t < numTables; ++t)
            packTrits(&tables[t * tableSize], tableSize, &packedTables[t * packedTableBytes]);

        const size_t numBlocks = (numTables + tablesPerBlock - 1) / tablesPerBlock;
        packedBlocks.assign(numBlocks * 64, 0);
        for (size_t t = 0; t < numTables; ++t)
        {
            const size_t block = t / tablesPerBlock;
            const size_t slot = t % tablesPerBlock;
            memcpy(&packedBlocks[block * 64 + slot * packedTableBytes], &packedTables[t * packedTableBytes], packedTableBytes);
        }
        out.assign(numTables, 0);
    }
};

int main()
{
#if !defined(NEONGATHER_BITPACKED_ARM64)
    printf("SKIP: not built for ARM64\n");
    return 0;
#else
    const int iterationsPerThread = 200000;
    const unsigned int hwThreads = std::thread::hardware_concurrency();
    printf("hardware_concurrency() = %u\n", hwThreads);

    double baselinePerSec = 0.0;
    for (unsigned int numThreads : {1u, 2u, 4u, 6u, 8u, hwThreads})
    {
        if (numThreads == 0 || numThreads > hwThreads * 2) continue;

        std::vector<ThreadWork> work(numThreads);
        for (unsigned int t = 0; t < numThreads; ++t) work[t].init(1000 + t);

        std::atomic<uint64_t> globalSink{0};
        auto worker = [&](unsigned int t)
        {
            uint64_t localSink = 0;
            auto& w = work[t];
            for (int it = 0; it < iterationsPerThread; ++it)
            {
                w.localIndices[0] = (uint8_t)(it % w.tableSize);
                gatherPackedBitsTBL4(w.packedBlocks.data(), w.packedTableBytes, w.tablesPerBlock,
                                      w.localIndices.data(), w.out.data(), w.numTables);
                localSink += w.out[0] + w.out[w.numTables - 1];
            }
            globalSink.fetch_add(localSink, std::memory_order_relaxed);
        };

        auto t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> pool;
        for (unsigned int t = 0; t < numThreads; ++t) pool.emplace_back(worker, t);
        for (auto& th : pool) th.join();
        auto t1 = std::chrono::steady_clock::now();

        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double totalGatherBatches = (double)numThreads * iterationsPerThread;
        const double batchesPerSec = totalGatherBatches / (ms / 1000.0);
        if (numThreads == 1) baselinePerSec = batchesPerSec;
        printf("threads=%2u: %8.2f ms total, %10.0f gather-batches/sec aggregate, %6.2fx vs 1-thread  (sink=%llu)\n",
               numThreads, ms, batchesPerSec, (baselinePerSec > 0) ? batchesPerSec / baselinePerSec : 0.0,
               (unsigned long long)globalSink.load());
    }
    return 0;
#endif
}
