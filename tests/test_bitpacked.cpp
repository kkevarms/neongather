#include <cstdio>
#include <cstring>
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

    std::mt19937_64 rng(99999);
    std::vector<uint8_t> tables(numTables * tableSize);
    std::vector<uint8_t> localIndices(numTables);
    for (auto& b : tables) b = (uint8_t)(rng() % 3);
    for (auto& idx : localIndices) idx = (uint8_t)(rng() % tableSize);

    // Pack each table individually first.
    std::vector<uint8_t> packedTables(numTables * packedTableBytes);
    for (size_t t = 0; t < numTables; ++t)
    {
        packTrits(&tables[t * tableSize], tableSize, &packedTables[t * packedTableBytes]);
    }

    // Layout into 64-byte blocks, tablesPerBlock tables/block.
    const size_t numBlocks = (numTables + tablesPerBlock - 1) / tablesPerBlock;
    std::vector<uint8_t> packedBlocks(numBlocks * 64, 0);
    for (size_t t = 0; t < numTables; ++t)
    {
        const size_t block = t / tablesPerBlock;
        const size_t slot = t % tablesPerBlock;
        memcpy(&packedBlocks[block * 64 + slot * packedTableBytes], &packedTables[t * packedTableBytes], packedTableBytes);
    }

    std::vector<uint8_t> gathered(numTables, 0);
    gatherPackedBitsTBL4(packedBlocks.data(), packedTableBytes, tablesPerBlock, localIndices.data(), gathered.data(), numTables);

    // Scalar reference against the ORIGINAL unpacked trit values directly (ground truth,
    // independent of the packing code) -- catches bugs in packTrits itself too, not just the gather.
    std::vector<uint8_t> expected(numTables);
    for (size_t t = 0; t < numTables; ++t)
    {
        expected[t] = tables[t * tableSize + localIndices[t]];
    }

    int mismatches = 0;
    for (size_t t = 0; t < numTables; ++t)
    {
        if (gathered[t] != expected[t])
        {
            if (mismatches < 15)
            {
                printf("MISMATCH at %zu: got=%u expected=%u idx=%u (byteIdx=%zu subPos=%zu)\n",
                       t, gathered[t], expected[t], localIndices[t], (size_t)localIndices[t] / 4, (size_t)localIndices[t] % 4);
            }
            ++mismatches;
        }
    }
    if (mismatches == 0)
    {
        printf("PASS: %zu bit-packed gathers, all match the original unpacked values\n", numTables);
        return 0;
    }
    printf("FAIL: %d/%zu mismatches\n", mismatches, numTables);
    return 1;
#endif
}
