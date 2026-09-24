#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "../src/gather_bitpacked.h"

using namespace neongather;

// Proves gatherPackedBitsTBL4 is genuinely generic over table size, not secretly hardcoded to
// bpp9000's 27. Runs the identical code path against several different table sizes (19, 27, and
// a couple of edge cases) and diffs each against the raw unpacked ground truth.
#if !defined(NEONGATHER_BITPACKED_ARM64)
int main() { printf("SKIP: not built for ARM64\n"); return 0; }
#else

static bool runOne(size_t tableSize, size_t numTables, uint64_t seed)
{
    const size_t packedTableBytes = (tableSize + 3) / 4;
    const size_t tablesPerBlock = tablesPerBlockFor(packedTableBytes);
    if (tablesPerBlock == 0)
    {
        printf("tableSize=%zu: SKIP (doesn't fit even 1 table in a 64-byte block)\n", tableSize);
        return true;
    }

    std::mt19937_64 rng(seed);
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

    std::vector<uint8_t> gathered(numTables, 0);
    gatherPackedBitsTBL4(packedBlocks.data(), packedTableBytes, tablesPerBlock, localIndices.data(), gathered.data(), numTables);

    int mismatches = 0;
    for (size_t t = 0; t < numTables; ++t)
    {
        const uint8_t expected = tables[t * tableSize + localIndices[t]];
        if (gathered[t] != expected)
        {
            if (mismatches < 5)
                printf("  MISMATCH at %zu: got=%u expected=%u idx=%u\n", t, gathered[t], expected, localIndices[t]);
            ++mismatches;
        }
    }
    printf("tableSize=%2zu (packedBytes=%zu, tablesPerBlock=%2zu): %s (%d/%zu mismatches)\n",
           tableSize, packedTableBytes, tablesPerBlock, mismatches == 0 ? "PASS" : "FAIL", mismatches, numTables);
    return mismatches == 0;
}

int main()
{
    bool allOk = true;
    // 19 (the size Kevin asked about), 27 (bpp9000's real size, regression check), plus a few
    // edge cases: 1 (smallest real table), 64 (exactly fills one packed table into the whole
    // block, tablesPerBlock=1), 4 (packs to exactly 1 byte, tablesPerBlock=64 -- more lanes than
    // TBL4's 16 output lanes, exercises the multi-block loop).
    for (size_t tableSize : {1u, 4u, 19u, 27u, 33u, 64u})
    {
        if (!runOne(tableSize, 997 /* not a multiple of any tested tablesPerBlock -- exercises the tail */, 42 + tableSize))
        {
            allOk = false;
        }
    }
    printf(allOk ? "\nALL PASS\n" : "\nSOME FAILED\n");
    return allOk ? 0 : 1;
}
#endif
