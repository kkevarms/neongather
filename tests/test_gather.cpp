#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "../src/gather.h"

using namespace neongather;

// Correctness test for the packed-table TBL4 gather: real bpp9000 shape (27-byte tables,
// 2 tables/64-byte block), random tables and indices, diffed against the scalar reference.
int main()
{
#if !defined(NEONGATHER_ARM64)
    printf("SKIP: not built for ARM64 (no TBL4 path to test)\n");
    return 0;
#else
    const size_t tableSize = 27;
    const size_t tablesPerBlock = 2; // floor(64/27) = 2
    const size_t numTables = 1024;   // bpp9000's real populationThreshold

    std::mt19937_64 rng(12345);
    std::vector<uint8_t> tables(numTables * tableSize);
    std::vector<uint8_t> localIndices(numTables);
    for (auto& b : tables) b = (uint8_t)(rng() % 3); // real bpp9000 values are trits {0,1,2}
    for (auto& idx : localIndices) idx = (uint8_t)(rng() % tableSize);

    // Packed layout: numTables must be a multiple of tablesPerBlock for this test.
    const size_t numBlocks = (numTables + tablesPerBlock - 1) / tablesPerBlock;
    std::vector<uint8_t> packed(numBlocks * 64, 0);
    for (size_t t = 0; t < numTables; ++t)
    {
        const size_t block = t / tablesPerBlock;
        const size_t slot = t % tablesPerBlock;
        memcpy(&packed[block * 64 + slot * tableSize], &tables[t * tableSize], tableSize);
    }

    std::vector<uint8_t> gathered(numTables, 0);
    gatherPackedTBL4(packed.data(), tableSize, tablesPerBlock, localIndices.data(), gathered.data(), numTables);

    // Scalar reference: build per-table pointers into the ORIGINAL (unpacked) tables array.
    std::vector<const uint8_t*> tablePtrs(numTables);
    for (size_t t = 0; t < numTables; ++t) tablePtrs[t] = &tables[t * tableSize];
    std::vector<uint8_t> expected(numTables, 0);
    gatherScalar(tablePtrs.data(), localIndices.data(), expected.data(), numTables);

    int mismatches = 0;
    for (size_t t = 0; t < numTables; ++t)
    {
        if (gathered[t] != expected[t])
        {
            if (mismatches < 10)
            {
                printf("MISMATCH at %zu: got=%u expected=%u (idx=%u)\n",
                       t, gathered[t], expected[t], localIndices[t]);
            }
            ++mismatches;
        }
    }

    if (mismatches == 0)
    {
        printf("PASS: %zu gathers, all match scalar reference\n", numTables);
        return 0;
    }
    printf("FAIL: %d/%zu mismatches\n", mismatches, numTables);
    return 1;
#endif
}
