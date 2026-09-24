#include <cstdio>
#include <cstdlib>
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
    const size_t numColumns = 27; // real bpp9000 lutSize
    const size_t n = 1024;        // real bpp9000 populationThreshold

    std::mt19937_64 rng(54321);
    // columns[k][i] = trit value for neuron i at LUT index k
    std::vector<std::vector<uint8_t>> columnStorage(numColumns, std::vector<uint8_t>(n));
    std::vector<const uint8_t*> columns(numColumns);
    for (size_t k = 0; k < numColumns; ++k)
    {
        for (size_t i = 0; i < n; ++i) columnStorage[k][i] = (uint8_t)(rng() % 3);
        columns[k] = columnStorage[k].data();
    }
    std::vector<uint8_t> indices(n);
    for (auto& idx : indices) idx = (uint8_t)(rng() % numColumns);

    std::vector<uint8_t> expected(n), got(n);
    selectScalar(columns.data(), numColumns, indices.data(), expected.data(), n);
    selectChain(columns.data(), numColumns, indices.data(), got.data(), n);

    int mismatches = 0;
    for (size_t i = 0; i < n; ++i)
    {
        if (got[i] != expected[i])
        {
            if (mismatches < 10) printf("MISMATCH at %zu: got=%u expected=%u idx=%u\n", i, got[i], expected[i], indices[i]);
            ++mismatches;
        }
    }
    if (mismatches == 0)
    {
        printf("PASS: %zu selects, all match scalar reference\n", n);
        return 0;
    }
    printf("FAIL: %d/%zu mismatches\n", mismatches, n);
    return 1;
#endif
}
