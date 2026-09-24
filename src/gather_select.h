#pragma once

// Alternate strategy: select-chain instead of gather. For the bpp9000 shape specifically
// (each neuron's LUT index is one of only 27 possible values), a real gather forces loading a
// mostly-wasted 64-byte block per TBL4 call (54 of 64 bytes used, only 2 tables' worth actually
// read). This flips the problem: store the LUT data TRANSPOSED (one contiguous array per
// possible index value k, holding every neuron's entry for that index), then compute the output
// as a chain of compare+blend passes -- no gather instruction at all, full 16-lane utilization
// every pass, zero wasted bytes loaded.
//
// columns[k] must point to `n` bytes: columns[k][i] = the original table[i]'s entry at index k.
// indices[i] is neuron i's chosen index, in [0, numColumns).

#include <cstdint>
#include <cstddef>

#if defined(__aarch64__) || defined(_M_ARM64)
#define NEONGATHER_SELECT_ARM64 1
#include <arm_neon.h>
#endif

namespace neongather
{

inline void selectScalar(const uint8_t* const* columns, size_t numColumns,
                          const uint8_t* indices, uint8_t* out, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        out[i] = columns[indices[i]][i];
    }
}

#if defined(NEONGATHER_SELECT_ARM64)

// One pass per possible index value: load 16 neurons' worth of this column + their indices,
// compare index==k, blend into the running result. numColumns passes total (27 for bpp9000),
// each touching every 16-lane group of n once -- O(numColumns * n/16) vector ops, all of it
// real useful data (no padding/wasted bytes like the packed-TBL4 gather).
inline void selectChain(const uint8_t* const* columns, size_t numColumns,
                         const uint8_t* indices, uint8_t* out, size_t n)
{
    size_t i = 0;
    for (; i + 16 <= n; i += 16)
    {
        uint8x16_t idx = vld1q_u8(indices + i);
        uint8x16_t result = vdupq_n_u8(0);
        for (size_t k = 0; k < numColumns; ++k)
        {
            uint8x16_t colVal = vld1q_u8(columns[k] + i);
            uint8x16_t mask = vceqq_u8(idx, vdupq_n_u8((uint8_t)k));
            result = vbslq_u8(mask, colVal, result);
        }
        vst1q_u8(out + i, result);
    }
    // scalar tail for n not a multiple of 16
    for (; i < n; ++i)
    {
        out[i] = columns[indices[i]][i];
    }
}

#endif // NEONGATHER_SELECT_ARM64

}
