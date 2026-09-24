#pragma once

// neongather: portable small-table gather.
//
// The real shape of the problem this targets (from the bpp9000 kernel, the first real
// consumer): N independent lookups, each into its OWN small table (bpp9000: 27 bytes/table),
// each with a single computed index. That's NOT "many indices into one shared table" (which is
// what a naive TBL reach-for would assume) -- it's a batch of independent tiny gathers.
//
// x86 (AVX2/AVX-512): real hardware gather handles arbitrary per-lane base addresses directly.
// No packing needed -- gather straight from each table's own memory location.
//
// ARM NEON: no hardware gather at all. TBL/TBL4 only index within ONE shared in-register table
// (16/64 bytes). The real trick used here: pack multiple small tables contiguously into one
// 64-byte buffer, then offset each lane's local index by its table's start position within that
// buffer -- TBL4 can then gather across the WHOLE 64-byte combined buffer per-lane in one call.
// With 27-byte tables, floor(64/27) = 2 tables per TBL4 call.

#include <cstdint>
#include <cstddef>
#include <cstring>

#if defined(__aarch64__) || defined(_M_ARM64)
#define NEONGATHER_ARM64 1
#include <arm_neon.h>
#elif defined(__AVX2__)
#define NEONGATHER_AVX2 1
#include <immintrin.h>
#endif

namespace neongather
{

// Scalar reference: always correct, the oracle every SIMD path is tested against.
// tables[i] points at lane i's own table; indices[i] is lane i's index into it.
inline void gatherScalar(const uint8_t* const* tables, const uint8_t* indices, uint8_t* out, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        out[i] = tables[i][indices[i]];
    }
}

#if defined(NEONGATHER_ARM64)

// Packed-table gather: `tableSize` bytes/table, `tablesPerBlock` tables packed contiguously in
// each 64-byte block of `packedTables` (tablesPerBlock * tableSize must be <= 64). `localIndices`
// are per-lane indices *within their own table* (i.e. in [0, tableSize)), laid out lane-major
// within each block of `tablesPerBlock` lanes. Processes n lanes, n must be a multiple of
// tablesPerBlock. One TBL4 call handles one full block (tablesPerBlock gathers at once).
inline void gatherPackedTBL4(const uint8_t* packedTables, size_t tableSize, size_t tablesPerBlock,
                              const uint8_t* localIndices, uint8_t* out, size_t n)
{
    const size_t blockBytes = 64;

    // Per-lane table offset (i*tableSize for lane i, 0xFF padding for unused lanes) -- built
    // ONCE, not per block, since it only depends on tableSize/tablesPerBlock, not the data.
    // Lanes >= tablesPerBlock use an out-of-range index (TBL emits 0 for those; unused/unread).
    uint8_t offsetBuf[16];
    for (size_t i = 0; i < 16; ++i)
    {
        offsetBuf[i] = (i < tablesPerBlock) ? (uint8_t)(i * tableSize) : 0xFF;
    }
    const uint8x16_t offsetVec = vld1q_u8(offsetBuf);

    size_t base = 0;
    // Fast path: full 16-lane blocks of localIndices, one vector add for the whole block's
    // indices instead of a scalar per-lane loop.
    for (; base + 16 <= n; base += 16)
    {
        // Only meaningful when tablesPerBlock == 16 (one TBL4 call covers exactly 16 lanes,
        // i.e. exactly one packed block); the general multi-block case falls through below.
        if (tablesPerBlock != 16) break;
        const uint8_t* block = packedTables + (base / tablesPerBlock) * blockBytes;
        uint8x16x4_t table = vld1q_u8_x4(block);
        uint8x16_t localIdx = vld1q_u8(localIndices + base);
        uint8x16_t idx = vaddq_u8(localIdx, offsetVec);
        uint8x16_t gathered = vqtbl4q_u8(table, idx);
        vst1q_u8(out + base, gathered);
    }

    // General path: one TBL4 call per block of `tablesPerBlock` lanes (tablesPerBlock < 16,
    // the common bpp9000-shaped case). Still avoids the outBuf memcpy for the common
    // tablesPerBlock in {1,2,4,8} sizes via a direct narrow store.
    for (; base < n; base += tablesPerBlock)
    {
        const uint8_t* block = packedTables + (base / tablesPerBlock) * blockBytes;
        uint8x16x4_t table = vld1q_u8_x4(block);

        uint8_t localBuf[16] = {0};
        const size_t count = (n - base < tablesPerBlock) ? (n - base) : tablesPerBlock;
        memcpy(localBuf, localIndices + base, count);
        uint8x16_t localIdx = vld1q_u8(localBuf);
        uint8x16_t idx = vaddq_u8(localIdx, offsetVec);
        uint8x16_t gathered = vqtbl4q_u8(table, idx);

        if (count == 16)
        {
            vst1q_u8(out + base, gathered);
        }
        else if (count >= 8)
        {
            vst1_u8(out + base, vget_low_u8(gathered));
            uint8_t rest[8];
            vst1_u8(rest, vget_high_u8(gathered));
            memcpy(out + base + 8, rest, count - 8);
        }
        else
        {
            uint8_t low[8];
            vst1_u8(low, vget_low_u8(gathered));
            memcpy(out + base, low, count);
        }
    }
}

#endif // NEONGATHER_ARM64

#if defined(NEONGATHER_AVX2)

// x86: real hardware gather handles arbitrary per-lane base addresses directly -- no table
// packing needed. baseAddrOffsets[i] is table[i] - tables[0] cast to int32 (gather needs a
// single base pointer + per-lane byte offsets); caller supplies that, or use gatherScalar-style
// per-lane tables directly since AVX2's 32-bit gather is the natural fit here anyway.
inline void gatherAVX2(const uint8_t* base, const int32_t* byteOffsets, uint8_t* out, size_t n)
{
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
    {
        __m256i offsets = _mm256_loadu_si256((const __m256i*)(byteOffsets + i));
        __m256i gathered = _mm256_i32gather_epi32((const int*)base, offsets, 1);
        // gather_epi32 reads 4 bytes per lane from (base + offset); we only want 1 byte/lane,
        // so mask down to the low byte of each 32-bit lane.
        alignas(32) int32_t tmp[8];
        _mm256_storeu_si256((__m256i*)tmp, gathered);
        for (int lane = 0; lane < 8; ++lane)
        {
            out[i + lane] = (uint8_t)(tmp[lane] & 0xFF);
        }
    }
    for (; i < n; ++i)
    {
        out[i] = *(base + byteOffsets[i]);
    }
}

#endif // NEONGATHER_AVX2

}
