#pragma once

// Bit-packed variant of the packed-table TBL4 gather. bpp9000's LUT values are trits {0,1,2},
// needing only 2 bits, not a full byte. Packing 4 trits/byte shrinks a 27-trit table from 27
// bytes to ceil(27/4)=7 bytes, letting floor(64/7)=9 tables fit in one 64-byte TBL4 block
// instead of 2 -- using far more of the fixed 16-lane budget per call.
//
// Real extra cost this adds vs the plain byte-packed version: after the TBL4 gather returns a
// packed byte (4 trits) per lane, each lane needs its OWN 2-bit sub-extraction (shift by 0, 2, 4,
// or 6 bits depending on localIndex%4) -- done with vshlq_u8's per-lane VARIABLE shift, a real
// NEON instruction built for exactly this ("shift each lane by its own lane's value").

#include <cstdint>
#include <cstddef>
#include <cstring>

#if defined(__aarch64__) || defined(_M_ARM64)
#define NEONGATHER_BITPACKED_ARM64 1
#include <arm_neon.h>
#endif

namespace neongather
{

// Pack `n` trit values (each 0/1/2, one per byte in `trits`) into ceil(n/4) bytes, 4 trits/byte,
// trit i at bit position (i%4)*2 of byte i/4.
inline void packTrits(const uint8_t* trits, size_t n, uint8_t* packedOut)
{
    const size_t packedBytes = (n + 3) / 4;
    memset(packedOut, 0, packedBytes);
    for (size_t i = 0; i < n; ++i)
    {
        packedOut[i / 4] |= (uint8_t)((trits[i] & 0x3) << ((i % 4) * 2));
    }
}

inline uint8_t unpackTrit(const uint8_t* packed, size_t index)
{
    return (uint8_t)((packed[index / 4] >> ((index % 4) * 2)) & 0x3);
}

inline void gatherScalarBitpacked(const uint8_t* const* packedTables, const uint8_t* indices, uint8_t* out, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        out[i] = unpackTrit(packedTables[i], indices[i]);
    }
}

#if defined(NEONGATHER_BITPACKED_ARM64)

// `packedTableBytes` = ceil(tableSize/4) (7 for bpp9000's 27-trit tables). `tablesPerBlock` =
// floor(64/packedTableBytes) (9 for bpp9000). `localIndices` are per-lane indices into the
// ORIGINAL (unpacked) table range [0, tableSize).
// One TBL4 call outputs exactly 16 lanes -- a HARD architectural ceiling, independent of how
// many tables the 64-byte packed buffer could theoretically hold. A caller that packs more than
// 16 tables into one block (real for small packedTableBytes, e.g. 1-4 byte tables) and passes
// that raw density straight through gets silent wrong answers past lane 15 -- found live by the
// generic-table-size test (tableSize 1 and 4 both failed this way before this fix). Always run
// the packing math through this helper instead of computing floor(64/packedTableBytes) directly.
inline size_t tablesPerBlockFor(size_t packedTableBytes)
{
    const size_t fitsInBlock = 64 / packedTableBytes;
    return (fitsInBlock < 16) ? fitsInBlock : 16;
}

inline void gatherPackedBitsTBL4(const uint8_t* packedBlocks, size_t packedTableBytes, size_t tablesPerBlock,
                                  const uint8_t* localIndices, uint8_t* out, size_t n)
{
    const size_t blockBytes = 64;

    // Per-lane table byte-offset within the block (i*packedTableBytes for lane i), built once.
    // tablesPerBlock must be <= 16 (see tablesPerBlockFor) -- a caller passing a larger value here
    // is a bug at the call site, not something this function can safely correct for.
    uint8_t offsetBuf[16];
    for (size_t i = 0; i < 16; ++i)
    {
        offsetBuf[i] = (i < tablesPerBlock) ? (uint8_t)(i * packedTableBytes) : 0xFF;
    }
    const uint8x16_t offsetVec = vld1q_u8(offsetBuf);

    for (size_t base = 0; base < n; base += tablesPerBlock)
    {
        const uint8_t* block = packedBlocks + (base / tablesPerBlock) * blockBytes;
        uint8x16x4_t table = vld1q_u8_x4(block);

        uint8_t localBuf[16] = {0};
        const size_t count = (n - base < tablesPerBlock) ? (n - base) : tablesPerBlock;
        memcpy(localBuf, localIndices + base, count);
        uint8x16_t localIdx = vld1q_u8(localBuf);

        // byteIdx = localIdx / 4 (>>2), subPos = localIdx % 4 (& 3)
        uint8x16_t byteIdx = vshrq_n_u8(localIdx, 2);
        uint8x16_t subPos = vandq_u8(localIdx, vdupq_n_u8(0x3));

        uint8x16_t gatherIdx = vaddq_u8(byteIdx, offsetVec);
        uint8x16_t packedByte = vqtbl4q_u8(table, gatherIdx);

        // shift right by subPos*2 -- vshlq_u8 takes a per-lane SIGNED shift amount, negative = right shift.
        int8x16_t shiftAmount = vnegq_s8(vreinterpretq_s8_u8(vshlq_n_u8(subPos, 1))); // -(subPos*2)
        uint8x16_t shifted = vshlq_u8(packedByte, shiftAmount);
        uint8x16_t result = vandq_u8(shifted, vdupq_n_u8(0x3));

        uint8_t outBuf[16];
        vst1q_u8(outBuf, result);
        memcpy(out + base, outBuf, count);
    }
}

#endif // NEONGATHER_BITPACKED_ARM64

}
