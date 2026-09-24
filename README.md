# neongather

A portable SIMD gather library. Native gather (AVX2/AVX-512 on x86) mapped through directly;
NEON has no hardware gather at all, so this is really a library of the tricks needed to fake
one well on ARM.

## Why this exists

Came out of porting Qubic's bpp9000 scoring kernel to Apple Silicon (`~/mac-arm64-qiner-port/`,
`~/neongather` split out as its own thing since the need is general, not specific to that
project). The kernel's hot loop is a per-element table lookup with a data-dependent index —
textbook gather — and x86's AVX-512 miner uses real hardware gather instructions for it, while
the ARM NEON build has nothing equivalent to reach for.

## The real landscape (checked, not assumed)

- **x86 AVX2/AVX-512**: real hardware gather (`vpgatherdd`/`vpgatherqd` etc). Native, fast,
  the easy case.
- **ARM NEON** (all Apple Silicon, most ARMv8 chips): **no gather instruction at all.** The only
  options are (a) scalar loads + lane insert, or (b) `TBL`/`TBL4` — a real vector table-lookup
  instruction, but it only indexes within a small in-register table (`TBL` = 16 bytes, `TBL4` =
  64 bytes across 4 registers), not arbitrary memory.
- **ARM SVE/SVE2**: does have real hardware gather (`LD1B`/`LD1D` with gather addressing), same
  class of capability as AVX-512. **Not implemented on any Apple Silicon chip** (Apple has
  stayed on NEON/AMX, not SVE) — so even though "ARM has an answer to this," it's not available
  on the platform that actually matters here. A NEON-native approach is the real target, not a
  stopgap until SVE support shows up.

## Design

Two gather strategies, picked per call site based on the table's real size/range, not a single
one-size-fits-all abstraction:

1. **Small-table gather (table fits in ≤64 bytes, e.g. bpp9000's 27-byte-per-neuron LUT rows)**:
   `TBL4`-based. Real hardware instruction, single-digit-cycle latency, no memory round-trip.
   This is the interesting/good case and the one worth building first.
2. **Large-range gather (arbitrary memory, can't fit a lookup table in registers)**: scalar
   loads + lane insert. This is what NEON is stuck with regardless of cleverness — the value is
   in packaging it once, correctly, with a clean API, not in beating the hardware limitation.

x86 side: thin wrapper directly over `_mm256_i32gather_epi32`/AVX-512 gather intrinsics — no
cleverness needed, just API parity with the ARM side so calling code doesn't care which
platform it's on.

## Status

Scoped 2026-09-24. Not yet built — this is the design record. First real target: the
small-table (`TBL4`) gather path, validated against the bpp9000 kernel's actual access pattern
as the first real consumer.

## Layout

- `src/` — the library itself (header-only to start)
- `tests/` — correctness tests (scalar-reference-vs-SIMD diff, same discipline as the bpp9000
  cross-platform verification work)
- `bench/` — throughput benchmarks, x86 native-gather vs ARM TBL4 vs ARM scalar-fallback
