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

## Status (2026-09-24) — correction: single-thread wins were not reliable

Correctness is real and solid: every gather strategy below is diffed bit-exact against a scalar
reference on real Apple Silicon (Apple M6) hardware, across multiple table sizes including a
real architectural bug caught and fixed (`TBL4` outputs at most 16 lanes per call, period,
regardless of how many tables a small enough packed size could theoretically fit into 64 bytes —
found by testing table sizes other than the one bpp9000-specific case, fixed via a
`tablesPerBlockFor()` helper that caps correctly. `src/gather_bitpacked.h`, `tests/test_generic_tablesize.cpp`).

**The single-thread throughput numbers, however, were wrong.** Both `TBL4`-based approaches
(`src/gather.h` byte-packed, `src/gather_bitpacked.h` bit-packed) were first reported as real
wins (1.48x, 1.88x) from single-shot benchmark runs. Repeating each benchmark several times
in a row tells a different, more honest story: scalar settles to a stable ~50ms baseline while
both `TBL4` variants stay flat around ~55-65ms — i.e. **scalar is not clearly beaten by either
`TBL4` approach on this hardware, and by the stabilized numbers may be faster.** The first-run
"wins" were measurement artifacts (likely thermal/frequency-state settling, not yet root-caused),
not real, reproducible results. Lesson: never trust a single benchmark run, average several.

The one number that held up as a real, substantial win: **multithreading**, up to **~8.5x**
aggregate throughput at 12 threads vs 1 on this chip's three real core tiers (2 "Super" + 4
"Performance" + 6 "Efficiency") — though even that was only measured once and should be
re-verified with repeated runs before being trusted as firmly as the correctness results are.

- **`src/gather_select.h`** — a compare-and-blend alternative, genuinely and clearly rejected:
  **0.88x**, consistently worse, because its cost is O(possible-values × n) not O(n). This one
  negative result held up on repeat and is trustworthy.

**Honest current state**: this library's real, solid deliverable so far is *correct* NEON gather
emulation (including a real caught architectural bug), not yet a *proven faster* one on
single-thread throughput — that claim needs to be re-earned with a proper repeated-measurement
benchmark harness before being reported again. The techniques are still real (TBL4 table-packing,
bit-packing for narrow value widths); whether they're actually faster than well-optimized scalar
code on this exact hardware is now an open question again, not a settled one.

**Not yet built**: the x86 AVX2/AVX-512 path (x86 has real hardware gather and doesn't need any
of these tricks); a proper repeated-run benchmark harness (the real next priority before trusting
any more throughput claims); the API is also still tuned to one workload's shape (27-entry
tables, 2-bit values) rather than a fully generic library, though the table-size generalization
is now tested down to that level.

## Layout

- `src/` — the library itself (header-only)
- `tests/` — correctness tests (scalar-reference-vs-SIMD diff, real hardware, not simulated)
- `bench/` — throughput benchmarks, including the real multithreaded scaling test
