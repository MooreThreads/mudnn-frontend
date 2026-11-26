#pragma once

#include "mate/flat/kda/chunk_kda_schedule_policy.hpp"

namespace mate::flat::kda {

// MP31 occupancy/resource choices.  They are injected into the common work
// traversal instead of being treated as KDA semantics.
// prepare CTAs/MP=5, recurrence CTAs/MP=3, value tile=64,
// metadata threads=512, metadata dynamic smem limit=192 KiB.
//
// The recurrence's register demand is dominated by its 2048-bit SQMMA state
// accumulator (a 64x128 fp32 fragment, 64 dword per thread in the one state
// squad), so at recurrence CTAs/MP = 3 the kernel already uses 107-108 of the
// 113 dword/thread the launch bound allows.  A 5.1.0-line backend cannot
// allocate those accumulators at all ("TEMP_2048"); the upstream driver defines
// MATE_KDA_MP31_REC_CTAS_PER_MP=2 for such compilers.  Grid and cap move
// together by construction: lowering only the cap measured no better at H=64/84
// and worse at H=168.
#if defined(MATE_KDA_MP31_REC_CTAS_PER_MP)
inline constexpr int kMp31RecurrenceCtasPerMp = MATE_KDA_MP31_REC_CTAS_PER_MP;
#else
inline constexpr int kMp31RecurrenceCtasPerMp = 3;
#endif

struct ChunkKdaMp31SchedulePolicy : ChunkKdaPrefillSchedulePolicy<5, kMp31RecurrenceCtasPerMp, 64, 512, 192 * 1024> {};

// The whole-head recurrence: one CTA owns both 64-column value slices of a
// head, one producer squad plus an output and a state squad per slice (640
// threads).  MP31 allocates registers per 128-thread warp squad, so two such
// CTAs an MP get 102 dword/thread and keep four value slices resident per MP
// where the one-slice layout above keeps three.  Each CTA's slices share the
// producer's key-side loads.  It needs the 5.3.0-line backend: with the
// one-slice layout at CTAs/MP = 3 as the marker, an older compiler builds only
// the one-slice recurrence.
inline constexpr bool kMp31WholeHeadRecurrence = kMp31RecurrenceCtasPerMp >= 3;

struct ChunkKdaMp31WholeHeadRecurrenceSchedulePolicy : ChunkKdaPrefillSchedulePolicy<5, 2, 128, 512, 192 * 1024> {};

// Host choice between the two recurrences from the one-slice grid (two CTAs a
// head and sequence), measured at T=8192 and on varlen batches over 56 MPs:
//  - The one-slice grid fits in one wave (H<=84): one-slice.  A whole-head CTA
//    puts both of its slices on one MP, 0.96 against 0.84 ms at H=84.
//  - The whole-head grid fits in one wave and the one-slice grid does not:
//    whole-head, H=96 and 112 run K2 in 0.97 and 1.00 ms instead of 1.46 and
//    1.50.
//  - More than two whole-head waves: whole-head, its four resident slices an MP
//    against three outweigh the partial last wave (H=252: 2.68 against 2.99 ms,
//    H=336: 3.52 against 3.83, six varlen sequences at H=96: 1.07 against 1.22).
//  - In between, the whole-head layout's partial second wave costs more
//    (H=224: 2.57 against 2.35 ms) and the one-slice layout is kept.
inline bool mp31_use_whole_head_recurrence(int slice_ctas, int mp_count) {
  if constexpr (!kMp31WholeHeadRecurrence) {
    return false;
  } else {
    if (mp_count <= 0 || slice_ctas <= 0) {
      return false;
    }
    long slice_wave = long(ChunkKdaMp31SchedulePolicy::RecurrenceCtasPerMp) * mp_count;
    long head_wave  = long(ChunkKdaMp31WholeHeadRecurrenceSchedulePolicy::RecurrenceCtasPerMp) * mp_count;
    long head_ctas  = (long(slice_ctas) + 1) / 2;
    if (slice_ctas <= slice_wave) {
      return false;
    }
    return head_ctas <= head_wave || head_ctas > 2 * head_wave;
  }
}

}  // namespace mate::flat::kda
