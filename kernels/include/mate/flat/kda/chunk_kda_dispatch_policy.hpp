#pragma once

#include <mutlass/arch/arch.h>

#include <type_traits>

#include "mate/flat/kda/chunk_kda_mp31_policy.hpp"

namespace mate::flat::kda {

// Forward declarations keep dispatch inspectable by a host-only compiler.
// Definitions are included by the public prepare/recurrence builders.
namespace prepare {
template <class Components, class TileShape, class StrideQ, class StrideK, class StrideG>
struct Mp31ChunkKdaPrepareCollectiveTmeWarpSpecialized;
}  // namespace prepare

namespace recurrence {
template <class Components, class TileShape, class StateElement, class StrideV, class StrideO>
struct Mp31ChunkKdaRecurrenceCollectiveTmeWarpSpecialized;
}  // namespace recurrence

template <class Collective, class TileScheduler>
struct ChunkKdaPrepareKernel;
template <class Collective, class TileScheduler>
struct ChunkKdaPrepareWarpKernel;

// Static Strategy / Abstract Factory: select a compatible resource policy,
// collective and kernel shell together, not with independent conditionals.
// An unsupported architecture deliberately has no fallback specialization.
template <class ArchTag>
struct ChunkKdaPrepareDispatchPolicy;

template <>
struct ChunkKdaPrepareDispatchPolicy<mutlass::arch::Mp31> {
  using SchedulePolicy = ChunkKdaMp31SchedulePolicy;

  template <class Components, class TileShape, class StrideQ, class StrideK, class StrideG>
  using Collective =
      prepare::Mp31ChunkKdaPrepareCollectiveTmeWarpSpecialized<Components, TileShape, StrideQ, StrideK, StrideG>;

  template <class CollectivePrepare, class TileScheduler>
  using Kernel = ChunkKdaPrepareKernel<CollectivePrepare, TileScheduler>;
};


template <class ArchTag, bool HasChunkStatesOut, bool WholeHead = false>
struct ChunkKdaRecurrenceDispatchPolicy;

template <bool HasChunkStatesOut, bool WholeHead>
struct ChunkKdaRecurrenceDispatchPolicy<mutlass::arch::Mp31, HasChunkStatesOut, WholeHead> {
  using SchedulePolicy =
      std::conditional_t<WholeHead, ChunkKdaMp31WholeHeadRecurrenceSchedulePolicy, ChunkKdaMp31SchedulePolicy>;

  template <class Components, class TileShape, class StateElement, class StrideV, class StrideO>
  using Collective = recurrence::
      Mp31ChunkKdaRecurrenceCollectiveTmeWarpSpecialized<Components, TileShape, StateElement, StrideV, StrideO>;
};



}  // namespace mate::flat::kda
