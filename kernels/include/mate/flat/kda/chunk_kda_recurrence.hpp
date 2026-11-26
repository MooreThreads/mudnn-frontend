#pragma once

#include <type_traits>

#include "mate/flat/flat_options.hpp"
#include "mate/flat/kda/chunk_kda_components.hpp"
#include "mate/flat/kda/chunk_kda_dispatch_policy.hpp"
#include "mate/flat/kda/collective/mp31_chunk_kda_recurrence_collective_tme_warpspecialized.hpp"
#include "mate/flat/kda/kda_arch_traits.hpp"
#include "mate/flat/kda/kernel/chunk_kda_recurrence_kernel.hpp"
#include "mate/flat/kda/kernel/chunk_kda_tile_scheduler.hpp"

namespace mate::flat::kda {

// K2 (recurrence) builder.  The two architectures need architecture-specific
// collectives -- MP32 widens the Q/P MMA to 2*kChunk rows, loads the
// inverse/P staging tile with plain loads and lays the decayed tile out as
// Q[0:kChunk], K[0:kChunk] -- so ArchTag selects between the collectives while
// the problem/workspace contract stays shared. MP32 assigns each CTA one
// recurrence chain directly; MP31 retains its persistent scheduler.
//
// MP31 has one recurrence collective in two layouts: a CTA a 64-column value
// slice, or (Tag::Mp31WholeHeadRecurrence) a CTA a whole head with an output
// and a state squad per slice.  The host picks one per launch from the grid
// (see chunk_kda_mp31_policy.hpp).
//
// MP32 has two recurrence collectives: the scan collective, which owns a head's
// whole 128-column value slice with four 32-column scan squads and keeps the
// state in registers, and the chunk-state collective, which owns two 64-column
// slices a head with separate state and output squads and a shared-memory state
// tile -- the one a chunk-state snapshot exports from.  The scan collective does
// not export chunk states, so the feature is part of the selection and not a
// runtime branch of either collective.
template <class ArchTag_,
          class Element_,
          class StateElement_,
          class CuSeqlensElement_,
          class LogicalShape_,
          class TileShape_,
          class StrideV_,
          class StrideO_,
          class... Options_>
struct ChunkKdaRecurrenceBuilder {
  using ArchTag                = ArchTag_;
  static constexpr bool IsMp32 = arch_traits<ArchTag>::is_mp32;
  static constexpr bool HasChunkStatesOut =
      mate::flat::find_option_t<mate::flat::Tag::HasChunkStatesOut, std::false_type, Options_...>::value;
  // Without a chunk-state export MP32 runs the scan collective.
  static constexpr bool UseMp32Scan = IsMp32 && !HasChunkStatesOut;
  static constexpr bool UseMp31WholeHead =
      !IsMp32 &&
      mate::flat::find_option_t<mate::flat::Tag::Mp31WholeHeadRecurrence, std::false_type, Options_...>::value;
  using Mp31SchedulePolicy =
      std::conditional_t<UseMp31WholeHead, ChunkKdaMp31WholeHeadRecurrenceSchedulePolicy, ChunkKdaMp31SchedulePolicy>;
  using DispatchPolicy = ChunkKdaRecurrenceDispatchPolicy<ArchTag, HasChunkStatesOut, UseMp31WholeHead>;
  using SchedulePolicy = typename DispatchPolicy::SchedulePolicy;
  using Components = ChunkKdaPrefillComponents<Element_, CuSeqlensElement_, LogicalShape_, SchedulePolicy, Options_...>;
  using CollectiveRecurrence =
      typename DispatchPolicy::template Collective<Components, TileShape_, StateElement_, StrideV_, StrideO_>;
  using TileScheduler = typename Components::RecurrenceTileScheduler;
  using Kernel        = ChunkKdaRecurrenceKernel<CollectiveRecurrence, TileScheduler>;
};

}  // namespace mate::flat::kda
