#pragma once

#include <mute/tensor.hpp>
#include <mutlass/gemm/collective/collective_builder.hpp>

namespace mate::flat::kda {

// A type-only recipe for KDA's shared/shared SQMMA operands.  MUTLASS remains
// the source of truth for the instruction, TiledMma and swizzled SMEM layouts.
// Register/shared MP32 atoms are deliberately NOT built through this recipe.
//
// BuilderStages describes the MUTLASS collective used for layout derivation,
// not the number of live KDA pipeline stages.  KDA re-stages those layouts at
// the consumer and never executes the generated GEMM collective's mainloop.
template <class ArchTag, class Element, int Alignment, class BuilderStages, class KernelSchedule>
struct ChunkKdaMmaPolicy {
  static_assert(Alignment > 0);
  static_assert(BuilderStages::value >= 2, "the MUTLASS SQMMA layout builder needs at least two stages");

  template <class TileShape, class LayoutA, class LayoutB>
  using Collective = typename mutlass::gemm::collective::CollectiveBuilder<ArchTag,
                                                                           mutlass::arch::OpClassTensorOp,
                                                                           Element,
                                                                           LayoutA,
                                                                           Alignment,
                                                                           Element,
                                                                           LayoutB,
                                                                           Alignment,
                                                                           float,
                                                                           TileShape,
                                                                           mute::Shape<mute::_1, mute::_1, mute::_1>,
                                                                           BuilderStages,
                                                                           KernelSchedule>::CollectiveOp;
};

}  // namespace mate::flat::kda
