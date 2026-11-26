#pragma once

#include <type_traits>

#include "mate/flat/kda/collective/mp31_chunk_kda_backward_grad_collective_tme_warpspecialized.hpp"

namespace mate::flat::kda {

// K4 (`chunk_kda_backward_grad`) builder.  Unlike K1 the work is not
// partitioned across a persistent grid: one CTA owns one (chunk, head) pair
// and consumes the whole 128-wide value axis, so the scheduler is just the
// record -> (sequence, chunk) decode inside the collective.
template <class ArchTag_, class Element_, class CuSeqlensElement_,
          bool IsVarlen_, bool NormalizeQK_, int Chunk_, int HeadDim_,
          int ValueDim_, class ElementGate_ = Element_,
          class ElementBeta_ = Element_, bool UseBetaSigmoid_ = true,
          bool AllowNegEigval_ = false, bool SafeGate_ = true,
          bool GateIsLog_ = false, bool GateIsLinear_ = false>
struct ChunkKdaBackwardGradBuilder {
  using ArchTag = ArchTag_;
  static_assert(std::is_same_v<ArchTag, mutlass::arch::Mp31>, "chunk_kda_backward_grad is implemented for MP31 only");
  using Collective =
      backward_grad::Mp31ChunkKdaBackwardGradCollectiveTmeWarpSpecialized<
          Element_, CuSeqlensElement_, IsVarlen_, NormalizeQK_, Chunk_,
          HeadDim_, ValueDim_, ElementGate_, ElementBeta_, UseBetaSigmoid_,
          AllowNegEigval_, SafeGate_, GateIsLog_, GateIsLinear_>;
  using Kernel     = backward_grad::ChunkKdaBackwardGradKernel<Collective>;
};

}  // namespace mate::flat::kda
