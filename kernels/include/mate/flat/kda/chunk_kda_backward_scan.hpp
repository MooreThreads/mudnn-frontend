#pragma once

#include <musa_runtime.h>
#include <mutlass/device_kernel.h>
#include <mutlass/mutlass.h>

#include <cstdint>

#include "mate/flat/kda/chunk_kda_problem.hpp"
#include "mate/flat/kda/collective/mp31_chunk_kda_backward_scan_collective_tme_warpspecialized.hpp"

namespace mate::flat::kda {

// MP31 K3: the reverse chunk scan that carries dS and publishes the
// per-chunk `dstate` pool plus `dstate0`.  Four kernels make up the backward
// (K1 prepare, K2 recurrence, K3 scan, K4 grads); this one owns only the
// sequential part, so it takes the workspace K1 wrote, the dO gradient and the
// optional d_final_state seed, and produces nothing else.
template <class Collective_>
struct ChunkKdaBackwardScanKernel {
  using CollectiveScan           = Collective_;
  using Element                  = typename CollectiveScan::Element;
  using CuSeqlensElement         = typename CollectiveScan::CuSeqlensElement;
  static constexpr int kChunk    = CollectiveScan::C;
  static constexpr int kHeadDim  = CollectiveScan::D;
  static constexpr int kValueDim = CollectiveScan::V;

  static constexpr int SharedStorageSize  = CollectiveScan::SharedStorageSize;
  static constexpr int MaxThreadsPerBlock = CollectiveScan::MaxThreadsPerBlock;
  static constexpr int SmemAlignmentBytes = CollectiveScan::SmemAlignmentBytes;
  // This launch bound controls the compiler's register reservation. Two is
  // the measured choice for the carried (64 x 128) fp32 accumulator: the
  // resulting kernel admits three CTAs per MP, while stronger bounds add
  // spilling without improving scan throughput.
  static constexpr uint32_t MinBlocksPerMultiprocessor = 2;

  struct Arguments {
    typename CollectiveScan::Arguments scan;
  };
  struct Params {
    typename CollectiveScan::Params scan;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    return Params{CollectiveScan::to_underlying_arguments(args.scan)};
  }

  static dim3 get_grid_shape(Params const& params) {
    return CollectiveScan::get_grid_shape(params.scan);
  }

  MUTLASS_DEVICE void operator()(Params const& params, char* smem) const {
    CollectiveScan::run(params.scan, smem);
  }
};

// ArchTag is a template parameter for interface parity with K1/K2's builders;
// only MP31 implements this scan today.
template <class ArchTag_, class Element_, class CuSeqlensElement_, class LogicalShape_, bool IsVarlen_>
struct ChunkKdaBackwardScanBuilder {
  using ArchTag = ArchTag_;
  static_assert(std::is_same_v<ArchTag, mutlass::arch::Mp31>, "chunk_kda_backward_scan currently implements only MP31");
  using LogicalShape = LogicalShape_;
  using Collective   = backward_scan::Mp31ChunkKdaBackwardScanCollective<Element_,
                                                                         CuSeqlensElement_,
                                                                         IsVarlen_,
                                                                         LogicalShape_::Chunk,
                                                                         LogicalShape_::HeadDim>;
  using Kernel       = ChunkKdaBackwardScanKernel<Collective>;
};

}  // namespace mate::flat::kda
