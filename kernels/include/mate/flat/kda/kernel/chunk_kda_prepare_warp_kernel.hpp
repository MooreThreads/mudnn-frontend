#pragma once

#include <musa_runtime.h>
#include <mutlass/mutlass.h>

#include <cstdint>
#include <type_traits>

#include "mate/flat/flat_options.hpp"
#include "mate/flat/kda/kda_arch_traits.hpp"
#include "mate/flat/kda/kernel/chunk_kda_tile_scheduler.hpp"

namespace mate::flat::kda {

// Warp-per-chunk prepare kernel (MP32): every warp of the CTA independently
// walks its share of the CTA's (chunk, head) records; see
// mp32_chunk_kda_prepare_collective_warp.hpp.  No CTA-level synchronization.
template <class CollectivePrepare_, class TileScheduler_>
struct ChunkKdaPrepareWarpKernel {
  using CollectivePrepare = CollectivePrepare_;
  using TileScheduler     = TileScheduler_;
  using Components        = typename CollectivePrepare::Components;
  using ArchTag           = typename CollectivePrepare::ArchTag;
  using CuSeqlensElement  = typename TileScheduler::CuSeqlensElement;
  using ProblemSize       = typename Components::PrepareProblemShape;
  using SharedStorage     = typename CollectivePrepare::SharedStorage;

  static_assert(std::is_same_v<ProblemSize, typename TileScheduler::ProblemShape>);

  static constexpr int NumWarps           = CollectivePrepare::NumWarps;
  static constexpr int SharedStorageSize  = CollectivePrepare::SharedStorageSize;
  static constexpr int SmemAlignmentBytes = CollectivePrepare::SmemAlignmentBytes;
  // The register budget of TileScheduler::CtasPerMp resident CTAs.
  static constexpr uint32_t MinBlocksPerMultiprocessor = TileScheduler::CtasPerMp;
  static constexpr int      MaxThreadsPerBlock         = CollectivePrepare::NumThreadsPerBlock;

  struct Arguments {
    ProblemSize                           problem_size;
    typename CollectivePrepare::Arguments prepare;
  };

  struct Params {
    ProblemSize                        problem_size;
    typename CollectivePrepare::Params prepare;
    typename TileScheduler::Params     scheduler;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    return Params{
        args.problem_size,
        CollectivePrepare::to_underlying_arguments(args.problem_size, args.prepare),
        TileScheduler::to_underlying_arguments(args.problem_size),
    };
  }

  MUTLASS_DEVICE void operator()(Params const& params, char* smem) {
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem);
    int const      warp           = int(threadIdx.x) / mutlass::NumThreadsPerWarp;
    int const      lane           = int(threadIdx.x) % mutlass::NumThreadsPerWarp;
    auto const&    ps             = params.problem_size;
    if constexpr (TileScheduler::IsVarlen) {
      // Every warp walks the CTA's cursor and takes every NumWarps-th record.
      TileScheduler tile_scheduler;
      auto          work_cursor = tile_scheduler.get_initial_work(params.scheduler, ps);
      for (int item = 0; work_cursor.remaining > 0; ++item) {
        __musa_loop_transparent_outermost();
        if (item % NumWarps == warp) {
          auto work_desc = tile_scheduler.get_work(params.scheduler, ps, work_cursor);
          CollectivePrepare::run_work(shared_storage, params.prepare, ps, work_desc, warp, lane);
        }
        tile_scheduler.advance(params.scheduler, ps, work_cursor);
      }
    } else {
      // Head-major contiguous split of the (head, batch, chunk) records, chunk
      // fastest, so a CTA walks runs of consecutive chunks of one (batch, head).
      int const nc         = mutlass::ceil_div(ps.T, CollectivePrepare::kChunk);
      int const total      = ps.H * ps.N * nc;
      int const partitions = int(gridDim.x);
      int const base       = total / partitions;
      int const remainder  = total - base * partitions;
      int const partition  = int(blockIdx.x);
      int       item       = partition * base + (partition < remainder ? partition : remainder);
      int const end        = item + base + int(partition < remainder);
      while (item < end) {
        __musa_loop_transparent_outermost();
        int const hb    = item / nc;
        int const chunk = item - hb * nc;
        int const head  = hb / ps.N;
        int const batch = hb - head * ps.N;
        int const run   = min(end - item, nc - chunk);
        CollectivePrepare::run_chunks(shared_storage, params.prepare, ps, batch, head, chunk, run, warp, lane);
        item += run;
      }
    }
  }
};

}  // namespace mate::flat::kda
