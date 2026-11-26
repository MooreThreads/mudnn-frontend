#pragma once

#include <musa_runtime.h>
#include <mutlass/mutlass.h>

#include <cstdint>
#include <mute/arch/tce_desc.hpp>
#include <mute/tensor.hpp>

#include "mate/flat/simd_helper.hpp"

namespace mate::flat::collective {

using namespace mute;

// Lower-triangular inverse of a Block x Block matrix, solved one column per
// thread, so Block must fit within a single warp.
//
// The solved columns live in shared_storage.solution - one padded row per
// column - and are drained by publish_solution().  This layout is fixed, it is
// not a tuning knob, and there are three reasons for it, in decreasing order of
// how easy they are to get wrong:
//
//  1. The working vector lives in shared memory, one padded row per column
//     (1280 bytes), and that is no longer a register decision: re-measured on
//     this kernel after the fold, the register form and the shared-memory form
//     land on the SAME entry resources - Temp 69 on MP32, Temp 64 on MP31, no
//     spill, no local memory, no stack frame - because the register peak now
//     belongs to another role (the consumer squad's fold/scale path).  The
//     original measurement, taken before the fold, was 62 registers in shared
//     memory vs 72 in registers, i.e. 8 vs 7 resident CTAs on MP32.  Shared
//     memory also wins on shape: the substitution takes its float4 operand from
//     one LDS.128, while a register vector has to be assembled out of four
//     registers first, and the register form measured ~0.25-0.34% SLOWER on MP31
//     (T=8192/H=96, 10 interleaved rounds).  And the tile cannot be dropped
//     anyway: only publish_solution reads the lane's own column, while A_rec's B
//     operand and the fold's A operand read other lanes' columns - and on MP32
//     the fold reads them from the consumer warps, not from the solving warp.
//
//  2. The unrolled loops below are what make the shared-memory form cheap: with
//    1280 extra bytes free, do NOT let a loop bound depend on col.  "The
//     solution of row < col is zero, so start the row walk at col" is correct
//     algebra and a 4.32x regression: a runtime bound stops the vector from
//     being promoted, the compiler demotes it to a stack frame
//     (private_memory_size 0 -> 64, plus an LSU.ST.B512 of all 16 floats per
//     row), and the kernel goes from 0.2198 ms to 0.9484 ms.  Skipping the
//     leading zeros buys nothing anyway - the warp issues the deepest lane's
//     trip count, and lane col = 0 still walks all Block rows.
//
//  3. publish_solution() is split out of solve_columns() on purpose.  Folding
//     the global store back into the solve keeps the 64-bit output pointer live
//     across the whole substitution, which measured two more registers than
//     computing the address afterwards.
//
// Beta is a row scale, and it is NOT part of the matrix the KKT writers store:
// they commit the bare GEMM product and solve_columns() stages each row's beta
// on that row's own diagonal, where the substitution reads it.  Solving
//   (diag(1/beta) + tril(KKT, -1)) x = I
// that way yields exactly A^-1 diag(beta) for A = I + diag(beta) tril(KKT, -1) -
// the same tile the previous form produced by pre-multiplying every
// off-diagonal element - while keeping a dependent global beta load, its
// sigmoid and its per-element shuffle out of the writers.  The diagonal is
// otherwise dead: the substitution only reads the strictly lower part, and the
// in-vector reads that do span the diagonal multiply it by a solution element
// that is still zero.
template <class Element_, int Block_>
struct CollectiveLowerTriangularInverseNxN {
  using Element = Element_;

  static constexpr int Block   = Block_;
  static constexpr int VecSize = 4;

  using SmemLayout = decltype(make_layout(make_shape(Int<Block>{}, Int<Block>{}), make_stride(Int<Block>{}, _1{})));

  // One row of the solve tile per column, padded by a full vector: lane c then
  // reads its float4s from bank group (5 * c) mod 8, so the eight lanes of a
  // 256-byte phase never share a bank group.
  static constexpr int SolutionStride = Block + VecSize;
  static constexpr int SolutionWords  = Block * SolutionStride;

  struct SharedStorage {
    alignas(16) float matrix[cosize_v<SmemLayout>];
    // The solve's working vector: one padded row per column, i.e. lane c owns
    // shared_storage.solution[c * SolutionStride .. c * SolutionStride + Block].
    alignas(16) float solution[SolutionWords];
  };

  static_assert(Block > 0);
  static_assert(Block <= mutlass::NumThreadsPerWarp);
  static_assert(Block % VecSize == 0);

  // Forward substitution of one column: the row scale is a *multiplier*, taken
  // from the matrix diagonal, and the right-hand side is the identity:
  //   solution[row] = matrix[row][row] * (delta(row,col) - sum_{k<row} KKT[row][k] * solution[k])
  // i.e. the diagonal holds beta and the system is (diag(1/beta) + L) x = e_col,
  // whose column is A^-1 diag(beta) e_col for A = I + diag(beta) L (see the beta
  // note above).  solution is the column's row of the tile, so every access
  // below is an immediate offset - see note 2 in the header before touching
  // either trip count.
  template <class TensorMatrix>
  MUTLASS_DEVICE static void solve_column(TensorMatrix const& matrix, int col, float* solution) {
    MUTE_UNROLL
    for (int row = 0; row < Block; ++row) {
      solution[row] = 0.0f;
    }

    MUTE_UNROLL
    for (int row = 0; row < Block; ++row) {
      float value = row == col ? -1.0f : 0.0f;

      MUTE_UNROLL
      for (int k = 0; k < row; k += VecSize) {
        float4 coeffs = simd::load_float4(&matrix(row, k));
        float4 sol    = *reinterpret_cast<float4 const*>(solution + k);
        simd::dot4(coeffs, sol, value);
      }

      solution[row] = -matrix(row, row) * value;
    }
  }

  // Solves every column into shared_storage.solution.  Only lanes [0, Block)
  // participate, which is why Block must fit in one warp.
  //
  // This entry performs no rendezvous of its own: the caller has already ordered
  // the matrix writers (its own wait on the KKT hand-off) before calling.  It
  // does own one warp-local step - staging the per-row beta on the diagonal,
  // which is the row scale solve_column reads - and the __syncwarp() that
  // publishes it to the lanes that read other rows' diagonals.
  MUTLASS_DEVICE void solve_columns(SharedStorage& shared_storage, float beta_lane, int local_tid) const {
    auto matrix = make_tensor(make_smem_ptr(shared_storage.matrix), SmemLayout{});
    if (local_tid < Block) {
      matrix(local_tid, local_tid) = beta_lane;
    }
    __syncwarp();

    if (local_tid < Block) {
      int    col  = local_tid;
      float* tile = shared_storage.solution + col * SolutionStride;
      solve_column(matrix, col, tile);
    }
  }

  // Drains the tile to the caller's output.  Deliberately a separate step from
  // solve_columns: the output address is only known here, so the 64-bit pointer
  // stays out of the solve's live range (see note 3 in the header).
  //
  // A lane owns one full COLUMN of the tile (col = lane) because the
  // substitution is sequential down a column and independent across columns, so
  // how that column reaches memory is a property of the destination's major -
  // Major_ is that choice:
  //
  //   Major::MN - the M axis (the row, i.e. the substitution axis) is the
  //     contiguous one, so a lane's whole column is ONE run of Block Elements:
  //     pack it in registers and drain it as Block / 8 16-byte stores.
  //   Major::K - the K axis (the contraction, i.e. the solve's column index) is
  //     the contiguous one, which is what Workspace::MatrixLayout declares.  A
  //     lane's values are then strided by Block, so the drain is scalar Element
  //     stores; the transpose a vector store would need is not worth its
  //     shuffle or shared-memory round trip at 16x16.
  //
  // The whole Block x Block tile is written, including the zero triangle above
  // the diagonal, because the consumer reads all Block rows.
  template <mute::TCE::Major Major_>
  MUTLASS_DEVICE static void publish_solution(SharedStorage const& shared_storage, Element* inv_output, int local_tid) {
    if (local_tid < Block) {
      int          col  = local_tid;
      float const* tile = shared_storage.solution + col * SolutionStride;
      using SolutionVec = mutlass::Array<float, VecSize>;
      using ElementVec  = mutlass::Array<Element, VecSize>;
      mutlass::NumericArrayConverter<Element, float, VecSize, mutlass::FloatRoundStyle::round_to_nearest>
          convert_solution;

      if constexpr (Major_ == mute::TCE::Major::MN) {
        constexpr int kDrainVec = 2 * VecSize;  // 16 bytes of Element
        static_assert(Block % kDrainVec == 0, "an M-major drain needs whole 16-byte runs");
        alignas(16) Element packed[Block];
        MUTE_UNROLL
        for (int row = 0; row < Block; row += VecSize) {
          ElementVec values = convert_solution(*reinterpret_cast<SolutionVec const*>(tile + row));
          MUTE_UNROLL
          for (int i = 0; i < VecSize; ++i) {
            packed[row + i] = values[i];
          }
        }
        using DrainVec = simd::PackedArray<Element, kDrainVec>;
        MUTE_UNROLL
        for (int row = 0; row < Block; row += kDrainVec) {
          *reinterpret_cast<DrainVec*>(inv_output + col * Block + row) =
              *reinterpret_cast<DrainVec const*>(packed + row);
        }
      } else {
        MUTE_UNROLL
        for (int row = 0; row < Block; row += VecSize) {
          SolutionVec values                  = *reinterpret_cast<SolutionVec const*>(tile + row);
          ElementVec  packed                  = convert_solution(values);
          inv_output[(row + 0) * Block + col] = packed[0];
          inv_output[(row + 1) * Block + col] = packed[1];
          inv_output[(row + 2) * Block + col] = packed[2];
          inv_output[(row + 3) * Block + col] = packed[3];
        }
      }
    }
  }
};

}  // namespace mate::flat::collective
