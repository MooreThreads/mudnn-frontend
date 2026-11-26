#pragma once

// MP31 chunk-KDA backward, K4: `chunk_kda_backward_grad`.
//
// The chunk-parallel half of the backward.  One CTA owns one (chunk, head)
// pair, so no cross-CTA reduction is needed (the whole 128-wide value axis
// lives in one CTA), and the only state that crosses a chunk boundary --
// dS -- is *loaded* from the `dstate` pool that K3 publishes, never carried.
//
// The elementwise maths is a one-for-one copy of the validated fp32-FMA
// reference (the upstream kernel); only the
// contractions changed. Every contraction with `Tile_N == 128` is evaluated
// on MP31 SQMMA as two 64-wide tiles. The two activation-typed 16x16 products
// (`dp` and `KKT`) occupy diagonal quadrants of one 32x32 SQMMA. The `da`
// product and the two `dM` steps retain their FP32 scalar contractions.
//
// Operand staging.  A built collective's `SmemLayoutA/B` is 128 B-swizzled,
// so an operand cannot alias a plain row-major buffer.  Each SQMMA operand is
// therefore staged into a single reused tile *through the layout accessor*,
// which also lets a `gemm_nn`/`gemm_tn` transpose be folded into the fill
// instead of needing a second collective with a transposed smem layout.
//
//   CVK (16,64,128)  A: (16,128) K-major      B: (64,128) K-major or MN-major
//   CV  (16,64,16)   A: (16,16)  K-major      B: (64,16)  MN-major
//
// Each SQMMA uses one 128-thread warp squad. All 512 CTA threads cooperate
// on staging and elementwise work.

#include <musa_runtime.h>
#include <mutlass/array.h>
#include <mutlass/half.h>
#include <mutlass/mutlass.h>
#include <mutlass/numeric_conversion.h>

#include <cstdint>
#include <mute/tensor.hpp>
#include <mutlass/gemm/collective/collective_builder.hpp>

#include "mate/common/layout.hpp"
#include "mate/common/mma_mp31_sqmma.hpp"
#include "mate/flat/kda/chunk_kda_problem.hpp"
#include "mate/flat/kda/chunk_kda_workspace.hpp"
#include "mate/flat/kda/kda_permutes.hpp"

// See K1: the correctly-rounded reciprocal square root is published in
// BuiltinsMTGPU.def but not declared in the public math headers.
extern "C" __device__ float __musa_rsqrt_rn_f(float);

namespace mate::flat::kda {
namespace backward_grad {

using namespace mute;

// Every rendezvous below is a full-CTA sync: all kThreads threads arrive, so
// it is `barrier.sync 0`.  These were `__barrier_sync_count(id, kThreads)`
// (frost's `barrier_cta_sync(id, count)` pattern), but mcc 5.2 never finishes
// compiling that intrinsic here -- a single K4 grad translation unit spins in
// the backend for over half an hour, which is what timed the CI smoke job out.
// `__syncthreads()` is the same instruction for a full-CTA rendezvous and, like
// the upstream K4, compiles in seconds.  Only reach for an explicit
// participant count again once the intrinsic compiles, and only when a role
// split actually narrows the count below kThreads.

inline constexpr int kThreads = 512;
// The SQMMA squad is threads [0, kSquadThreads): every TiledMma in this file is
// 128 threads wide. The other three squads stage operands and run
// the elementwise work, which is what the kernel is actually bound by.
inline constexpr int kSquadThreads = 128;

// ---------------------------------------------------------------------------
// fp32 contraction helpers for the shapes SQMMA cannot reach: every M x N of
// 16 x 16, plus the K=128 dot products behind them.  The accumulation order is
// copied verbatim from the validated reference, so the scalar path is
// bit-identical to it.
//   nt: out[m][n] += sum_k A[m][k] * B[n][k]
//   tn: out[m][n] += sum_k A[k][m] * B[k][n]
//
// WIDE LOADS ARE LOAD-BEARING.  One output per thread means a warp's lanes span
// n = 0..15 for two values of m, and both operand rows are K-contiguous with a
// 256 B (K=128) row pitch -- i.e. every lane's B address for a given k lands in
// ONE shared-memory bank.  The scalar loop is therefore a 16-way bank conflict
// per k: measured in isolation at 2 CTA/MP it costs 68,417 cycles for one
// (16,16,128) gemm, and it was 25% of the whole K4 kernel.  Moving four k per
// 8 B load keeps the same FMAs in the same order but cuts the conflicted
// accesses 4x: 21,144 cycles.  (Padding the B row to 136 elements is better
// still -- 13,683 -- but it would grow this kernel's shared memory, which is
// exactly what the two-CTA residency depends on.)
// ---------------------------------------------------------------------------
template <int K, class TA, class TB>
MUTLASS_DEVICE void dot4(TA const* a, TB const* b, float& acc) {
  static_assert(K % 4 == 0, "the vectorised dot needs K to be a multiple of 4");
  using VA = mutlass::Array<TA, 4>;
  using VB = mutlass::Array<TB, 4>;
  MUTE_UNROLL
  for (int k = 0; k < K; k += 4) {
    VA const av = *reinterpret_cast<VA const*>(a + k);
    VB const bv = *reinterpret_cast<VB const*>(b + k);
    MUTE_UNROLL
    for (int t = 0; t < 4; ++t) {
      acc = fmaf(static_cast<float>(av[t]), static_cast<float>(bv[t]), acc);
    }
  }
}

// `out` is the live accumulator on entry, exactly as the original loop had it.
template <int M, int N, int K, int BStride = K, class TA, class TB>
MUTLASS_DEVICE void gemm_nt(TA const* A, TB const* B, float* out) {
  // CEIL, not floor: a block larger than the M*N tile must idle extra threads,
  // never drop the tail of the tile.
  constexpr int TOTAL = M * N;
  constexpr int PER   = (TOTAL + kThreads - 1) / kThreads;
  int const     tid   = threadIdx.x;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    int const idx = tid * PER + j;
    if (idx >= TOTAL) break;
    int const m   = idx / N;
    int const n   = idx % N;
    float     acc = out[idx];
    dot4<K>(A + m * K, B + n * BStride, acc);
    out[idx] = acc;
  }
}

template <int M, int N, int K, class TA, class TB>
MUTLASS_DEVICE void gemm_tn(TA const* A, TB const* B, float* out) {
  // CEIL, not floor: a block larger than the M*N tile must idle extra threads,
  // never drop the tail of the tile.
  constexpr int TOTAL = M * N;
  constexpr int PER   = (TOTAL + kThreads - 1) / kThreads;
  int const     tid   = threadIdx.x;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    int const idx = tid * PER + j;
    if (idx >= TOTAL) break;
    int const m   = idx / N;
    int const n   = idx % N;
    float     acc = out[idx];
#pragma unroll 8
    for (int k = 0; k < K; ++k) {
      acc = fmaf(static_cast<float>(A[k * M + m]), static_cast<float>(B[k * N + n]), acc);
    }
    out[idx] = acc;
  }
}

// ---------------------------------------------------------------------------
// shared memory plan (C = 16, D = 128, Vt = 128)
//
//   staging (SQMMA operand tiles, reused behind barriers)      20.0 KiB
//   plain operands  ki dO r u dt(folded into the CVK A tile)   20.0 KiB
//   p / inverse / 16x16 chain (static)                          5.0 KiB
//   fp32 accumulators dU dkd dqd dki dkr dG                    48.0 KiB
//   scalars                                                     1.8 KiB
//                                                    total    94.9 KiB
//
// One CTA is 512 threads on one chunk x head pair.  The gmem-sourced operands
// (kd, qd, kr, S_in, dS) and the recomputed E are the residue of the residency
// work in docs/kda_backward.md: at 95 KiB total (static + dynamic) the CTA is
// under 98,304 B, which is what a second resident CTA per multiprocessor needs.
// The reference kept dS in fp32 because it carried and updated it across the
// chunk loop; K4 only reads it, so it stays in the pool's bf16.
// ---------------------------------------------------------------------------
// The six operands of the 16x16-chain contractions live in STATIC shared
// memory rather than in the dynamic storage below.  Measured on an MP31 device:
// the same 16x16x128 gemm_nt costs 10.5 us with the operand pointers provably
// in the shared space and 64.1 us when they are derived from the dynamic smem
// base (a 6x / ~875-cycle-per-k-step penalty), and a single dynamic operand is
// enough to halve the win (34 us).  Moving these out of the dynamic block and
// shrinking the dynamic request by the same 28 KiB keeps the CTA inside the
// 192 KiB limit with 12 KiB to spare.
template <class Element_, int C, int D, int Vt>
struct alignas(1024) BackwardGradOperands {
  using Element = Element_;
  // `kd` used to be cached here too; it is a read-only gmem image (the first
  // half of the decayed workspace record) consumed at four sites, so it is read
  // from gmem instead.  Freeing its 1 KiB slot is what lets the dynamic request
  // land under the two-CTA threshold.
  //
  // `dO`'s row-major cache is gone the same way -- but its slot is spent on a
  // second CVK A operand tile rather than returned, which is what lets phases 8
  // and 9a share one `S_in` staging round (they need `dt` and `dO` resident at
  // the same time).  Same 4 KiB, so the total is unchanged and `dO` is instead
  // read from gmem through the guarded fills.
  alignas(1024) Element stage_a2[C * D];  // second (16,128) K-major A tile
  // Eight padding elements per row keep the 16x16x128 FP32 contractions out
  // of the shared-memory bank-conflict pattern without changing the logical
  // tile shape.
  alignas(1024) Element r[C * (Vt + 8)];
  alignas(1024) Element u[C * (Vt + 8)];
  alignas(1024) Element ki[C * D];
  alignas(1024) float dU[C * Vt];
  // the 16x16 chain's own matrices and scalars: read/written by the solve, the
  // two small gemms and the dGtot reductions, all of which are serial loops.
  alignas(1024) float KKT[C * C];
  alignas(1024) float inv[C * C];
  alignas(1024) float X[C * C];
  alignas(1024) float dL[C * C];
  alignas(1024) float dp[C * C];
  alignas(16) Element ainv[C * C];
  alignas(16) Element p[C * C];
  alignas(16) float bs[C];
  alignas(16) float dbs[C];
  alignas(16) float Etot[D];
  alignas(16) float dE15[D];
  alignas(16) float dGtot[D];
};

// ---------------------------------------------------------------------------
template <class Element_, int C, int D, int Vt>
struct alignas(1024) BackwardGradSharedStorage {
  using Element = Element_;

  // SQMMA staging.  Each tile is 1024 B aligned so the 128 B swizzle atom is
  // happy; the aliasing layouts are chosen by the collective.
  alignas(1024) Element stage_a_cvk[C * D];   // (16,128) K-major
  alignas(1024) Element stage_b_cvk[64 * D];  // (64,128) K-major / MN-major aliased

  // The workspace operands `kd`, `qd`, `kr` and the two pool rows `S_in`, `dS`
  // are NOT cached here any more: every one of them is a read-only gmem image
  // that is consumed a bounded number of times, so the staging helpers read it
  // straight out of gmem.  `E` is likewise gone -- it was exactly
  // `Element(exp2_ftz(G))`, and phase 15 recomputes that from the G that stays
  // in `dG`, bit for bit, which is cheaper than 4 KiB of storage.  Together
  // with `dt` (folded into the CVK A tile) this is what puts the CTA under the
  // 98,304 B that two resident CTAs per multiprocessor need.

  alignas(16) float dkd[C * D];
  alignas(16) float dqd[C * D];
  alignas(16) float dki[C * D];
  alignas(16) float dkr[C * D];
  alignas(16) float dG[C * D];  // starts as the gate increment, becomes dG

  alignas(16) float rq[C];
  alignas(16) float rk[C];
};

// ---------------------------------------------------------------------------
template <class Element, class CuSeqlensElement, int C, int D, int Vt,
          class ElementGate = Element, class ElementBeta = Element>
struct ChunkKdaBackwardGradParams {
  Element const*          ptr_q;
  Element const*          ptr_k;
  Element const*          ptr_v;
  ElementGate const *ptr_g;
  ElementBeta const *ptr_beta;
  float const*            ptr_A_log;
  float const*            ptr_dt_bias;
  float                   scale;
  float                   lower_bound;
  Element const*          ptr_dO;
  void const*             ptr_chunk_states;
  void const*             ptr_dstate;
  Element*                ptr_dq;
  Element*                ptr_dk;
  Element*                ptr_dv;
  float*                  ptr_dg;
  float*                  ptr_dbeta;
  Element const*          ptr_ws_decayed;
  Element const*          ptr_ws_k_restored;
  Element const*          ptr_ws_p;
  Element const*          ptr_ws_inverse;
  Element const*          ptr_ws_total;
  int64_t                 stride_q1, stride_q2, stride_q3;
  int64_t                 stride_v1, stride_v2, stride_v3;
  int                     nseq;
  int                     T;
  int                     H;
  int                     workspace_chunks;
  CuSeqlensElement const* cu_seqlens;
};

// ---------------------------------------------------------------------------
template <class Element_, class CuSeqlensElement_, bool IsVarlen_,
          bool NormalizeQK_, int kChunk_, int kHeadDim_, int kValueDim_,
          class ElementGate_ = Element_, class ElementBeta_ = Element_,
          bool UseBetaSigmoid_ = true, bool AllowNegEigval_ = false,
          bool SafeGate_ = true, bool GateIsLog_ = false,
          bool GateIsLinear_ = false>
struct Mp31ChunkKdaBackwardGradCollectiveTmeWarpSpecialized {
  static constexpr bool UseBetaSigmoid = UseBetaSigmoid_;
  static constexpr bool AllowNegEigval = AllowNegEigval_;
  static constexpr bool SafeGate = SafeGate_;
  static constexpr bool GateIsLog = GateIsLog_;
  static constexpr bool GateIsLinear = GateIsLinear_;
  MUTLASS_DEVICE static float beta_from_raw(float x) {
    if constexpr (UseBetaSigmoid)
      return (AllowNegEigval ? 2.f : 1.f) *
             (1.f / (1.f + __musa_exp2_f(-x * kLog2E)));
    else
      return x;
  }
  MUTLASS_DEVICE static float beta_derivative(float b) {
    if constexpr (UseBetaSigmoid)
      return b * (1.f - (AllowNegEigval ? .5f : 1.f) * b);
    else
      return 1.f;
  }

  using Element                     = Element_;
  using ElementGate = ElementGate_;
  using ElementBeta = ElementBeta_;
  using CuSeqlensElement            = CuSeqlensElement_;
  static constexpr bool IsVarlen    = IsVarlen_;
  static constexpr bool NormalizeQK = NormalizeQK_;
  static constexpr int  C           = kChunk_;
  static constexpr int  D           = kHeadDim_;
  static constexpr int  Vt          = kValueDim_;
  static constexpr int  NT          = kThreads;
  // Aliases the FFI template uses to validate its arguments.
  static constexpr int   kChunk    = C;
  static constexpr int   kHeadDim  = D;
  static constexpr int   kValueDim = Vt;
  static constexpr float kLog2E    = 1.4426950408889634f;
  static constexpr float kLn2      = 0.6931471805599453f;
  static constexpr float kL2Eps    = 1.0e-6f;
  static constexpr float kRcpEps   = 1.0e-38f;
  static_assert(Vt == D, "the MP31 K4 grad kernel consumes the whole value axis");
  static_assert(C == 16 && D == 128, "MP31 K4 is built for C=16, D=128");

  using Params = ChunkKdaBackwardGradParams<Element, CuSeqlensElement, C, D, Vt,
                                            ElementGate, ElementBeta>;
  using Storage        = BackwardGradSharedStorage<Element, C, D, Vt>;
  using Operands       = BackwardGradOperands<Element, C, D, Vt>;
  using Workspace      = ChunkKdaWorkspace<Element, C, D>;
  using HeadDimPermute = PermuteN<mutlass::arch::Mp31, D>;
  using LogicalToSlot  = typename HeadDimPermute::LogicalToSlot;

  // ---- the three SQMMA shapes, built exactly the way K2 builds its own ----
  using StrideQK                         = ChunkKdaQkStride;
  using StrideVO                         = ChunkKdaVoStride;
  static constexpr int kOperandAlignment = 8;
  using DummyStageCount                  = mutlass::gemm::collective::StageCount<2>;

  template <class ShapeMNK, class StrideA, class StrideB>
  using MkCollective =
      typename mutlass::gemm::collective::CollectiveBuilder<mutlass::arch::Mp31,
                                                            mutlass::arch::OpClassTensorOp,
                                                            Element,
                                                            StrideA,
                                                            kOperandAlignment,
                                                            Element,
                                                            StrideB,
                                                            kOperandAlignment,
                                                            float,
                                                            ShapeMNK,
                                                            Shape<_1, _1, _1>,
                                                            DummyStageCount,
                                                            mutlass::gemm::KernelTmeWarpSpecialized>::CollectiveOp;

  // CVK: (16, 64, 128) -- the large-K contractions.
  //   K-major B  (StrideQK): B[n][k] is K-contiguous -> S_in / dS as stored
  //   MN-major B (StrideVO): B[n][k] is N-contiguous -> S_in / dS transposed
  using CollectiveCVK   = MkCollective<Shape<Int<C>, Int<64>, Int<D>>, StrideQK, StrideQK>;
  using CollectiveCVKVO = MkCollective<Shape<Int<C>, Int<64>, Int<D>>, StrideQK, StrideVO>;
  // CV: (16, 64, 16) -- the WY-chain contractions.
  using CollectiveCV = MkCollective<Shape<Int<C>, Int<64>, Int<C>>, StrideQK, StrideVO>;

  // Pack dp and KKT into diagonal quadrants of one 32x32 contraction.
  using CollectivePacked = MkCollective<Shape<Int<32>, Int<32>, Int<D>>, StrideQK, StrideQK>;
  using TiledMmaPacked   = typename CollectivePacked::TiledMma;
  using TiledMmaCVK      = typename CollectiveCVK::TiledMma;
  using TiledMmaCVKVO    = typename CollectiveCVKVO::TiledMma;
  using TiledMmaCV       = typename CollectiveCV::TiledMma;
  static_assert(decltype(size(TiledMmaCVK{}))::value == 128);
  static_assert(decltype(size(TiledMmaCVKVO{}))::value == 128);
  static_assert(decltype(size(TiledMmaCV{}))::value == 128);

  template <class Layout>
  using Unstage1 = decltype(take<0, 2>(mate::unstage_smem_layout(Layout{}, Int<1>{})));

  using SmemLayoutCVK_A  = Unstage1<typename CollectiveCVK::SmemLayoutA>;    // (16,128)
  using SmemLayoutCVK_BK = Unstage1<typename CollectiveCVK::SmemLayoutB>;    // (64,128) K-major
  using SmemLayoutCVK_BV = Unstage1<typename CollectiveCVKVO::SmemLayoutB>;  // (64,128) MN-major
  using SmemLayoutCV_A   = Unstage1<typename CollectiveCV::SmemLayoutA>;     // (16,16)
  using SmemLayoutCV_BV  = Unstage1<typename CollectiveCV::SmemLayoutB>;     // (64,16) MN-major

  using SmemLayoutPackedA = Unstage1<typename CollectivePacked::SmemLayoutA>;
  using SmemLayoutPackedB = Unstage1<typename CollectivePacked::SmemLayoutB>;
  static_assert(cosize_v<SmemLayoutPackedA> == 2 * C * D);
  static_assert(cosize_v<SmemLayoutPackedB> == 2 * C * D);

  // Every SQMMA smem operand must sit on its swizzle atom's boundary, and the
  // dynamic shared-memory base is NOT strongly aligned (measured on an MP31 device:
  // `cast_smem_ptr_to_uint(smem_raw) == 0xc`), so the storage is aligned at
  // runtime and this budget carries the alignment slack.
  static constexpr int kSmemAlignment     = 1024;
  static constexpr int SharedStorageSize  = sizeof(Storage) + kSmemAlignment;
  static constexpr int MaxThreadsPerBlock = NT;
  static constexpr int NumBarriers        = 0;
  static_assert(sizeof(Storage) % kSmemAlignment == 0, "the storage keeps its own 1024 B alignment");
  static_assert(SharedStorageSize <= 192 * 1024, "MP31 K4 shared memory exceeds the dynamic limit");
  static_assert(sizeof(Storage) + sizeof(Operands) + kSmemAlignment <= 192 * 1024,
                "static operands + dynamic storage exceed the per-CTA shared-memory limit");
  static_assert(cosize_v<SmemLayoutCVK_A> == C * D, "CVK A tile is 16x128");
  static_assert(cosize_v<SmemLayoutCVK_BK> == 64 * D, "CVK B tile is 64x128");
  static_assert(cosize_v<SmemLayoutCVK_BV> == 64 * D, "CVK B tile is 64x128");
  static_assert(cosize_v<SmemLayoutCV_A> == C * C, "CV A tile is 16x16");
  static_assert(cosize_v<SmemLayoutCV_BV> == 64 * C, "CV B tile is 64x16");

  struct WorkDesc {
    int64_t bos;
    int     seq_idx;
    int64_t seq_len;
    int     chunk_idx;
  };

  // Chunk-parallel work: one CTA per (record, head).  A `record` is the
  // chunk-major slot K1 wrote, which for varlen leaves gaps between
  // sequences, so a record that belongs to no sequence is invalid.
  MUTLASS_DEVICE static bool decode_work(Params const& p, int rec, WorkDesc& wd) {
    int const max_chunks = (p.T + C - 1) / C;
    if constexpr (!IsVarlen) {
      int const seq = rec / p.workspace_chunks;
      int const ci  = rec - seq * p.workspace_chunks;
      if (seq >= p.nseq || ci >= max_chunks) return false;
      wd = WorkDesc{int64_t(seq) * p.T, seq, p.T, ci};
      return true;
    } else {
      for (int sq = 0; sq < p.nseq; ++sq) {
        int64_t const b  = int64_t(p.cu_seqlens[sq]);
        int64_t const e  = int64_t(p.cu_seqlens[sq + 1]);
        int64_t const wb = (b + C - 1) / C + sq;
        if (int64_t(rec) < wb) break;  // `wb` is strictly increasing in `sq`
        int64_t const nc = (e - b + C - 1) / C;
        if (int64_t(rec) < wb + nc) {
          wd = WorkDesc{b, sq, e - b, int(int64_t(rec) - wb)};
          return true;
        }
      }
      return false;
    }
  }

  static dim3 get_grid_shape(Params const& p) {
    int const max_chunks = (p.T + C - 1) / C;
    int const records    = IsVarlen ? max_chunks + p.nseq : p.nseq * p.workspace_chunks;
    return dim3(unsigned(records) * unsigned(p.H), 1, 1);
  }

  // ---------------------------------------------------------------------
  // Vectorised staging.  Element is 2 bytes, so 8 elements are 16 B and the
  // SQMMA layouts keep each aligned 8-element group contiguous, making
  // a 16-byte chunk of the tile physically contiguous and safe to move with a
  // single uint4.  Scalar 2-byte staging is what the K3 bisect measured at
  // ~330 unhidden cycles per warp load (no coalescing, no overlap).
  // ---------------------------------------------------------------------
  static constexpr int kVec = 8;

  // Row-major source, K-major tile: `cols` is the fast axis of both.
  template <class Tensor>
  MUTLASS_DEVICE static void stage_row_major(
      Tensor& sT, Element const* src, int rows, int cols, int src_row_stride = -1) {
    int const     tid         = int(threadIdx.x);
    int const     total       = rows * cols;
    constexpr int kIterations = (decltype(size(Tensor{}))::value + NT * kVec - 1) / (NT * kVec);
    MUTE_UNROLL
    for (int iter = 0; iter < kIterations; ++iter) {
      int const i = tid * kVec + iter * NT * kVec;
      if (i >= total) continue;
      int const r = i / cols;
      int const c = i % cols;
      int const stride = (src_row_stride < 0) ? cols : src_row_stride;
      *reinterpret_cast<uint4*>(&sT(make_coord(r, c))) =
          *reinterpret_cast<uint4 const*>(src + r * stride + c);
    }
  }

  // MN-major tile, (k, n) row-major source: `n` is the fast axis of the tile.
  template <class Tensor>
  MUTLASS_DEVICE static void stage_mn_major(
      Tensor& sT, Element const* src, int n_total, int k_total, int src_row_stride, int src_col_base) {
    int const     tid         = int(threadIdx.x);
    int const     total       = n_total * k_total;
    constexpr int kIterations = (decltype(size(Tensor{}))::value + NT * kVec - 1) / (NT * kVec);
    MUTE_UNROLL
    for (int iter = 0; iter < kIterations; ++iter) {
      int const i = tid * kVec + iter * NT * kVec;
      if (i >= total) continue;
      int const k = i / n_total;
      int const n = i % n_total;
      *reinterpret_cast<uint4*>(&sT(make_coord(n, k))) =
          *reinterpret_cast<uint4 const*>(src + k * src_row_stride + src_col_base + n);
    }
  }

  // The two guarded fills below are what let `dO` stop being cached in shared
  // memory.  A partial chunk's tail rows have no gmem to read -- on a varlen
  // batch they would run off the end of the tensor -- and, more subtly, they
  // must come out ZERO: `dU`, `dkd`, `dqd`, `dki` and `dkr` are all required to
  // be zero above `actual` for phase 15's reverse cumsum over `dG`, which runs
  // over every row.  The old cached copy was zero-filled by its producer, so the
  // fill just moves here.  `kVec` consecutive lanes share a row (cols and
  // n_total are multiples of 4), so the guard is uniform across the vector.
  template <class Tensor>
  MUTLASS_DEVICE static void stage_row_major_guarded(
      Tensor& sT, Element const* src, int rows, int cols, int live_rows, int src_row_stride) {
    int const     tid         = int(threadIdx.x);
    int const     total       = rows * cols;
    constexpr int kIterations = (decltype(size(Tensor{}))::value + NT * kVec - 1) / (NT * kVec);
    MUTE_UNROLL
    for (int iter = 0; iter < kIterations; ++iter) {
      int const i = tid * kVec + iter * NT * kVec;
      if (i >= total) continue;
      int const r = i / cols;
      int const c = i % cols;
      if (r < live_rows) {
        *reinterpret_cast<uint4*>(&sT(make_coord(r, c))) =
            *reinterpret_cast<uint4 const*>(src + r * src_row_stride + c);
      } else {
#pragma unroll
        for (int j = 0; j < kVec; ++j) sT(make_coord(r, c + j)) = Element(0.0f);
      }
    }
  }

  template <class Tensor>
  MUTLASS_DEVICE static void stage_mn_major_guarded(
      Tensor& sT, Element const* src, int n_total, int k_total, int live_k, int src_row_stride, int src_col_base) {
    int const     tid         = int(threadIdx.x);
    int const     total       = n_total * k_total;
    constexpr int kIterations = (decltype(size(Tensor{}))::value + NT * kVec - 1) / (NT * kVec);
    MUTE_UNROLL
    for (int iter = 0; iter < kIterations; ++iter) {
      int const i = tid * kVec + iter * NT * kVec;
      if (i >= total) continue;
      int const k = i / n_total;
      int const n = i % n_total;
      if (k < live_k) {
        *reinterpret_cast<uint4*>(&sT(make_coord(n, k))) =
            *reinterpret_cast<uint4 const*>(src + k * src_row_stride + src_col_base + n);
      } else {
#pragma unroll
        for (int j = 0; j < kVec; ++j) sT(make_coord(n + j, k)) = Element(0.0f);
      }
    }
  }

  // `k_restored` is the one workspace field whose headDim axis is in the
  // kernel-private slot order (see kda_permutes.hpp); this is the same
  // un-permute K4 used to do once into a private `kr` buffer, now applied at
  // each of the three read sites. The SQMMA staging helper loads contiguous
  // physical slots, then scatters them to their logical head columns.
  MUTLASS_DEVICE static int kr_slot(int d) {
    return 8 * (d & 15) + (d >> 4);
  }

  // Row-major K-major tile from a slot-ordered source.
  template <class Tensor>
  MUTLASS_DEVICE static void stage_row_major_kr(Tensor& sT, Element const* src, int rows, int cols) {
    int const     tid         = int(threadIdx.x);
    constexpr int kKrVec      = 4;
    constexpr int kIterations = (decltype(size(Tensor{}))::value + NT * kKrVec - 1) / (NT * kKrVec);
    MUTE_UNROLL
    for (int iter = 0; iter < kIterations; ++iter) {
      int const i = tid * kKrVec + iter * NT * kKrVec;
      if (i >= rows * cols) continue;
      int const                             r      = i / cols;
      int const                             slot   = i % cols;
      mutlass::Array<Element, kKrVec> const values = *reinterpret_cast<mutlass::Array<Element, kKrVec> const*>(src + i);
      MUTE_UNROLL
      for (int j = 0; j < kKrVec; ++j) {
        int const d          = 16 * ((slot + j) & 7) + ((slot + j) >> 3);
        sT(make_coord(r, d)) = values[j];
      }
    }
  }

  // Lanes that cooperate on one token in the row-wise reductions.  Every
  // `C`-token row loop used to run on `C` threads with a serial 128-step
  // global-memory walk, which is both latency- and coalescing-bound; the same
  // work spread over a whole sub-warp makes the reads contiguous.
  static constexpr int kLanesPerToken = NT / C;
  static_assert(kLanesPerToken >= 1 && kLanesPerToken <= 32 && (kLanesPerToken & (kLanesPerToken - 1)) == 0,
                "kLanesPerToken must be a power of two no larger than a warp");

  // Butterfly sum over kLanesPerToken lanes.  MP31 emulates __shfl_xor_sync, so
  // this is an index shuffle with an absolute source lane, as in K1.
  MUTLASS_DEVICE static void subwarp_reduce_sum(float& value, int lane) {
    MUTE_UNROLL
    for (int offset = kLanesPerToken / 2; offset > 0; offset >>= 1) {
      value += __shfl_sync(uint32_t(-1), value, lane ^ offset, kLanesPerToken);
    }
  }

  MUTLASS_DEVICE static float exp2_ftz(float x) {
    float const y = __musa_exp2_f(x > 127.0f ? 127.0f : x);
    return x < -126.0f ? 0.0f : y;
  }

  // `2^G` for element (t, d), i.e. what the old materialised `E` buffer held.
  // `dG` carries the forward log2 cumsum until phase 15 overwrites it with the
  // gradient, and the last live row is the one place the workspace `total`
  // overrides the exponential.  Caller guarantees t < actual.
  MUTLASS_DEVICE static float decay_of(Storage const& s, int i, int t, int d, int actual, Element const* rec_total) {
    if (t == actual - 1) return static_cast<float>(rec_total[d]);
    return static_cast<float>(Element(exp2_ftz(s.dG[i])));
  }

  // ---- SQMMA plumbing ----------------------------------------------------

  // Issue one SQMMA over the whole K of the staged tiles and wait for it.
  //
  // The K loop is explicit: the `(16,64,128)` atom is `16x64x64`, so that
  // tile's operands carry two K blocks and `gemm` must be called once per
  // block.  The `(16,64,16)` tile is a single block, so the same code is a
  // no-op loop there.  The commit/wait pair is what makes the operand tiles
  // reusable, so every call site below can refill the same buffers.
  //
  MUTLASS_DEVICE static void retire_gemm() {
    if (int(threadIdx.x) >= kSquadThreads) return;
    mate::warpsquad_commit_batch();
    mate::warpsquad_wait<0>();
  }

  template <class TiledMma, class TensorA, class TensorB, class Acc>
  MUTLASS_DEVICE static void run_gemm_zero(TiledMma& mma, Acc& acc, TensorA const& sA, TensorB const& sB) {
    if (int(threadIdx.x) >= kSquadThreads) return;
    auto thr  = mma.get_thread_slice(int(threadIdx.x));
    auto tCrA = thr.make_fragment_A(thr.partition_A(sA));
    auto tCrB = thr.make_fragment_B(thr.partition_B(sB));
    mma.accumulate_ = MP31::SQMMA::ScaleOut::Zero;
    MUTE_UNROLL
    for (int kb = 0; kb < size<2>(tCrA); ++kb) {
      gemm(mma, tCrA(_, _, kb), tCrB(_, _, kb), acc);
      mma.accumulate_ = MP31::SQMMA::ScaleOut::One;
    }
    retire_gemm();
  }

  template <class TiledMma, class TensorA, class TensorB, class Acc>
  MUTLASS_DEVICE static void run_gemm(TiledMma const& mma, Acc& acc, TensorA const& sA, TensorB const& sB) {
    if (int(threadIdx.x) >= kSquadThreads) return;
    auto thr  = mma.get_thread_slice(int(threadIdx.x));
    auto tCrA = thr.make_fragment_A(thr.partition_A(sA));
    auto tCrB = thr.make_fragment_B(thr.partition_B(sB));
    MUTE_UNROLL
    for (int kb = 0; kb < size<2>(tCrA); ++kb) {
      gemm(mma, tCrA(_, _, kb), tCrB(_, _, kb), acc);
    }
    retire_gemm();
  }

  template <class TiledMma, class TensorA0, class TensorA1, class TensorB, class Acc0, class Acc1>
  MUTLASS_DEVICE static void run_gemm_pair_zero(
      TiledMma& mma, Acc0& acc0, TensorA0 const& sA0, Acc1& acc1, TensorA1 const& sA1, TensorB const& sB) {
    if (int(threadIdx.x) >= kSquadThreads) return;
    auto thr = mma.get_thread_slice(int(threadIdx.x));
    auto tCrB = thr.make_fragment_B(thr.partition_B(sB));
    auto tCrA0 = thr.make_fragment_A(thr.partition_A(sA0));
    mma.accumulate_ = MP31::SQMMA::ScaleOut::Zero;
    MUTE_UNROLL
    for (int kb = 0; kb < size<2>(tCrA0); ++kb) {
      gemm(mma, tCrA0(_, _, kb), tCrB(_, _, kb), acc0);
      mma.accumulate_ = MP31::SQMMA::ScaleOut::One;
    }
    auto tCrA1 = thr.make_fragment_A(thr.partition_A(sA1));
    mma.accumulate_ = MP31::SQMMA::ScaleOut::Zero;
    MUTE_UNROLL
    for (int kb = 0; kb < size<2>(tCrA1); ++kb) {
      gemm(mma, tCrA1(_, _, kb), tCrB(_, _, kb), acc1);
      mma.accumulate_ = MP31::SQMMA::ScaleOut::One;
    }
    retire_gemm();
  }

  template <class Acc, class Coords>
  MUTLASS_DEVICE static void acc_into_smem(
      Acc const& acc, Coords const& coords, float* dst, int row_stride, int n_base) {
    if (int(threadIdx.x) >= kSquadThreads) return;
    MUTE_UNROLL
    for (int i = 0; i < size(acc); ++i) {
      int const m = int(get<0>(coords(i)));
      int const n = int(get<1>(coords(i)));
      dst[m * row_stride + n_base + n] += acc(i);
    }
  }

  template <class Acc, class Coords>
  MUTLASS_DEVICE static void acc_store_bf16(
      Acc const& acc, Coords const& coords, Element* dst, int row_stride, int n_base) {
    if (int(threadIdx.x) >= kSquadThreads) return;
    MUTE_UNROLL
    for (int i = 0; i < size(acc); ++i) {
      int const m                      = int(get<0>(coords(i)));
      int const n                      = int(get<1>(coords(i)));
      dst[m * row_stride + n_base + n] = Element(acc(i));
    }
  }

  // -------------------------------------------------------------------------
  MUTLASS_DEVICE void operator()(Params const& p, char* smem_raw) const {
    uintptr_t const aligned =
        (reinterpret_cast<uintptr_t>(smem_raw) + uintptr_t(kSmemAlignment - 1)) & ~uintptr_t(kSmemAlignment - 1);
    Storage& s = *reinterpret_cast<Storage*>(aligned);
    // Static-space operands: see BackwardGradOperands.  `__shared__` inside a
    // device function is a fixed static allocation, which is exactly what the
    // gemm inner loops need to keep their loads on the banked shared path.
    __shared__ Operands so;
    int const           tid = int(threadIdx.x);

    // ---- 0. work decode -------------------------------------------------
    // `rec` is the chunk-major (record, head) flat index shared by the K1
    // workspace ([records, H, RecordStride]) and the two bf16 pools
    // ([rows, H, V, D]) -- i.e. exactly `blockIdx.x`.  `row` is the record
    // index alone (what the dense/varlen scheduler decodes).
    int const rec  = int(blockIdx.x);
    int const head = rec % p.H;
    int const row  = rec / p.H;
    WorkDesc  wd;
    if (!decode_work(p, row, wd)) return;
    int const     seq         = wd.seq_idx;
    int const     ci          = wd.chunk_idx;
    int64_t const bos         = wd.bos;
    int64_t const seq_len     = wd.seq_len;
    int64_t const chunk_start = int64_t(ci) * C;
    int const     actual      = int(seq_len - chunk_start) < C ? int(seq_len - chunk_start) : C;
    int64_t const tok0        = bos + chunk_start;

    float const alpha = (SafeGate ? __musa_exp2_f(p.ptr_A_log[head] * kLog2E)
                                  : 1.f); // exp(A_log)
    float const *__restrict__ dtb =
        SafeGate ? p.ptr_dt_bias + int64_t(head) * D : nullptr;

    int64_t const tq = p.stride_q1, hq = p.stride_q2, bq = p.stride_q3;
    int64_t const tv = p.stride_v1, hv = p.stride_v2, bv = p.stride_v3;
    // The batch index is folded into `bos` for varlen and into an explicit
    // *offset* (never the stride) for the dense layout.
    int64_t const seq_q = IsVarlen ? bos * tq : int64_t(seq) * bq;
    int64_t const seq_v = IsVarlen ? bos * tv : int64_t(seq) * bv;

    // ---- 1. workspace operands -----------------------------------------
    Element const* rec_decayed = p.ptr_ws_decayed + int64_t(rec) * Workspace::RecordStride;
    Element const* rec_kr      = p.ptr_ws_k_restored + int64_t(rec) * Workspace::RecordStride;
    Element const* rec_p       = p.ptr_ws_p + int64_t(rec) * Workspace::RecordStride;
    Element const* rec_inv     = p.ptr_ws_inverse + int64_t(rec) * Workspace::RecordStride;
    Element const* rec_total   = p.ptr_ws_total + int64_t(rec) * Workspace::RecordStride;

    // The read-only operands live in gmem and are consumed straight from
    // there, so this phase only keeps the two things that are NOT a gmem image:
    // the WY chain's `p`/`ainv` (bf16 workspace fields), staged once because
    // the 16x16 chain touches them in four places.
    //
    //   kd  = rec_decayed[0 .. C*D)     row-major, row stride D
    //   qd  = rec_decayed[C*D .. 2*C*D) row-major, row stride D
    //   kr  = rec_kr, headDim axis in slot order
    //   S_in, dS = the two pool rows, value-major
    Element const* g_kd  = rec_decayed;
    Element const* g_qd  = rec_decayed + C * D;
    Element const* g_kr  = rec_kr;
    Element const* g_sIn = reinterpret_cast<Element const*>(p.ptr_chunk_states) + int64_t(rec) * Vt * D;
    Element const* g_ds  = reinterpret_cast<Element const*>(p.ptr_dstate) + int64_t(rec) * Vt * D;
    // dO is row-major (C, Vt) in gmem with the token stride `tv`, and only its
    // first `actual` rows exist; see stage_*_guarded.
    Element const* g_dO = p.ptr_dO + seq_v + head * hv + chunk_start * tv;

    for (int i = tid; i < C * C; i += NT) {
      so.p[i]    = rec_p[i];
      so.ainv[i] = rec_inv[i];
    }

    // ---- 2. gate increment -> chunk-local log2 cumsum -------------------
    // The cumsum stays in dG and is *kept* there: `2^G` is recomputed from it
    // where it is needed instead of being materialised into its own buffer.
    for (int i = tid; i < C * D; i += NT) {
      int const t    = i / D;
      int const d    = i - t * D;
      float     gate = 0.0f;
      if (t < actual) {
        float const gv =
            static_cast<float>(
                p.ptr_g[seq_q + (chunk_start + t) * tq + head * hq + d]) +
            (SafeGate ? dtb[d] : 0.f);
        // sigmoid(x) = 1/(1+2^{-x*log2e}); the log2e factor is load-bearing.
        if constexpr (SafeGate)
          gate = p.lower_bound * kLog2E /
                 (1.f + __musa_exp2_f(-(alpha * gv) * kLog2E));
        else if constexpr (GateIsLinear)
          gate = ::log2f(fmaxf(gv,1e-10f));
        else
          gate = gv * (GateIsLog ? kLog2E : 1.f);
      }
      s.dG[i] = gate;
    }
    for (int t = tid; t < C; t += NT) {
      float bv = -80.0f;
      if (t < actual) bv = static_cast<float>(p.ptr_beta[(tok0 + t) * p.H + head]);
      so.bs[t] = t < actual ? beta_from_raw(bv) : 0.f;
    }
    // Fences the gate sweep above before the 128-lane cumsum reads s.dG.
    __syncthreads();

    if (tid < D) {
      float acc = 0.0f;
      for (int t = 0; t < C; ++t) {
        acc += s.dG[t * D + tid];
        s.dG[t * D + tid] = acc;
      }
    }
    // `E` is not materialised: the one thing special about the last live row is
    // that `rec_total` overrides `2^G` there, so both readers below apply that
    // override at the point of use.  This is the same expression, rounded the
    // same way (`Element(exp2_ftz(G))`), so the values are bit-identical.
    if (tid < D) so.Etot[tid] = static_cast<float>(rec_total[tid]);
    __syncthreads();

    // ---- 3. rebuild rq / rk / ki / dO / v -------------------------------
    if constexpr (NormalizeQK) {
      // 1/kLanesPerToken of the block per token; each lane walks D in strides
      // and the sub-warp reduces.
      int const t    = tid / kLanesPerToken;
      int const lane = tid % 32;
      float     sq = 0.0f, sk = 0.0f;
      if (t < C && t < actual) {
#pragma unroll 4
        for (int d = tid % kLanesPerToken; d < D; d += kLanesPerToken) {
          float const qv = static_cast<float>(p.ptr_q[seq_q + (chunk_start + t) * tq + head * hq + d]);
          float const kv = static_cast<float>(p.ptr_k[seq_q + (chunk_start + t) * tq + head * hq + d]);
          sq             = fmaf(qv, qv, sq);
          sk             = fmaf(kv, kv, sk);
        }
      }
      subwarp_reduce_sum(sq, lane);
      subwarp_reduce_sum(sk, lane);
      if (lane == 0 && t < C) {
        s.rq[t] = (t < actual) ? __musa_rsqrt_rn_f(sq + kL2Eps) : 1.0f;
        s.rk[t] = (t < actual) ? __musa_rsqrt_rn_f(sk + kL2Eps) : 1.0f;
      }
    }
    // C * D == C * Vt, so `ki`, `dO`/`r` and the `dU` zero fill share one
    // iteration space and run in one loop: three separate passes each walked
    // gmem with only four loads per thread in flight, which on a 2048-element
    // sweep is all latency.  Fused, each thread has eleven independent loads
    // outstanding instead of four.
    // Four consecutive head columns per lane: k/v/q loads become single 8 B
    // accesses and the three outputs single 8/8/16 B stores.  The per-element
    // expressions and their order are unchanged.
    {
      int const i0 = int(tid) * 4;
      int const t  = i0 / D;
      int const d  = i0 - t * D;
      float      kh[4];
      float      vv[4];
      MUTE_UNROLL
      for (int j = 0; j < 4; ++j) {
        kh[j] = 0.0f;
        vv[j] = 0.0f;
      }
      if (t < actual) {
        mutlass::Array<Element, 4> const k4 = *reinterpret_cast<mutlass::Array<Element, 4> const *>(
            p.ptr_k + seq_q + (chunk_start + t) * tq + head * hq + d);
        mutlass::Array<Element, 4> const v4 = *reinterpret_cast<mutlass::Array<Element, 4> const *>(
            p.ptr_v + seq_v + (chunk_start + t) * tv + head * hv + d);
        MUTE_UNROLL
        for (int j = 0; j < 4; ++j) {
          float kraw = static_cast<float>(k4[j]);
          if constexpr (NormalizeQK) kraw = static_cast<float>(Element(kraw * s.rk[t]));
          kh[j] = kraw;
          vv[j] = static_cast<float>(v4[j]);
        }
      }
      float e[4];
      MUTE_UNROLL
      for (int j = 0; j < 4; ++j) {
        e[j] = (t == actual - 1) ? static_cast<float>(rec_total[d + j])
                                 : static_cast<float>(Element(exp2_ftz(s.dG[i0 + j])));
      }
      mutlass::Array<Element, 4> ki4;
      mutlass::Array<Element, 4> r4;
      mutlass::Array<float, 4>   du4;
      MUTE_UNROLL
      for (int j = 0; j < 4; ++j) {
        ki4[j] = (t < actual) ? Element(kh[j] / (e[j] + kRcpEps)) : Element(0.0f);
        r4[j]  = (t < actual) ? Element(vv[j]) : Element(0.0f);
        du4[j] = 0.0f;
      }
      *reinterpret_cast<mutlass::Array<Element, 4> *>(&so.ki[i0])                = ki4;
      *reinterpret_cast<mutlass::Array<Element, 4> *>(&so.r[t * (Vt + 8) + d])  = r4;
      *reinterpret_cast<mutlass::Array<float, 4> *>(&so.dU[i0])                 = du4;
    }
    // dG holds the chunk-local forward cumsum G and STAYS that way: the gate
    // chain below wants only the gradient terms, and phase 15 assigns them
    // (rather than accumulating onto a zeroed buffer) so `2^G` is still
    // recoverable there.
    static_assert((C * D) % (NT * 4) == 0, "vectorized phase-3 clears require an exact float4 tile");
    for (int i = tid * 4; i < C * D; i += NT * 4) {
      *reinterpret_cast<float4*>(&s.dkd[i]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      *reinterpret_cast<float4*>(&s.dqd[i]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      *reinterpret_cast<float4*>(&s.dki[i]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      *reinterpret_cast<float4*>(&s.dkr[i]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    for (int i = tid; i < C * C; i += NT) {
      so.dp[i]  = 0.0f;
      so.X[i]   = 0.0f;
      so.dL[i]  = 0.0f;
      so.KKT[i] = 0.0f;
      so.inv[i] = 0.0f;
    }
    for (int i = tid; i < D; i += NT) {
      so.dE15[i]  = 0.0f;
      so.dGtot[i] = 0.0f;
    }
    // No barrier here: the zeroed tiles are first read phases later, and the
    // phase-4 staging barrier already fences them.

    // =====================================================================
    // SQMMA phase
    // =====================================================================
    TiledMmaCVK   mma_cvk;
    TiledMmaCVKVO mma_cvkvo;
    TiledMmaCV    mma_cv;
    auto          thr_cvk = mma_cvk.get_thread_slice(tid);
    auto          thr_cv  = mma_cv.get_thread_slice(tid);

    auto sA_cvk   = make_tensor(make_smem_ptr(s.stage_a_cvk), SmemLayoutCVK_A{});
    auto sB_cvk   = make_tensor(make_smem_ptr(s.stage_b_cvk), SmemLayoutCVK_BK{});
    auto sB_cvk_v = make_tensor(make_smem_ptr(s.stage_b_cvk), SmemLayoutCVK_BV{});
    auto sA_cv    = make_tensor(make_smem_ptr(s.stage_a_cvk), SmemLayoutCV_A{});
    // A second CVK A tile (static, where `dO`'s cache used to be) and a second
    // CV A tile (the tail of stage_a_cvk, which is 4 KiB but only needs 512 B
    // for a CV operand).  Together they let two phases share one B staging.
    auto sA2_cvk = make_tensor(make_smem_ptr(so.stage_a2), SmemLayoutCVK_A{});
    auto sA2_cv  = make_tensor(make_smem_ptr(s.stage_a_cvk + C * C), SmemLayoutCV_A{});
    auto sB_cv   = make_tensor(make_smem_ptr(s.stage_b_cvk), SmemLayoutCV_BV{});

    auto cC_cvk = thr_cvk.partition_C(make_identity_tensor(make_shape(Int<C>{}, Int<64>{})));
    auto cC_cv  = thr_cv.partition_C(make_identity_tensor(make_shape(Int<C>{}, Int<64>{})));

    // ---- 4. R = v - kd S_in^T -------------------------------------------
    // Seeded with -v so the SQMMA's `acc +=` leaves -R, exactly the
    // reference's trick.
    for (int half = 0; half < Vt / 64; ++half) {
      stage_row_major(sA_cvk, g_kd, C, D);
      // K-major B: B(n=v,k=d) = S_in[v][d], a row copy of the pool row.
      stage_row_major(sB_cvk, g_sIn + half * 64 * D, 64, D);
      __syncthreads();
      auto acc = partition_fragment_C(mma_cvk, Shape<Int<C>, Int<64>>{});
      if (tid < kSquadThreads) {
        MUTE_UNROLL
        for (int i = 0; i < size(acc); ++i) {
          int const m = int(get<0>(cC_cvk(i)));
          int const n = int(get<1>(cC_cvk(i)));
          acc(i)      = -static_cast<float>(so.r[m * (Vt + 8) + half * 64 + n]);
        }
      }
      run_gemm(mma_cvk, acc, sA_cvk, sB_cvk);
      if (tid < kSquadThreads) {
        MUTE_UNROLL
        for (int i = 0; i < size(acc); ++i) {
          int const m                  = int(get<0>(cC_cvk(i)));
          int const n                  = int(get<1>(cC_cvk(i)));
          so.r[m * (Vt + 8) + half * 64 + n] = Element(-acc(i));
        }
      }
      __syncthreads();
    }

    // ---- 5. U = a R ------------------------------------------------------
    for (int half = 0; half < Vt / 64; ++half) {
      for (int i = tid; i < C * C; i += NT) {
        sA_cv(make_coord(i / C, i % C)) = so.ainv[i];
      }
      // MN-major B: B(n=v,k=c) = R[c][v]; the fastest tile axis is n.
      stage_mn_major(sB_cv, so.r, 64, C, Vt + 8, half * 64);
      __syncthreads();
      auto acc = partition_fragment_C(mma_cv, Shape<Int<C>, Int<64>>{});
      run_gemm_zero(mma_cv, acc, sA_cv, sB_cv);
      acc_store_bf16(acc, cC_cv, so.u, Vt + 8, half * 64);
      __syncthreads();
    }

    // ---- 6. dU = p^T dO + kr dS^T ---------------------------------------
    for (int half = 0; half < Vt / 64; ++half) {
      // A = p^T: A(m=t,k=c) = p[c][t].
      for (int i = tid; i < C * C; i += NT) {
        int const m             = i / C;
        int const k             = i - m * C;
        sA_cv(make_coord(m, k)) = so.p[k * C + m];
      }
      stage_mn_major_guarded(sB_cv, g_dO, 64, C, actual, tv, half * 64);
      __syncthreads();
      auto acc = partition_fragment_C(mma_cv, Shape<Int<C>, Int<64>>{});
      run_gemm_zero(mma_cv, acc, sA_cv, sB_cv);
      acc_into_smem(acc, cC_cv, so.dU, Vt, half * 64);

      // += kr dS^T, K-major B over dS.
      stage_row_major_kr(sA_cvk, g_kr, C, D);
      stage_row_major(sB_cvk, g_ds + half * 64 * D, 64, D);
      __syncthreads();
      auto acc2 = partition_fragment_C(mma_cvk, Shape<Int<C>, Int<64>>{});
      run_gemm_zero(mma_cvk, acc2, sA_cvk, sB_cvk);
      acc_into_smem(acc2, cC_cvk, so.dU, Vt, half * 64);
      __syncthreads();
    }

    // ---- 7. dtmp = a^T dU ; dv = dtmp ; -dtmp becomes dkd's A operand ----
    // The two 64-wide halves of `-(a^T dU)` are written straight into the
    // swizzled CVK A tile, which is the operand phase 8 wants anyway, so the
    // private `dt` staging buffer is gone.  Both C fragments stay in registers
    // across the loop: the CV A/B tiles alias the CVK tiles, so the fill for
    // the second half would clobber the first half's `dt` if it were written
    // out per iteration.
    {
      auto accA = partition_fragment_C(mma_cv, Shape<Int<C>, Int<64>>{});
      auto accB = partition_fragment_C(mma_cv, Shape<Int<C>, Int<64>>{});
      for (int half = 0; half < Vt / 64; ++half) {
        for (int i = tid; i < C * C; i += NT) {
          int const m             = i / C;
          int const k             = i - m * C;
          sA_cv(make_coord(m, k)) = so.ainv[k * C + m];
        }
        for (int i = tid; i < 64 * C; i += NT) {
          int const n             = i % 64;
          int const k             = i / 64;
          sB_cv(make_coord(n, k)) = Element(so.dU[k * Vt + half * 64 + n]);
        }
        __syncthreads();
        if (half == 0) {
          run_gemm_zero(mma_cv, accA, sA_cv, sB_cv);
        } else {
          run_gemm_zero(mma_cv, accB, sA_cv, sB_cv);
        }
        __syncthreads();
      }
      if (tid < kSquadThreads) {
        MUTE_UNROLL
        for (int i = 0; i < size(accA); ++i) {
          int const m = int(get<0>(cC_cv(i)));
          int const n = int(get<1>(cC_cv(i)));
          if (m < actual) {
            p.ptr_dv[seq_v + (chunk_start + m) * tv + head * hv + n] = Element(accA(i));
          }
          sA_cvk(make_coord(m, n)) = Element(-accA(i));
        }
        MUTE_UNROLL
        for (int i = 0; i < size(accB); ++i) {
          int const m  = int(get<0>(cC_cv(i)));
          int const n  = int(get<1>(cC_cv(i)));
          int const vv = 64 + n;
          if (m < actual) {
            p.ptr_dv[seq_v + (chunk_start + m) * tv + head * hv + vv] = Element(accB(i));
          }
          sA_cvk(make_coord(m, vv)) = Element(-accB(i));
        }
      }
    }

    // ---- 8 + 9a. dkd = -dtmp S_in and dqd = dO S_in, one S_in staging -----
    // Both contractions take the SAME MN-major `S_in` tile, so staging it once
    // per half feeds both: two rounds per half instead of four, and one
    // `retire_gemm` instead of two.  `dt` stays in the CVK A tile while `dO` is
    // staged into the second one, which is why that tile exists.  Each
    // accumulator keeps its own accumulation order, so this stays bit-identical.
    stage_row_major_guarded(sA2_cvk, g_dO, C, D, actual, tv);
    for (int half = 0; half < Vt / 64; ++half) {
      stage_mn_major(sB_cvk_v, g_sIn, 64, D, D, half * 64);
      __syncthreads();
      auto acc_dkd = partition_fragment_C(mma_cvkvo, Shape<Int<C>, Int<64>>{});
      auto acc_dqd = partition_fragment_C(mma_cvkvo, Shape<Int<C>, Int<64>>{});
      run_gemm_pair_zero(mma_cvkvo, acc_dkd, sA_cvk, acc_dqd, sA2_cvk, sB_cvk_v);  // A = -dtmp, dO
      acc_into_smem(acc_dkd, cC_cvk, s.dkd, D, half * 64);
      acc_into_smem(acc_dqd, cC_cvk, s.dqd, D, half * 64);
      __syncthreads();
    }

    // ---- 9b. dkr = U dS (MN-major B) ------------------------------------
    stage_row_major(sA_cvk, so.u, C, D, Vt + 8);
    for (int half = 0; half < Vt / 64; ++half) {
      stage_mn_major(sB_cvk_v, g_ds, 64, D, D, half * 64);
      __syncthreads();
      auto acc = partition_fragment_C(mma_cvkvo, Shape<Int<C>, Int<64>>{});
      run_gemm_zero(mma_cvkvo, acc, sA_cvk, sB_cvk_v);
      acc_into_smem(acc, cC_cvk, s.dkr, D, half * 64);
      __syncthreads();
    }

    // ---- 10 + 11: the 16x16 chain and the two dGtot terms ---------------
    // These have no cross dependencies beyond the four points marked below, so
    // they share one barrier region instead of nine.  The dp/da/KKT chain does
    // not touch inv/dbs/dE15/dGtot; the solve reads only KKT and da; the dM
    // chain reads inv, da and ainv; the dGtot reductions read neither.  Each
    // accumulator zero/copy is done by the same lane that later accumulates
    // that element (PER == 1 below), so only the genuinely cross-lane points
    // are barriers.
    static_assert(NT >= C * C,
                  "merged 16x16 region assumes PER == 1: the lane that zeroes an "
                  "accumulator element must be the lane that accumulates it");

    // The 16 KiB B stage is dead after dkr. Its two 8 KiB halves now
    // hold A=[dO; kd] and B=[U; ki]. Only the two diagonal 16x16
    // quadrants are consumed. The FP32 dU contraction remains scalar.
    {
      TiledMmaPacked mma_packed;
      auto           sPackedA = make_tensor(make_smem_ptr(s.stage_b_cvk), SmemLayoutPackedA{});
      auto           sPackedB = make_tensor(make_smem_ptr(s.stage_b_cvk + 2 * C * D), SmemLayoutPackedB{});
      for (int i = tid * kVec; i < C * D; i += NT * kVec) {
        int const r = i / D;
        int const d = i % D;
        if (r < actual) {
          *reinterpret_cast<uint4*>(&sPackedA(make_coord(r, d))) = *reinterpret_cast<uint4 const*>(g_dO + r * tv + d);
        } else {
          *reinterpret_cast<uint4*>(&sPackedA(make_coord(r, d))) = make_uint4(0, 0, 0, 0);
        }
        *reinterpret_cast<uint4*>(&sPackedA(make_coord(C + r, d))) = *reinterpret_cast<uint4 const*>(g_kd + i);
        *reinterpret_cast<uint4*>(&sPackedB(make_coord(r, d))) =
            *reinterpret_cast<uint4 const*>(so.u + r * (Vt + 8) + d);
        *reinterpret_cast<uint4*>(&sPackedB(make_coord(C + r, d))) = *reinterpret_cast<uint4 const*>(so.ki + i);
      }
      __syncthreads();
      auto acc = partition_fragment_C(mma_packed, Shape<Int<32>, Int<32>>{});
      run_gemm_zero(mma_packed, acc, sPackedA, sPackedB);
      gemm_nt<C, C, Vt, Vt + 8>(so.dU, so.r, so.X);  // da = dU R^T, FP32 inputs
      if (tid < kSquadThreads) {
        auto coords =
            mma_packed.get_thread_slice(tid).partition_C(make_identity_tensor(make_shape(Int<32>{}, Int<32>{})));
        MUTE_UNROLL
        for (int i = 0; i < size(acc); ++i) {
          int const r = int(get<0>(coords(i)));
          int const c = int(get<1>(coords(i)));
          if (r < C && c < C) so.dp[r * C + c] = acc(i);
          if (r >= C && c >= C) so.KKT[(r - C) * C + c - C] = acc(i);
        }
      }
    }
    __syncthreads();  // [1] the solve reads KKT and da

    // The solve needs 16 lanes; the dGtot reductions are independent of it and
    // of inv, so they run on a different half of the block, in its shadow.
    // dE15[d] = sum_c dkr[c,d] kr[c,d]; dGtot[d] = (dS . S_in) Etot ln2 +
    // dE15[d] ln2, both read off the *incoming* dS and the stored operands.
    if (tid < C) {
      // inv = (I + tril(diag(bs) KKT, -1))^-1, one lane per column, and
      // dbs = diag(inv^T da) in the same pass.
      int const col = tid;
      for (int t = 0; t < C; ++t) so.inv[t * C + col] = (t == col) ? 1.0f : 0.0f;
      for (int t = col + 1; t < C; ++t) {
        float acc = 0.0f;
        for (int k = col; k < t; ++k) {
          float const Ltk = (k < t) ? so.KKT[t * C + k] * so.bs[t] : 0.0f;
          acc             = fmaf(Ltk, so.inv[k * C + col], acc);
        }
        so.inv[t * C + col] = -acc;
      }
      float acc = 0.0f;
      for (int t = 0; t < C; ++t) acc = fmaf(so.inv[t * C + tid], so.X[t * C + tid], acc);
      so.dbs[tid] = acc;
    } else if (tid >= D && tid < 2 * D) {
      // 128 lanes, one per column, and 368 lanes idle: this block is the single
      // worst latency hole in the kernel after the residency work, because the
      // two pool rows are gmem now.  `state_dot` is a serial 128-step FMA chain
      // over a 256 B stride, so leaving the loads inside the chain exposes the
      // full L2 latency 128 times.  Hoisting four rows of loads per step keeps
      // the *same* accumulation order (still bit-identical) while putting eight
      // independent loads in flight.
      int const d   = tid - D;
      float     acc = 0.0f;
#pragma unroll 4
      for (int c = 0; c < C; c += 4) {
        float const k0 = static_cast<float>(g_kr[(c + 0) * D + kr_slot(d)]);
        float const k1 = static_cast<float>(g_kr[(c + 1) * D + kr_slot(d)]);
        float const k2 = static_cast<float>(g_kr[(c + 2) * D + kr_slot(d)]);
        float const k3 = static_cast<float>(g_kr[(c + 3) * D + kr_slot(d)]);
        acc            = fmaf(s.dkr[(c + 0) * D + d], k0, acc);
        acc            = fmaf(s.dkr[(c + 1) * D + d], k1, acc);
        acc            = fmaf(s.dkr[(c + 2) * D + d], k2, acc);
        acc            = fmaf(s.dkr[(c + 3) * D + d], k3, acc);
      }
      so.dE15[d] = acc;
      // This dot stays here, in the shadow of the 16-lane solve (see the block
      // comment): folding it into the SQMMA loop above put it on that loop's
      // critical path and cost 30% of the kernel, measured.
      float state_dot = 0.0f;
#pragma unroll 4
      for (int vv = 0; vv < Vt; vv += 4) {
        float const d0 = static_cast<float>(g_ds[(vv + 0) * D + d]);
        float const d1 = static_cast<float>(g_ds[(vv + 1) * D + d]);
        float const d2 = static_cast<float>(g_ds[(vv + 2) * D + d]);
        float const d3 = static_cast<float>(g_ds[(vv + 3) * D + d]);
        float const i0 = static_cast<float>(g_sIn[(vv + 0) * D + d]);
        float const i1 = static_cast<float>(g_sIn[(vv + 1) * D + d]);
        float const i2 = static_cast<float>(g_sIn[(vv + 2) * D + d]);
        float const i3 = static_cast<float>(g_sIn[(vv + 3) * D + d]);
        state_dot      = fmaf(d0, i0, state_dot);
        state_dot      = fmaf(d1, i1, state_dot);
        state_dot      = fmaf(d2, i2, state_dot);
        state_dot      = fmaf(d3, i3, state_dot);
      }
      so.dGtot[d] = state_dot * so.Etot[d] * kLn2 + so.dE15[d] * kLn2;
    }
    // dM = -(inv^T da a^T); dL = tril(dM,-1); dLtmp = tril(dL * bs, -1)
    for (int i = tid; i < C * C; i += NT) so.dL[i] = 0.0f;
    __syncthreads();  // [2] inv visible, dL zeroed

    gemm_tn<C, C, C>(so.inv, so.X, so.dL);  // dL = inv^T da
    for (int i = tid; i < C * C; i += NT) so.X[i] = so.dL[i];
    __syncthreads();  // [3] da (in X) visible to the next gemm

    for (int i = tid; i < C * C; i += NT) so.dL[i] = 0.0f;
    gemm_nt<C, C, C>(so.X, so.ainv, so.dL);  // dL = X a^T   (a = workspace inverse)
    if constexpr (!UseBetaSigmoid) {
      __syncthreads();
      if (tid < C) {
        float acc = so.dbs[tid];
        for (int ss = 0; ss < tid; ++ss)
          acc = fmaf(-so.dL[tid * C + ss], so.KKT[tid * C + ss], acc);
        so.dbs[tid] = acc;
      }
      __syncthreads();
    }
    for (int i = tid; i < C * C; i += NT) {
      int const t  = i / C;
      int const ss = i - t * C;
      so.dL[i]     = (ss < t) ? -so.dL[i] * so.bs[t] : 0.0f;  // dLtmp
    }
    __syncthreads();  // [4] dLtmp visible to the dbs lanes

    if constexpr (UseBetaSigmoid)
      if (tid < C) {
        float acc = so.dbs[tid];
        int const t = tid;
        for (int ss = 0; ss < t; ++ss)
          acc = fmaf(so.dL[t * C + ss] / so.bs[t], so.KKT[t * C + ss], acc);
        so.dbs[tid] = acc;
      }

    // ---- 12 + 14a. dkd += dLtmp ki and dqd += dpt ki, one ki staging -----
    // Same B operand (`ki` as MN-major), so one staging round and one
    // `retire_gemm` serve both.  `dpt` depends only on `dp` (phase 10), so it is
    // simply computed here instead of in phase 14; the second CV A tile lives in
    // the tail of `stage_a_cvk`, which is 4 KiB but only needs 512 B per CV
    // operand.  Each accumulator keeps its own contribution order.
    for (int i = tid; i < C * C; i += NT) {
      int const t  = i / C;
      int const ss = i - t * C;
      so.X[i]      = (ss <= t) ? so.dp[i] : 0.0f;  // dpt
    }
    __syncthreads();
    for (int i = tid; i < C * C; i += NT) {
      sA_cv(make_coord(i / C, i % C))  = Element(so.dL[i]);  // A = dLtmp
      sA2_cv(make_coord(i / C, i % C)) = Element(so.X[i]);   // A = dpt
    }
    __syncthreads();
    for (int half = 0; half < Vt / 64; ++half) {
      stage_mn_major(sB_cv, so.ki, 64, C, D, half * 64);
      __syncthreads();
      auto acc_dkd = partition_fragment_C(mma_cv, Shape<Int<C>, Int<64>>{});
      auto acc_dqd = partition_fragment_C(mma_cv, Shape<Int<C>, Int<64>>{});
      run_gemm_pair_zero(mma_cv, acc_dkd, sA_cv, acc_dqd, sA2_cv, sB_cv);
      acc_into_smem(acc_dkd, cC_cv, s.dkd, D, half * 64);
      acc_into_smem(acc_dqd, cC_cv, s.dqd, D, half * 64);
      __syncthreads();
    }

    // ---- 13. dki = dLtmp^T kd -------------------------------------------
    for (int i = tid; i < C * C; i += NT) {
      int const m             = i / C;
      int const k             = i - m * C;
      sA_cv(make_coord(m, k)) = Element(so.dL[k * C + m]);
    }
    for (int half = 0; half < Vt / 64; ++half) {
      stage_mn_major(sB_cv, g_kd, 64, C, D, half * 64);
      __syncthreads();
      auto acc = partition_fragment_C(mma_cv, Shape<Int<C>, Int<64>>{});
      run_gemm_zero(mma_cv, acc, sA_cv, sB_cv);
      acc_into_smem(acc, cC_cv, s.dki, D, half * 64);
      __syncthreads();
    }

    // ---- 14b. dki += dpt^T qd -------------------------------------------
    for (int i = tid; i < C * C; i += NT) {
      int const m             = i / C;
      int const k             = i - m * C;
      sA_cv(make_coord(m, k)) = Element(so.X[k * C + m]);
    }
    for (int half = 0; half < Vt / 64; ++half) {
      stage_mn_major(sB_cv, g_qd, 64, C, D, half * 64);
      __syncthreads();
      auto acc = partition_fragment_C(mma_cv, Shape<Int<C>, Int<64>>{});
      run_gemm_zero(mma_cv, acc, sA_cv, sB_cv);
      acc_into_smem(acc, cC_cv, s.dki, D, half * 64);
      __syncthreads();
    }

    // ---- 15. decay reduction + gate chain -------------------------------
    static_assert((C * D) % NT == 0, "the unrolled epilogue sweeps assume an exact trip count");
    // One 4-element vector per lane: the four fp32 smem fields move as single
    // 16 B accesses and decayed/ki as single 8 B accesses.  Only the kr slot
    // permutation stays scalar, because kr_slot(d) is not contiguous in d.
    // The per-element expression and its order are unchanged.
    {
      int const i0 = tid * 4;
      int const t  = i0 / D;
      int const d  = i0 - t * D;
      float     e[4];
      MUTE_UNROLL
      for (int j = 0; j < 4; ++j) {
        e[j] = (t == actual - 1) ? static_cast<float>(rec_total[d + j])
                                 : static_cast<float>(Element(exp2_ftz(s.dG[i0 + j])));
      }
      mutlass::Array<float, 4> const   dkd4 = *reinterpret_cast<mutlass::Array<float, 4> const *>(&s.dkd[i0]);
      mutlass::Array<float, 4> const   dki4 = *reinterpret_cast<mutlass::Array<float, 4> const *>(&s.dki[i0]);
      mutlass::Array<float, 4> const   dqd4 = *reinterpret_cast<mutlass::Array<float, 4> const *>(&s.dqd[i0]);
      mutlass::Array<float, 4> const   dkr4 = *reinterpret_cast<mutlass::Array<float, 4> const *>(&s.dkr[i0]);
      mutlass::Array<Element, 4> const kd4  = *reinterpret_cast<mutlass::Array<Element, 4> const *>(&g_kd[i0]);
      mutlass::Array<Element, 4> const qd4  = *reinterpret_cast<mutlass::Array<Element, 4> const *>(&g_qd[i0]);
      mutlass::Array<Element, 4> const ki4  = *reinterpret_cast<mutlass::Array<Element, 4> const *>(&so.ki[i0]);
      mutlass::Array<float, 4>         dgo;
      mutlass::Array<float, 4>         dko;
      mutlass::Array<float, 4>         dqo;
      MUTE_UNROLL
      for (int j = 0; j < 4; ++j) {
        float const dkdv  = dkd4[j];
        float const dkiv  = dki4[j];
        float const dqdv  = dqd4[j];
        float const dkrv  = dkr4[j];
        float const krv   = static_cast<float>(g_kr[t * D + kr_slot(d + j)]);
        float const dkhat = dkdv * e[j] + dkiv / e[j] + dkrv * so.Etot[d + j] / e[j];
        float const dqhat = dqdv * e[j];
        float       dg    = (dkdv * static_cast<float>(kd4[j]) + dqdv * static_cast<float>(qd4[j]) -
                        dkiv * static_cast<float>(ki4[j]) - dkrv * krv) *
                       kLn2;
        if (t == actual - 1) dg += so.dGtot[d + j];
        dgo[j] = dg;
        dko[j] = dkhat;
        dqo[j] = dqhat;
      }
      *reinterpret_cast<mutlass::Array<float, 4> *>(&s.dG[i0])  = dgo;
      *reinterpret_cast<mutlass::Array<float, 4> *>(&s.dkd[i0]) = dko;
      *reinterpret_cast<mutlass::Array<float, 4> *>(&s.dqd[i0]) = dqo;
    }

    __syncthreads();
    // Suffix scan over the chunk rows, in parallel: each warp covers two head
    // columns (16 lanes each), so the 16 serial smem steps become four shuffle
    // steps.  The summation grouping changes (tree instead of serial), which is
    // the only non-bit-identical part of this pass.
    {
      int const warp = int(tid) / 32;
      int const lane = int(tid) % 32;
      int const t    = lane % 16;
      int const base = lane & 16;
      MUTE_UNROLL
      for (int round = 0; round < D / 32; ++round) {
        int const d = warp * 2 + (lane / 16) + round * 32;
        float     v = s.dG[t * D + d];
        MUTE_UNROLL
        for (int off = 1; off < 16; off <<= 1) {
          float const o = __shfl_sync(uint32_t(-1), v, base + ((t + off) & 15), 32);
          if (t + off < 16) v += o;
        }
        s.dG[t * D + d] = v;
      }
    }
    __syncthreads();
// The gate chain and the `dg` store run in one pass: both walk C*D with the
// same lane-to-index map, so the value the store needs is the one the same
// lane just computed -- the barrier and the smem round-trip between them
// were pure overhead.  (Bit-identical: the old code stored `dx * alpha` to
// smem as float and read it straight back.)
    {
      int const i0 = tid * 4;
      int const t  = i0 / D;
      int const d  = i0 - t * D;
      if (t < actual) {
        mutlass::Array<float, 4> const   g4  = *reinterpret_cast<mutlass::Array<float, 4> const *>(&s.dG[i0]);
        mutlass::Array<ElementGate, 4> const gv4 =
            *reinterpret_cast<mutlass::Array<ElementGate, 4> const *>(
                p.ptr_g + seq_q + (chunk_start + t) * tq + head * hq + d);
        mutlass::Array<float, 4> out;
        MUTE_UNROLL
        for (int j = 0; j < 4; ++j) {
          float const gv =
              static_cast<float>(gv4[j]) + (SafeGate ? dtb[d + j] : 0.f);
          float const inner = alpha * gv;
          float const sig   = 1.0f / (1.0f + __musa_exp2_f(-inner * kLog2E));
          float const dx    = g4[j] * (p.lower_bound * kLog2E) * sig * (1.0f - sig);
          if constexpr (SafeGate)
            out[j] = dx * alpha;
          else if constexpr (GateIsLinear)
            out[j] = (float(gv4[j])>=1e-10f ? g4[j]*kLog2E/float(gv4[j]) : 0.f);
          else
            out[j] = g4[j] * (GateIsLog ? kLog2E : 1.f);
        }
        *reinterpret_cast<mutlass::Array<float, 4> *>(
            p.ptr_dg + seq_q + (chunk_start + t) * tq + head * hq + d) = out;
      }
    }

    if (tid < C) {
      int const t = tid;
      if (t < actual) {
        float const b                        = so.bs[t];
        p.ptr_dbeta[(tok0 + t) * p.H + head] =
            (UseBetaSigmoid && b == 0.f) ? 0.f : so.dbs[t] * beta_derivative(b);
      }
    }

    // ---- 16. q/k L2-normalisation backward ------------------------------
    // ONE warp per token: lane l owns the four consecutive head columns
    // 4l..4l+3 of token t = warp id.  The two dot products are a warp
    // reduction over those four columns, so the coefficients never leave the
    // warp -- no smem round-trip through dGtot/dE15 and no barrier between the
    // dot products and the store.  The summation grouping differs from the old
    // (lane-strided) split, so this is the one step that is not bit-identical.
    {
      int const t  = int(tid) / 32;
      int const d  = (int(tid) % 32) * 4;
      int const i0 = t * D + d;
      if (t < actual) {
        mutlass::Array<Element, 4> const q4 = *reinterpret_cast<mutlass::Array<Element, 4> const *>(
            p.ptr_q + seq_q + (chunk_start + t) * tq + head * hq + d);
        mutlass::Array<Element, 4> const k4 = *reinterpret_cast<mutlass::Array<Element, 4> const *>(
            p.ptr_k + seq_q + (chunk_start + t) * tq + head * hq + d);
        mutlass::Array<float, 4> const dqd4 = *reinterpret_cast<mutlass::Array<float, 4> const *>(&s.dqd[i0]);
        mutlass::Array<float, 4> const dkd4 = *reinterpret_cast<mutlass::Array<float, 4> const *>(&s.dkd[i0]);
        float dotq = 0.0f, dotk = 0.0f;
        if constexpr (NormalizeQK) {
          MUTE_UNROLL
          for (int j = 0; j < 4; ++j) {
            float const yq = static_cast<float>(q4[j]) * s.rq[t];
            float const yk = static_cast<float>(k4[j]) * s.rk[t];
            dotq           = fmaf(dqd4[j], yq, dotq);
            dotk           = fmaf(dkd4[j], yk, dotk);
          }
          MUTE_UNROLL
          for (int off = 16; off > 0; off >>= 1) {
            dotq += __shfl_sync(uint32_t(-1), dotq, (int(tid) % 32) ^ off, 32);
            dotk += __shfl_sync(uint32_t(-1), dotk, (int(tid) % 32) ^ off, 32);
          }
        }
        mutlass::Array<Element, 4> dqo;
        mutlass::Array<Element, 4> dko;
        MUTE_UNROLL
        for (int j = 0; j < 4; ++j) {
          float const qraw = static_cast<float>(q4[j]);
          float const kraw = static_cast<float>(k4[j]);
          float       dqraw, dkraw;
          if constexpr (NormalizeQK) {
            float const yq = qraw * s.rq[t];
            float const yk = kraw * s.rk[t];
            dqraw          = s.rq[t] * p.scale * (dqd4[j] - dotq * yq);
            dkraw          = s.rk[t] * (dkd4[j] - dotk * yk);
          } else {
            dqraw = dqd4[j] * p.scale;
            dkraw = dkd4[j];
          }
          dqo[j] = Element(dqraw);
          dko[j] = Element(dkraw);
        }
        *reinterpret_cast<mutlass::Array<Element, 4> *>(
            p.ptr_dq + seq_q + (chunk_start + t) * tq + head * hq + d) = dqo;
        *reinterpret_cast<mutlass::Array<Element, 4> *>(
            p.ptr_dk + seq_q + (chunk_start + t) * tq + head * hq + d) = dko;
      }
    }
  }
};

// The thin kernel shell K4's launcher instantiates: the collective owns the
// whole body and this only re-exposes the launch contract.
template <class Collective_>
struct ChunkKdaBackwardGradKernel {
  using Collective                        = Collective_;
  using Params                            = typename Collective::Params;
  static constexpr int kChunk             = Collective::C;
  static constexpr int kHeadDim           = Collective::D;
  static constexpr int kValueDim          = Collective::Vt;
  static constexpr int SharedStorageSize  = Collective::SharedStorageSize;
  static constexpr int MaxThreadsPerBlock = Collective::MaxThreadsPerBlock;

  static dim3 get_grid_shape(Params const& params) {
    return Collective::get_grid_shape(params);
  }

  MUTLASS_DEVICE void operator()(Params const& params, char* smem_raw) const {
    Collective{}(params, smem_raw);
  }
};

}  // namespace backward_grad
}  // namespace mate::flat::kda
