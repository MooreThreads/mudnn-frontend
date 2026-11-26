#pragma once

// MP31 chunk-KDA backward scan (K3). Each 128-thread CTA carries a
// 32-value x 128-key block of dS in fp32 through the reverse chunk scan:
//
//   dstate[c] = Element(dS)              (publish the incoming state)
//   dU        = p^T dO + kr dS^T
//   dtmp      = inverse^T Element(dU)
//   dS        = E_total * dS + [-Element(dtmp)^T, dO^T] [kd; qd]
//
// MP31 has no 16x32 SQMMA atom, so CV/CVK pad the token output axis to 32;
// only the first 16 rows feed the following contraction. Packing the two
// state updates along K preserves their accumulation order and retires them
// together. The first half of the VK A tile holds dU until the inverse GEMM
// retires, then holds -dtmp; its second half retains dO throughout the chunk.
//
// P, inverse, and E_total are adjacent in each workspace record. One TME
// transfer loads this 16x40 slab into a 32x40 shared tile; TME fills the padded
// token rows with zero. Other TME loads fetch [kd; qd] and dO while threads
// restore the workspace's KR permutation. dO uses a scalar copy when its
// pointer or strides do not satisfy TME's 16-byte alignment requirement.
//
// Static shared storage is explicitly 1024-byte aligned: the MMA swizzles
// require that base alignment, and static addresses avoid the dynamic-base
// address live ranges that otherwise spill in this persistent loop.

#include <musa_runtime.h>
#include <mutlass/mutlass.h>
#include <mutlass/numeric_conversion.h>

#include <cstdint>
#include <mute/arch/copy_mp31_tme.hpp>
#include <mute/tensor.hpp>
#include <mutlass/arch/barrier.hpp>
#include <mutlass/gemm/collective/collective_builder.hpp>
#include <type_traits>
#include <utility>

#include "mate/common/layout.hpp"
#include "mate/common/mma_mp31_sqmma.hpp"
#include "mate/flat/kda/chunk_kda_workspace.hpp"
#include "mate/flat/kda/named_barrier.hpp"

namespace mate::flat::kda {
namespace backward_scan {

using namespace mute;

template <class Element_, class CuSeqlensElement_, bool IsVarlen_, int Chunk_, int HeadDim_>
struct Mp31ChunkKdaBackwardScanCollective {
  using ArchTag                    = mutlass::arch::Mp31;
  using Element                    = Element_;
  using CuSeqlensElement           = CuSeqlensElement_;
  static constexpr bool IsVarlen   = IsVarlen_;
  static constexpr int  C          = Chunk_;
  static constexpr int  D          = HeadDim_;
  static constexpr int  V          = HeadDim_;
  static constexpr int  kValueTile = 32;
  static_assert(C == 16, "the MP31 K3 scan is the 16-token chunk");
  static_assert(D == 128, "the MP31 K3 scan is the 128-wide head dim");
  static_assert(V == D, "the MP31 K3 scan consumes the whole value axis");
  static_assert(V % kValueTile == 0);

  static constexpr int kThreads          = 128;
  using StrideQK                         = ChunkKdaQkStride;
  static constexpr int kOperandAlignment = 8;
  using DummyStageCount                  = mutlass::gemm::collective::StageCount<2>;
  using Workspace                        = ChunkKdaWorkspace<Element, C, D>;
  static_assert(Workspace::InverseOffset == Workspace::POffset + C * C);
  static_assert(Workspace::TotalOffset == Workspace::InverseOffset + C * C);

  // CV/CVK compute 16 useful token rows with a 32-row MMA atom.
  using TileShapeCV  = Shape<Int<2 * C>, Int<kValueTile>, Int<C>>;
  using TileShapeCVK = Shape<Int<2 * C>, Int<kValueTile>, Int<D>>;
  using TileShapeVK  = Shape<Int<kValueTile>, Int<D>, Int<2 * C>>;

  template <class ShapeMNK, class StrideA = StrideQK, class StrideB = StrideQK>
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

  // MN-major makes the token-strided tiles contiguous along their value/head
  // axis, matching the external dO/decayed storage and the TME load descriptor.
  using CollectiveCV  = MkCollective<TileShapeCV, ChunkKdaVoStride, ChunkKdaVoStride>;
  using CollectiveCVK = MkCollective<TileShapeCVK>;
  using CollectiveVK  = MkCollective<TileShapeVK, ChunkKdaVoStride, ChunkKdaVoStride>;
  using TiledMmaCV    = typename CollectiveCV::TiledMma;
  using TiledMmaCVK   = typename CollectiveCVK::TiledMma;
  using TiledMmaVK    = typename CollectiveVK::TiledMma;
  static_assert(decltype(size(TiledMmaCV{}))::value == kThreads);
  static_assert(decltype(size(TiledMmaCVK{}))::value == kThreads);
  static_assert(decltype(size(TiledMmaVK{}))::value == kThreads);

  // The builder's layouts carry the two-stage policy it is built with; the scan
  // keeps one live stage per operand and re-stages it in place, so the layouts
  // are taken down to a single (M|N, K) tile.
  using SmemLayoutA_CV  = decltype(take<0, 2>(typename CollectiveCV::SmemLayoutA{}));
  using SmemLayoutB_CV  = decltype(take<0, 2>(typename CollectiveCV::SmemLayoutB{}));
  using SmemLayoutA_CVK = decltype(take<0, 2>(typename CollectiveCVK::SmemLayoutA{}));
  using SmemLayoutB_CVK = decltype(take<0, 2>(typename CollectiveCVK::SmemLayoutB{}));
  using SmemLayoutA_VK  = decltype(take<0, 2>(typename CollectiveVK::SmemLayoutA{}));
  using SmemLayoutB_VK  = decltype(take<0, 2>(typename CollectiveVK::SmemLayoutB{}));

  // Both decayed halves form one K=32 operand, loaded in a single transfer.
  using TME_decayed =
      decltype(make_tme_copy(MP31_TME_LOAD{},
                             make_tensor(make_gmem_ptr(static_cast<Element const *>(nullptr)),
                                         make_layout(make_shape(Int<D>{}, Int<2 * C>{}, int32_t{}),
                                                     make_stride(_1{}, Int<D>{}, Int<Workspace::RecordStride>{}))),
                             SmemLayoutB_VK{}));
  static constexpr uint32_t kDecayedBarrier = 1;

  using CollectiveMeta =
      MkCollective<Shape<Int<2 * C>, Int<kValueTile>, Int<4 * C>>, ChunkKdaVoStride, ChunkKdaVoStride>;
  using SmemLayoutMetaPadded = decltype(take<0, 2>(typename CollectiveMeta::SmemLayoutA{}));
  using SmemLayoutMeta       = decltype(composition(
      SmemLayoutMetaPadded{}.layout_a(),
      SmemLayoutMetaPadded{}.offset(),
      make_layout(make_shape(Int<2 * C>{}, Int<2 * C + D / C>{}), make_stride(_1{}, Int<2 * C>{}))));
  using TME_matrix =
      decltype(make_tme_copy(MP31_TME_LOAD{},
                             make_tensor(make_gmem_ptr(static_cast<Element const *>(nullptr)),
                                         make_layout(make_shape(Int<C>{}, Int<2 * C + D / C>{}, int32_t{}),
                                                     make_stride(_1{}, Int<C>{}, Int<Workspace::RecordStride>{}))),
                             SmemLayoutMeta{}));
  using TME_dO = decltype(make_tme_copy(MP31_TME_LOAD{},
                                        make_tensor(make_gmem_ptr(static_cast<Element const *>(nullptr)),
                                                    make_layout(make_shape(Int<V>{}, int32_t{}, int32_t{}, int32_t{}),
                                                                make_stride(_1{}, int64_t{}, int64_t{}, int64_t{}))),
                                        SmemLayoutB_CV{}));
  // ---- TME publishing of the dS block ------------------------------------
  // One TME store publishes each incoming 8 KiB state tile.  This is
  // K2's chunk_states machinery (mp31_chunk_kda_recurrence_collective_...::
  // store_chunk_states_for_chunk) with one deliberate change: K2's tiler
  // carries the 8x8 v-axis transpose its writers bake in, while sB_cv_ds is
  // written straight from the VK C-fragment's logical (m, n), so the two tiler
  // index strides are swapped.
  //
  // The descriptor views one pool row as (b:8, a:16, d:128):(128, 1024, 1),
  // i.e. element offset 1024*a + 128*b + d -- the pool row is (v:128,
  // d:128):(128, 1) with v = 8*a + b.  SmemLayoutStateBlocked re-exposes the
  // smem tile's row r as (r_hi, r_lo) with smem offset 1024*r_hi + 128*r_lo,
  // i.e. r = 8*r_hi + r_lo. With the tile written in logical v order, the tiler
  // must send (r_hi, r_lo) to (a, b) = (r_hi, r_lo), i.e. index strides (8, 1)
  // rather than K2's (1, 8).
  using SmemLayoutState        = SmemLayoutB_CVK;
  using SmemLayoutStateBlocked = decltype(composition(
      SmemLayoutState{}.layout_a(),
      SmemLayoutState{}.offset(),
      make_layout(make_shape(make_shape(_4{}, _8{}), _128{}), make_stride(make_stride(_1024{}, _128{}), _1{}))));
  using TilerDstate =
      decltype(make_layout(make_shape(make_shape(_4{}, _8{}), _128{}), make_stride(make_stride(_8{}, _1{}), _128{})));
  using TME_dstate = decltype(make_tme_copy(MP31_TME_STORE{},
                                            make_tensor(make_gmem_ptr(static_cast<Element *>(nullptr)),
                                                        make_layout(make_shape(_8{}, _16{}, _128{}, int32_t{}),
                                                                    make_stride(_128{}, _1024{}, _1{}, _16384{}))),
                                            SmemLayoutStateBlocked{},
                                            TilerDstate{}));
  // Only the issuing lane may wait on its own store, so the retirement is a
  // squad rendezvous behind one named barrier.
  static constexpr uint32_t kDstateStoreBarrier = 0;
  static constexpr uint32_t kNumNamedBarriers   = 8;

  // 256 B is what a K-major swizzle line needs; the *base* needs 1024 (below).
  static constexpr int SmemAlignmentBytes     = 256;
  static constexpr int SmemBaseAlignmentBytes = 1024;
  template <class SmemLayout>
  using SmemArray = array_aligned<Element, cosize_v<SmemLayout>, SmemAlignmentBytes>;

  struct SharedStorage {
    // P, inverse, and E_total; CV A views select the first two segments.
    SmemArray<SmemLayoutMeta> smem_meta;
    // A of CVK: kr, full K = D.
    SmemArray<SmemLayoutA_CVK> smem_a_cv_kr;
    // B of CVK and the operand the pool publish reads: the carried dS block.
    SmemArray<SmemLayoutB_CVK> smem_b_cv_ds;
    // [dU; dO] becomes [-dtmp; dO] after the inverse GEMM retires.
    SmemArray<SmemLayoutA_VK> smem_a_vk_cols;
    // Both [kd; qd] halves stay live until the packed VK retires.
    SmemArray<SmemLayoutB_VK> smem_b_vk_cols;
  };

  // The kernel declares aligned static storage and needs no dynamic allocation.
  static constexpr int SharedStorageBytes = sizeof(SharedStorage);
  static constexpr int SharedStorageSize  = 0;
  static constexpr int MaxThreadsPerBlock = kThreads;

  struct Arguments {
    Element const          *ptr_dO;
    float const            *ptr_d_final_state;
    void                   *ptr_dstate;   // Element pool [rows, H, V, D], V-major, D contiguous
    float                  *ptr_dstate0;  // [N, H, V, D] fp32, one row per sequence
    Element const          *ptr_ws_decayed;
    Element const          *ptr_ws_k_restored;
    Element const          *ptr_ws_p;
    Element const          *ptr_ws_inverse;
    Element const          *ptr_ws_total;
    int64_t                 stride_o1;  // token
    int64_t                 stride_o2;  // head
    int64_t                 stride_o3;  // batch
    int                     nseq;
    int                     T;
    int                     H;
    int                     workspace_chunks;
    CuSeqlensElement const *cu_seqlens;
    TME_dstate              tme_dstate;
    TME_decayed             tme_decayed;
    TME_matrix              tme_p;
    TME_dO                  tme_dO;
    bool                    use_tme_dO = false;
  };
  using Params = Arguments;

  static MUTLASS_HOST_DEVICE int32_t dstate_pool_rows(Arguments const &args) {
    if constexpr (IsVarlen) {
      return (mutlass::ceil_div(args.T, C) + args.nseq) * args.H;
    } else {
      return args.nseq * args.workspace_chunks * args.H;
    }
  }

  static Params to_underlying_arguments(Arguments const &args) {
    Params  p    = args;
    int32_t rows = dstate_pool_rows(args);
    // One rank-4 descriptor covers the whole pool: dim3 is the flattened
    // (record, head) row and `a` selects the value block, so a per-chunk
    // store only changes coordinates.
    auto pool =
        make_tensor(make_gmem_ptr(static_cast<Element *>(args.ptr_dstate)),
                    make_layout(make_shape(_8{}, _16{}, _128{}, rows), make_stride(_128{}, _1024{}, _1{}, _16384{})));
    p.tme_dstate  = make_tme_copy(MP31_TME_STORE{}, pool, SmemLayoutStateBlocked{}, TilerDstate{});
    auto decayed  = make_tensor(make_gmem_ptr(args.ptr_ws_decayed),
                               make_layout(make_shape(Int<D>{}, Int<2 * C>{}, rows),
                                           make_stride(_1{}, Int<D>{}, Int<Workspace::RecordStride>{})));
    p.tme_decayed = make_tme_copy(MP31_TME_LOAD{}, decayed, SmemLayoutB_VK{});
    auto p_tile   = make_tensor(make_gmem_ptr(args.ptr_ws_p),
                              make_layout(make_shape(Int<C>{}, Int<2 * C + D / C>{}, rows),
                                          make_stride(_1{}, Int<C>{}, Int<Workspace::RecordStride>{})));
    p.tme_p       = make_tme_copy(MP31_TME_LOAD{}, p_tile, SmemLayoutMeta{});
    p.use_tme_dO  = (reinterpret_cast<uintptr_t>(args.ptr_dO) % 16 == 0) && (args.stride_o1 % 8 == 0) &&
                   (args.stride_o2 % 8 == 0) && (args.stride_o3 % 8 == 0);
    if (p.use_tme_dO) {
      auto dO = make_tensor(
          make_gmem_ptr(args.ptr_dO),
          make_layout(make_shape(Int<V>{}, int32_t(args.T), int32_t(args.H), int32_t(IsVarlen ? 1 : args.nseq)),
                      make_stride(_1{}, args.stride_o1, args.stride_o2, args.stride_o3)));
      p.tme_dO = make_tme_copy(MP31_TME_LOAD{}, dO, SmemLayoutB_CV{});
    }
    return p;
  }

  // One CTA per (sequence, head, 32-row value block): K3 has no cross-value
  // reduction, so the split costs nothing and halves the carried accumulator.
  static dim3 get_grid_shape(Params const &p) {
    return dim3(unsigned(p.nseq) * unsigned(p.H) * unsigned(V / kValueTile), 1, 1);
  }

  // A compile-time for that the optimizer cannot refuse.  The plain
  // `#pragma unroll` (MUTE_UNROLL) of the equivalent runtime loop is silently
  // dropped by this compiler -- "loop not unrolled: the optimizer was unable to
  // perform the requested transformation" -- which leaves every gmem load in
  // the loop exposing its own latency and pushes the staging arrays into local
  // memory.  Instantiating the body once per index makes the index a constant,
  // so the loads issue together and the array stays in registers.
  template <class Fn, int... Is>
  MUTLASS_DEVICE static void for_each_impl(Fn const &fn, std::integer_sequence<int, Is...>) {
    (void)std::initializer_list<int>{(fn(std::integral_constant<int, Is>{}), 0)...};
  }
  template <int N, class Fn>
  MUTLASS_DEVICE static void for_each(Fn const &fn) {
    for_each_impl(fn, std::make_integer_sequence<int, N>{});
  }

  // The dynamic-smem base is not strongly aligned (0xc mod 1024 measured on
  // MP31 for 1 KiB, 4 KiB and 183 KiB requests alike), so the storage is
  // placed at the next 1024 B boundary inside the requested region.
  static MUTLASS_DEVICE void squad_sync() {
    __syncthreads();
  }

  static MUTLASS_DEVICE void init_barriers(int tid) {
    mutlass::arch::allocate_async_barriers(kNumNamedBarriers);
    if (tid == 0) {
      named_barrier_init(kDstateStoreBarrier, mutlass::NumWarpsPerWarpSquad);
      named_barrier_init(kDecayedBarrier, 1);
    }
  }

  // One lane, one instruction: the whole (64 x 128) dS block goes out as a
  // single TME store.  No other lane touches the pool.
  MUTLASS_DEVICE static void issue_dstate_store(
      Params const &p, SharedStorage &s, int v0, int pool_row, int32_t pool_rows) {
    auto sState  = make_tensor(make_smem_ptr(s.smem_b_cv_ds.data()), SmemLayoutStateBlocked{});
    auto gcoord  = p.tme_dstate.get_tme_tensor(make_shape(Int<8>{}, Int<16>{}, Int<128>{}, pool_rows));
    auto gc      = domain_offset(make_coord(_0{}, int32_t(v0 / 8), _0{}, int32_t(pool_row)), gcoord);
    auto cta_tme = p.tme_dstate.get_slice(_0{});
    auto tS      = group_modes<0, 3>(cta_tme.partition_S(sState));
    auto tG      = group_modes<0, 3>(cta_tme.partition_D(gc));
    copy(p.tme_dstate, tS, tG(_, _0{}, _0{}));
    tme_store_arrive();
  }

  MUTLASS_DEVICE static void issue_inputs(
      Params const &p, SharedStorage &s, int pool_row, int seq, int head, int v0, int token) {
    uint32_t const bytes =
        (2 * C * D + 2 * C * (2 * C + D / C) + (p.use_tme_dO ? C * kValueTile : 0)) * sizeof(Element);
    named_barrier_arrive_and_expect_tx(kDecayedBarrier, bytes);
    auto sKd = make_tensor(make_smem_ptr(s.smem_b_vk_cols.data()), SmemLayoutB_VK{});
    auto gd  = p.tme_decayed.get_tme_tensor(make_shape(Int<D>{}, Int<2 * C>{}, dstate_pool_rows(p)));
    auto td  = p.tme_decayed.get_slice(_0{});
    auto gk  = domain_offset(make_coord(_0{}, 0, pool_row), gd);
    auto tGk = group_modes<0, 2>(td.partition_S(gk));
    auto tSk = group_modes<0, 2>(td.partition_D(sKd));
    copy(p.tme_decayed.with(named_barrier_id(kDecayedBarrier)), tGk(_, _0{}, _0{}), tSk(_, _0{}));
    auto sMeta = make_tensor(make_smem_ptr(s.smem_meta.data()), SmemLayoutMeta{});
    auto gP    = p.tme_p.get_tme_tensor(make_shape(Int<C>{}, Int<2 * C + D / C>{}, dstate_pool_rows(p)));
    auto tp    = p.tme_p.get_slice(_0{});
    auto gp    = domain_offset(make_coord(_0{}, _0{}, pool_row), gP);
    auto tGp   = group_modes<0, 2>(tp.partition_S(gp));
    auto tSp   = group_modes<0, 2>(tp.partition_D(sMeta));
    copy(p.tme_p.with(named_barrier_id(kDecayedBarrier)), tGp(_, _0{}, _0{}), tSp(_, _0{}));
    if (p.use_tme_dO) {
      auto sDo = make_tensor(make_smem_ptr(s.smem_a_vk_cols.data() + C * kValueTile), SmemLayoutB_CV{});
      auto gDo =
          p.tme_dO.get_tme_tensor(make_shape(Int<V>{}, int32_t(p.T), int32_t(p.H), int32_t(IsVarlen ? 1 : p.nseq)));
      auto go  = domain_offset(make_coord(v0, token, head, IsVarlen ? 0 : seq), gDo);
      auto to  = p.tme_dO.get_slice(_0{});
      auto tGo = group_modes<0, 2>(to.partition_S(go));
      auto tSo = group_modes<0, 2>(to.partition_D(sDo));
      copy(p.tme_dO.with(named_barrier_id(kDecayedBarrier)), tGo(_, _0{}, _0{}, _0{}), tSo(_, _0{}));
    }
  }

  // Only the issuing lane can wait; the rendezvous is what tells the rest of
  // the squad that the tile is free to overwrite.
  MUTLASS_DEVICE static void retire_dstate_store(int tid) {
    if (tid == 0) {
      tme_store_wait();
    }
    sync_warp_squad(int(kDstateStoreBarrier));
  }

  static MUTLASS_DEVICE void mma_retire() {
    mate::warpsquad_commit_batch();
    mate::warpsquad_wait();
  }

  MUTLASS_DEVICE static void run(Params const &p, char *smem_raw) {
    __shared__ __align__(1024) SharedStorage s;
    int const                                tid     = int(threadIdx.x);
    constexpr int                            kBlocks = V / kValueTile;
    int const                                v_blk   = int(blockIdx.x) % kBlocks;
    int const                                head    = (int(blockIdx.x) / kBlocks) % p.H;
    int const                                seq     = int(blockIdx.x) / (kBlocks * p.H);
    int const                                v0      = v_blk * kValueTile;

    init_barriers(tid);
    squad_sync();

    int64_t bos, seq_len, workspace_base;
    if constexpr (IsVarlen) {
      bos            = int64_t(p.cu_seqlens[seq]);
      seq_len        = int64_t(p.cu_seqlens[seq + 1]) - bos;
      workspace_base = (bos + C - 1) / C + seq;
    } else {
      bos            = int64_t(seq) * p.T;
      seq_len        = p.T;
      workspace_base = int64_t(seq) * p.workspace_chunks;
    }
    int const n_chunks = int((seq_len + C - 1) / C);
    if (n_chunks == 0) {
      return;
    }

    // The dense layout carries its batch index in an explicit stride; the
    // packed (varlen) layout has a flat token axis and `bos` is already a
    // global token index.  Neither may be conflated with the stride itself.
    int64_t const to1   = p.stride_o1;
    int64_t const to2   = p.stride_o2;
    int64_t const seq_o = IsVarlen ? bos * to1 : int64_t(seq) * p.stride_o3;

    auto sA_cv_cols = make_tensor(make_smem_ptr(s.smem_meta.data()), SmemLayoutA_CV{});
    auto sA_cv_kr   = make_tensor(make_smem_ptr(s.smem_a_cv_kr.data()), SmemLayoutA_CVK{});
    auto sB_cv_cols = make_tensor(make_smem_ptr(s.smem_a_vk_cols.data()), SmemLayoutB_CV{});
    auto sB_cv_ds   = make_tensor(make_smem_ptr(s.smem_b_cv_ds.data()), SmemLayoutB_CVK{});
    auto sA_vk_cols = make_tensor(make_smem_ptr(s.smem_a_vk_cols.data()), SmemLayoutA_VK{});
    auto sB_vk_cols = make_tensor(make_smem_ptr(s.smem_b_vk_cols.data()), SmemLayoutB_VK{});
    auto sDo        = make_tensor(make_smem_ptr(s.smem_a_vk_cols.data() + C * kValueTile), SmemLayoutB_CV{});
    auto sInv       = make_tensor(make_smem_ptr(s.smem_meta.data() + 2 * C * C), SmemLayoutA_CV{});
    auto sMeta      = make_tensor(make_smem_ptr(s.smem_meta.data()), SmemLayoutMeta{});

    TiledMmaCV  tiled_cv;
    TiledMmaCVK tiled_cvk;
    TiledMmaVK  tiled_vk;
    auto        thr_cv  = tiled_cv.get_thread_slice(tid);
    auto        thr_cvk = tiled_cvk.get_thread_slice(tid);
    auto        thr_vk  = tiled_vk.get_thread_slice(tid);

    // Descriptors are addresses only, so they are built once and survive every
    // in-place re-staging below.
    auto tA_cv_cols = thr_cv.make_fragment_A(thr_cv.partition_A(sA_cv_cols));
    auto tB_cv_cols = thr_cv.make_fragment_B(thr_cv.partition_B(sB_cv_cols));
    auto tA_cv_kr   = thr_cvk.make_fragment_A(thr_cvk.partition_A(sA_cv_kr));
    auto tB_cv_ds   = thr_cvk.make_fragment_B(thr_cvk.partition_B(sB_cv_ds));
    auto tA_vk_cols = thr_vk.make_fragment_A(thr_vk.partition_A(sA_vk_cols));
    auto tB_vk_cols = thr_vk.make_fragment_B(thr_vk.partition_B(sB_vk_cols));
    auto tA_inv     = thr_cv.make_fragment_A(thr_cv.partition_A(sInv));
    auto tB_do      = thr_cv.make_fragment_B(thr_cv.partition_B(sDo));

    auto          cCV    = make_identity_tensor(make_shape(Int<2 * C>{}, Int<kValueTile>{}));
    auto          tCcCV  = thr_cv.partition_C(cCV);
    auto          cVK    = make_identity_tensor(make_shape(Int<kValueTile>{}, Int<D>{}));
    auto          tCcVK  = thr_vk.partition_C(cVK);
    constexpr int kAccCV = (2 * C * kValueTile) / kThreads;  //  8
    constexpr int kAccVK = (kValueTile * D) / kThreads;      // 64
    static_assert(kAccCV * kThreads == 2 * C * kValueTile);
    static_assert(kAccVK * kThreads == kValueTile * D);

    // The carried state gradient, value-block local: one (64 x D) fp32 C
    // fragment.  Its C layout is shared with both VK contractions, so the same
    // accumulator feeds each of them in place.
    auto acc_ds = partition_fragment_C(tiled_vk, make_shape(Int<kValueTile>{}, Int<D>{}));
    clear(acc_ds);
    if (p.ptr_d_final_state != nullptr) {
      float const *src = p.ptr_d_final_state + (int64_t(seq) * p.H + head) * V * D + int64_t(v0) * D;
      MUTE_UNROLL
      for (int i = 0; i < kAccVK; ++i) {
        auto c    = tCcVK(i);
        acc_ds(i) = src[int(get<0>(c)) * D + int(get<1>(c))];
      }
    }

    for_each<C * D / kThreads>([&](auto itc) {
      int i                      = tid + itc.value * kThreads;
      sA_cv_kr(C + i / D, i % D) = Element(0.f);
    });
    squad_sync();
    for (int ci = n_chunks - 1; ci >= 0; --ci) {
      int64_t const chunk_start = int64_t(ci) * C;
      int const     actual      = int(seq_len - chunk_start) < C ? int(seq_len - chunk_start) : C;
      int const     rec         = int((workspace_base + ci) * p.H + head);

      Element const *rec_kr = p.ptr_ws_k_restored + int64_t(rec) * Workspace::RecordStride;

      // Retire the preceding asynchronous publish before overwriting its tile.
      if (ci != n_chunks - 1) {
        retire_dstate_store(tid);
      }

      // ---- the INCOMING dS, published before anything updates it ------------
      MUTE_UNROLL
      for (int i = 0; i < kAccVK; ++i) {
        auto c                                   = tCcVK(i);
        sB_cv_ds(int(get<0>(c)), int(get<1>(c))) = Element(acc_ds(i));
      }
      squad_sync();

      // ---- publish the INCOMING dS through TME ------------------------------
      if (tid == 0) {
        issue_dstate_store(p, s, v0, rec, dstate_pool_rows(p));
        issue_inputs(p, s, rec, seq, head, v0, int((IsVarlen ? bos : 0) + chunk_start));
      }

      // ---- dU += kr dS_v^T --------------------------------------------------
      // The k_restored workspace stores its head-dim axis permuted, exactly as
      // K1 wrote it for K2's UK operand. Read contiguous physical slots, then
      // undo that permutation at the shared-memory store.
      {
        constexpr int kIters = (C * D) / kThreads;
        Element       v[kIters];
        for_each<kIters>([&](auto itc) {
          constexpr int it = itc.value;
          int const     i  = tid + it * kThreads;
          v[it]            = rec_kr[i];
        });
        for_each<kIters>([&](auto itc) {
          constexpr int it   = itc.value;
          int const     i    = tid + it * kThreads;
          int const     slot = i % D;
          int const     k    = 16 * (slot & 7) + (slot >> 3);
          sA_cv_kr(i / D, k) = v[it];
        });
      }
      if (!p.use_tme_dO) {
        for_each<C * kValueTile / kThreads>([&](auto itc) {
          int i   = tid + itc.value * kThreads;
          int row = i / kValueTile;
          int col = i % kValueTile;
          sDo(col, row) =
              row < actual ? p.ptr_dO[seq_o + int64_t(chunk_start + row) * to1 + head * to2 + v0 + col] : Element(0.f);
        });
      }
      wait_warp(kDecayedBarrier, (n_chunks - 1 - ci) & 1);
      if constexpr (IsVarlen) {
        if (actual < C) {
          for_each<C * kValueTile / kThreads>([&](auto itc) {
            int i = tid + itc.value * kThreads;
            if (i / kValueTile >= actual) sDo(i % kValueTile, i / kValueTile) = Element(0.f);
          });
        }
      }
      squad_sync();
      auto acc_u           = partition_fragment_C(tiled_cv, make_shape(Int<2 * C>{}, Int<kValueTile>{}));
      tiled_cv.accumulate_ = MP31::SQMMA::ScaleOut::Zero;
      gemm(tiled_cv, acc_u, tA_cv_cols, tB_do, acc_u);
      tiled_cv.accumulate_ = MP31::SQMMA::ScaleOut::One;
      gemm(tiled_cvk, acc_u, tA_cv_kr, tB_cv_ds, acc_u);
      mma_retire();

      // ---- dtmp = a^T dU
      MUTE_UNROLL
      for (int i = 0; i < kAccCV; ++i) {
        auto c = tCcCV(i);
        if (int(get<0>(c)) < C) sB_cv_cols(int(get<1>(c)), int(get<0>(c))) = Element(acc_u(i));
      }
      squad_sync();
      auto acc_tmp         = partition_fragment_C(tiled_cv, make_shape(Int<2 * C>{}, Int<kValueTile>{}));
      tiled_cv.accumulate_ = MP31::SQMMA::ScaleOut::Zero;
      gemm(tiled_cv, acc_tmp, tA_inv, tB_cv_cols, acc_tmp);
      mma_retire();

      // ---- dS_v = E_tot (.) dS_v;  dS_v -= dtmp^T kd;  dS_v += dO_v^T qd
      // -----
      MUTE_UNROLL
      for (int i = 0; i < kAccVK; ++i) {
        auto c = tCcVK(i);
        acc_ds(i) *= static_cast<float>(sMeta(int(get<1>(c)) % C, 2 * C + int(get<1>(c)) / C));
      }
      MUTE_UNROLL
      for (int i = 0; i < kAccCV; ++i) {
        auto c = tCcCV(i);
        if (int(get<0>(c)) < C) sA_vk_cols(int(get<1>(c)), int(get<0>(c))) = Element(-acc_tmp(i));
      }
      squad_sync();
      gemm(tiled_vk, acc_ds, tA_vk_cols, tB_vk_cols, acc_ds);
      mma_retire();
    }

    // The last store must be retired before the kernel (and the pool) is handed
    // back; only its issuing lane can wait.
    retire_dstate_store(tid);

    // After c = 0 the accumulator is dL/dS_0.
    if (p.ptr_dstate0 != nullptr) {
      float *dst = p.ptr_dstate0 + (int64_t(seq) * p.H + head) * V * D + int64_t(v0) * D;
      MUTE_UNROLL
      for (int i = 0; i < kAccVK; ++i) {
        auto c                                   = tCcVK(i);
        dst[int(get<0>(c)) * D + int(get<1>(c))] = acc_ds(i);
      }
    }
  }
};

}  // namespace backward_scan
}  // namespace mate::flat::kda
