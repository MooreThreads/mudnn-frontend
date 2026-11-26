#pragma once

#include <musa_runtime.h>
#include <mutlass/mutlass.h>
#include <mutlass/numeric_conversion.h>

#include <cassert>
#include <cstdint>
#include <limits>
#include <mute/arch/copy_mp31.hpp>
#include <mute/arch/copy_mp31_tme.hpp>
#include <mute/atom/copy_atom.hpp>
#include <mute/atom/copy_traits_mp31.hpp>
#include <mute/swizzle_layout.hpp>
#include <mute/tensor.hpp>
#include <type_traits>

#include "mate/common/layout.hpp"
#include "mate/common/mma_mp31_sqmma.hpp"
#include "mate/flat/collective/flat_collective_inverse_nxn.hpp"
#include "mate/flat/collective/flat_collective_mma_tme_sqmma.hpp"
#include "mate/flat/kda/chunk_kda_components.hpp"
#include "mate/flat/kda/collective/chunk_kda_mma_policy.hpp"
#include "mate/flat/kda/kda_arch_traits.hpp"
#include "mate/flat/kda/kda_permutes.hpp"
#include "mate/flat/kda/named_barrier.hpp"
#include "mate/flat/simd_helper.hpp"

// The MP31 frontend exposes the correctly-rounded reciprocal-square-root
// builtin in BuiltinsMTGPU.def but does not declare it in the public math
// headers.  Declare the device entry explicitly so mcc can select the native
// instruction without the generic rsqrtf edge-case path.
extern "C" __device__ float __musa_rsqrt_rn_f(float);

namespace mate::flat::kda {

namespace prepare {

using namespace mute;

template <class Components_, class TileShape_, class StrideQ_, class StrideK_, class StrideG_>
struct Mp31ChunkKdaPrepareCollectiveTmeWarpSpecialized {
  static constexpr float kLog2E = 1.4426950408889634074f;

  using ArchTag                       = mutlass::arch::Mp31;
  using Components = Components_;
  using Element = typename Components::Element;
  using ElementGate = typename Components::ElementGate;
  using ElementBeta = typename Components::ElementBeta;
  static constexpr bool UseBetaSigmoid = Components::UseBetaSigmoid;
  static constexpr bool AllowNegEigval = Components::AllowNegEigval;
  static constexpr bool GateIsLog = Components::GateIsLog;
  static constexpr bool GateIsLinear = Components::GateIsLinear;
  using TileShape                     = TileShape_;
  using StrideQ                       = StrideQ_;
  using StrideK                       = StrideK_;
  using StrideV                       = ChunkKdaVoStride;
  using StrideG                       = StrideG_;
  static constexpr bool HasGateParams = Components::HasGateParams;
  static constexpr bool NormalizeQK   = Components::NormalizeQK;
  static constexpr bool IsVarlen      = Components::IsVarlen;

  static constexpr int kChunk   = Components::kChunk;
  static constexpr int kHeadDim = Components::kHeadDim;
  // The order K1 writes the k_restored headDim axis in; see kda_permutes.hpp.
  using HeadDimPermute           = PermuteN<mutlass::arch::Mp31, kHeadDim>;
  using LogicalToSlot            = typename HeadDimPermute::LogicalToSlot;
  static constexpr int kValueDim = Components::kValueDim;
  using Workspace                = typename Components::Workspace;
  static_assert(decltype(get<0>(TileShape{}))::value == kChunk);
  static_assert(decltype(get<1>(TileShape{}))::value == kChunk);
  static_assert(decltype(get<2>(TileShape{}))::value == kHeadDim);
  static constexpr int kValueMmaN = kValueDim < 64 ? kValueDim : 64;
  static_assert(decltype(get<0>(TileShape{}))::value == decltype(get<1>(TileShape{}))::value);
  static_assert(kChunk == 16, "the 32x32 SQMMA warp-role permutation requires chunk size 16");

  // Two warp squads: the producer squad owns warps 0-3 and the SQMMA/epilogue
  // squad owns warps 4-7, the same split the recurrence kernel uses.
  //
  //   warp 0: load Q -> registers -> RMS norm -> smem_norm_q
  //   warp 1: load K -> registers -> RMS norm -> smem_norm_k
  //   warp 2: load G -> gate cumsum (fp32) + workspace.total
  //   warp 3: the 16x16 lower-triangular solve, the fold's A-operand staging,
  //           the P-side epilogue (A_rec = P @ inv on ONE warp's wmma) and the
  //           inverse publish, and nothing else
  //   warps 4-7: the chunk pipeline -
  //              [scale_normalized_qk] decayed = norm*q * gate, k_inverse =
  //              norm*k / gate, k_restored staged in the fold's B layout;
  //              [decayed @ k_inverse] the combined KKT + P SQMMA, scattered
  //              into the KKT matrix and the staged P tile for warp 3;
  //              [inv^T @ k_restored] the fold SQMMA, whose result T is written
  //              directly to the slot-ordered workspace frame - plus the TME store.
  //
  // The inverse is consumed inside K1 (the squad's fold and warp 3's A_rec),
  // so the solve sits on warp 3 between its two consumers; the recurrence
  // additionally reads the published inverse field until its own IP GEMM is
  // folded away (a separate step).
  static constexpr int NumProducerWarps   = 4;
  static constexpr int NumConsumerWarps   = 4;
  static constexpr int NumThreadsPerBlock = (NumProducerWarps + NumConsumerWarps) * mutlass::NumThreadsPerWarp;
  // Global warp index of the first consumer warp.  Everything the consumer
  // squad derives from `mutlass::canonical_warp_idx()` subtracts it.
  static constexpr int ConsumerWarpBase = NumProducerWarps;
  static constexpr int NormalizeQWarp   = 0;
  static constexpr int NormalizeKWarp   = 1;
  static constexpr int GateWarp         = 2;
  static constexpr int InverseWarp      = 3;
  static_assert(NumConsumerWarps * mutlass::NumThreadsPerWarp == kHeadDim);

  // Warp-local index of the consumer warp that issues the TMA stores; its
  // global index is ConsumerWarpBase + StoreConsumerWarp.
  static constexpr int StoreConsumerWarp = NumConsumerWarps - 1;

  static constexpr int SmemAlignmentBytes = 256;
  static constexpr int VecSize            = 4;

  // Async barrier ids are hardware ids, not shared-memory byte offsets.
  // Keep the named barrier range and the two pipeline ranges disjoint.

  // The MUTLASS builder requires at least two stages, but the producer
  // register path only needs the consumer's operand tiles.
  using BuilderStageCount = mutlass::gemm::collective::StageCount<2>;
  using TileShapeQK       = Shape<Int<2 * kChunk>, Int<2 * kChunk>, Int<kHeadDim>>;
  // The SQMMA operand layouts follow ArchTag: MP32 needs its own SQMMA
  // encoding (the MP31 atom does not survive the backend on an mp_32 target),
  // while the workspace drain path is selected separately below.
  using MmaPolicy = ChunkKdaMmaPolicy<ArchTag, Element, 2, BuilderStageCount, mutlass::gemm::KernelTmeWarpSpecialized>;

  using CollectiveMmaQK              = typename MmaPolicy::template Collective<TileShapeQK, StrideQ, StrideK>;
  static constexpr int BuilderStages = CollectiveMmaQK::DispatchPolicy::Stages;
  static_assert(BuilderStages == BuilderStageCount::value);

  using TiledMmaQK = typename CollectiveMmaQK::TiledMma;
  static_assert(decltype(size(TiledMmaQK{}))::value == NumConsumerWarps * mutlass::NumThreadsPerWarp);

  // ---- the inverse fold: T = inverse_beta^T * k_restored -------------------
  // The tile is M = 16 (chunk token) x N = 128 (head) x K = 16 (chunk).  MP31's
  // bf16 SQMMA has no M=16/N=128 atom, so the atom is MP31_32x128x16: M is padded
  // 16 -> 32 and the atom's extra rows are dropped, while N = 128 is covered by
  // ONE atom in ONE warpsquad -- no head split, no second gemm.  A =
  // inverse_beta^T declares K-major, which is the solve's own column layout; B =
  // k_restored declares MN-major, i.e. the head axis is contiguous, exactly how the
  // scale pass writes it: nothing is transposed or repacked.  The scale pass
  // writes that head axis in NATURAL order (which vectorizes the staging
  // store); the workspace's LogicalToSlot permutation is applied at the T
  // writeback instead.
  static constexpr int kNPadKRInv = 32;  // the atom's M; rows >= kChunk are dropped
  using AtomKRInv                 = conditional_t<is_same_v<Element, mutlass::bfloat16_t>,
                                                  mute::MP31_32x128x16_F32BF16BF16_SS<mute::TCE::Major::K, mute::TCE::Major::MN>,
                                                  mute::MP31_32x128x16_F32F16F16_SS<mute::TCE::Major::K, mute::TCE::Major::MN>>;
  using TiledMmaKRInv             = decltype(mute::make_tiled_mma(AtomKRInv{}));
  static_assert(decltype(size(TiledMmaKRInv{}))::value == NumConsumerWarps * mutlass::NumThreadsPerWarp);
  using SmemLayoutInverseBeta = decltype(mute::tile_to_shape(
      mute::MP31::SQMMA::
          make_canonical_gemm_smem_atom_layout<Element, mute::TCE::Major::K, Int<kNPadKRInv>, Int<kChunk>>(),
      Shape<Int<kNPadKRInv>, Int<kChunk>>{}));
  using SmemLayoutKRestore    = decltype(mute::tile_to_shape(
      mute::MP31::SQMMA::
          make_canonical_gemm_smem_atom_layout<Element, mute::TCE::Major::MN, Int<kHeadDim>, Int<kChunk>>(),
      Shape<Int<kHeadDim>, Int<kChunk>>{}));
  static_assert(cosize_v<SmemLayoutInverseBeta> == kNPadKRInv * kChunk);
  static_assert(cosize_v<SmemLayoutKRestore> == kHeadDim * kChunk);

  static constexpr int PipelineStages = 1;
  // Depth of the raw Q/K/G staging.  Two stages let a producer warp issue the
  // next chunk's TMA before the consumers have finished reading the current
  // one, which is the whole point of the TME load; one stage leaves a single
  // 4 KiB request in flight per warp and exposes a full TMA round trip on the
  // critical path.  Costs 3 * 4 KiB of shared memory.
  static constexpr int RawStages = 1;
  using SmemLayoutDecayed =
      decltype(mate::unstage_smem_layout(typename CollectiveMmaQK::SmemLayoutA{}, Int<PipelineStages>{}));
  using SmemLayoutKInverse =
      decltype(mate::unstage_smem_layout(typename CollectiveMmaQK::SmemLayoutB{}, Int<PipelineStages>{}));
  using CollectiveInverseNxN = collective::CollectiveLowerTriangularInverseNxN<Element, kChunk>;

  // Producer/consumer tiles: plain row-major layouts (no TME, no SQMMA
  // access), fp32 for the gate cumsum, bf16 for the normalized tiles.
  using NormLayout   = Layout<Shape<Int<kChunk>, Int<kHeadDim>>, Stride<Int<kHeadDim>, _1>>;
  using CumsumLayout = NormLayout;

  // ---------------------------------------------------------------------
  // Raw Q/K/G tiles arrive through TME.
  //
  // None of the three has a swizzle or a fold: each is a plain row-major
  // 16x128 staging buffer, so a TMA box describes it exactly and the smem
  // destination needs no padding.  The gmem side is the (B*T, Hqk, kHeadDim)
  // view of the input -- token stride and head stride come straight from
  // `stride_q` / `stride_k` / `stride_g`, so a q|k|v-interleaved or otherwise
  // strided input is addressed identically to the old robust LDGSTS path.
  // (The batch fold requires the batch stride to be the token stride times T,
  // which `to_underlying_arguments` checks.)
  //
  // The smem view is the same 16x128 block seen as (kChunk, 1, kHeadDim) with
  // a zero-stride singleton middle mode so that the box mode order matches the
  // gmem mode order.
  using SmemLayoutRaw = Layout<Shape<Int<kChunk>, _1, Int<kHeadDim>>, Stride<Int<kHeadDim>, _0, _1>>;
  // The raw staging buffers are [RawStages][kChunk * kHeadDim] so a stage is a
  // plain pointer offset; the per-stage box view is still SmemLayoutRaw.
  using RawStageLayout =
      Layout<Shape<Int<kChunk>, Int<kHeadDim>, Int<RawStages>>, Stride<Int<kHeadDim>, _1, Int<kChunk * kHeadDim>>>;
  static constexpr int kRawTileElems      = kChunk * kHeadDim;
  using RawGmemShape                      = Shape<int, int, Int<kHeadDim>>;
  using RawGmemStride                     = Stride<int64_t, int64_t, _1>;
  static constexpr uint32_t kRawTileBytes = uint32_t(kChunk) * uint32_t(kHeadDim) * uint32_t(sizeof(Element));
  static_assert(kRawTileBytes == 4096, "the TMA box is one 4 KiB 16x128 bf16 tile");

  using TMELoadRaw =
      decltype(make_tme_copy(MP31_TME_LOAD{},
                             make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
                                         make_layout(RawGmemShape{}, make_stride(int64_t{}, int64_t{}, _1{}))),
                             SmemLayoutRaw{}));
  using TMELoadGate = decltype(make_tme_copy(
      MP31_TME_LOAD{},
      make_tensor(
          make_gmem_ptr(static_cast<ElementGate const *>(nullptr)),
          make_layout(RawGmemShape{}, make_stride(int64_t{}, int64_t{}, _1{}))),
      SmemLayoutRaw{}));

  // Staging tile for the P publish: the SQMMA C fragment scatters into it, and
  // it is read back one contiguous vector per lane (see compute_inverse_and_p).
  using PStagingLayout                = Layout<Shape<Int<kChunk>, Int<kChunk>>, Stride<Int<kChunk>, _1>>;
  static constexpr int kPStageLanes   = 2 * mutlass::NumThreadsPerWarp;
  static constexpr int kPStagePerLane = kChunk * kChunk / kPStageLanes;
  static_assert(kChunk * kChunk % kPStageLanes == 0);
  static_assert(kPStagePerLane == VecSize);

  static constexpr int KInverseValidBytes = kChunk * kHeadDim * sizeof(Element);

  // TME store of the decayed tile (issued by producer-squad warp 3).
  using TMEWorkspaceDecayed = decltype(make_tme_copy(
      MP31_TME_STORE{},
      make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), typename Workspace::DecayedLayout{}),
      take<0, 2>(SmemLayoutDecayed{})));

  // The T writeback deliberately has NO mute gmem layout.  A layout's offset is
  // `sum(stride * coord)`, and with a 32-bit coordinate that product wraps at
  // 2 GiB; the chunk-major workspace crosses 2 GiB once a case has more than
  // 114912 records (a 16224-token, 128-head sequence has 129920), and every
  // record past the wrap was silently left unwritten.  The writeback therefore
  // does its own 64-bit pointer arithmetic; see the fold's epilogue.

  template <class StorageElement, class SmemLayout>
  using SmemArray = array_aligned<StorageElement, cosize_v<SmemLayout>, SmemAlignmentBytes>;

  struct SeparateGateStorage {
    SmemArray<ElementGate, RawStageLayout> raw;
    SmemArray<Element, RawStageLayout> out;
  };
  static constexpr bool SeparateGate = !std::is_same_v<ElementGate, Element>;
  using GateStorage = std::conditional_t<SeparateGate, SeparateGateStorage,
                                         SmemArray<Element, RawStageLayout>>;
  struct SharedStorage {
    SmemArray<Element, RawStageLayout>    smem_norm_q;
    SmemArray<Element, RawStageLayout>    smem_norm_k;
    GateStorage smem_g_cumsum;
    SmemArray<Element, SmemLayoutDecayed> smem_decayed;
    // k_inverse is the 2x-swizzled SQMMA B tile: 4 KiB data plus 4 KiB pad.
    // The union uses [0, 4 KiB) for k_inverse and [4 KiB, 8 KiB) for
    // k_restored; k_restored must stay at offset 4 KiB because the TMA store
    // is described by the k_inverse layout. The two physical pipeline pairs
    // defer reuse until the corresponding consumer phase has completed.
    //
    // The NxN inverse workspace (16x16 float matrix plus the solve scratch)
    // deliberately lives OUTSIDE this union. Aliasing it with smem_k_inverse
    // let a consumer warp that had already finished its P scatter run ahead
    // into the next chunk's scale pass and overwrite the matrix while warp 0
    // was still running the column solve, corrupting ~0.6% of the inverse
    // output. The extra 1 KiB still fits 6 CTAs per MP.
    union {
      SmemArray<Element, SmemLayoutKInverse> smem_k_inverse;
      struct {
        uint8_t                                                       dummy_pad[KInverseValidBytes];
        array_aligned<Element, kChunk * kHeadDim, SmemAlignmentBytes> smem_k_restored;
      };
    };
    typename CollectiveInverseNxN::SharedStorage smem_inverse_workspace;
    // 16x16 bf16 P tile staged by the four consumer warps; warp 3 reads it for
    // the A_rec = P @ inv wmma, so its reuse is ordered by KktFree.
    array_aligned<Element, kChunk * kChunk, 16> smem_p_stage;
    // The fold's A operand: inverse_beta staged as bf16 in the atom's own
    // layout (K = i contiguous).  +1 KiB.
    SmemArray<Element, SmemLayoutInverseBeta> smem_inverse_beta;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);
  // One raw stage keeps the struct at ~31.8 KiB (fold A operand included),
  // which lets five CTAs/MP fit the 192 KiB/MP budget.
  static_assert(SharedStorageSize <=
                    40 * 1024 + (SeparateGate ? sizeof(ElementGate) *
                                                    kRawTileElems * RawStages
                                              : 0),
                "shared memory must fit 5 CTAs per MP");

  MUTLASS_DEVICE static ElementGate *raw_gate_data(SharedStorage &s) {
    if constexpr (SeparateGate)
      return s.smem_g_cumsum.raw.data();
    else
      return s.smem_g_cumsum.data();
  }
  MUTLASS_DEVICE static Element *out_gate_data(SharedStorage &s) {
    if constexpr (SeparateGate)
      return s.smem_g_cumsum.out.data();
    else
      return s.smem_g_cumsum.data();
  }

  struct Params;

  struct Arguments {
    Element const*                      ptr_q;
    Element const *ptr_k;
    ElementGate const *ptr_g;
    ElementBeta const *ptr_beta;
    float const*                        ptr_A_log;
    float const*                        ptr_dt_bias;
    StrideQ                             stride_q;
    StrideK                             stride_k;
    StrideG                             stride_g;
    float                               scale;
    float                               lower_bound;
    typename Workspace::MutablePointers workspace;
  };

  struct Params {
    Element const*                      ptr_q;
    Element const *ptr_k;
    ElementGate const *ptr_g;
    ElementBeta const *ptr_beta;
    float const*                        ptr_A_log;
    float const*                        ptr_dt_bias;
    StrideQ                             stride_q;
    StrideK                             stride_k;
    StrideG                             stride_g;
    float                               scale;
    float                               lower_bound;
    typename Workspace::MutablePointers workspace;
    TMEWorkspaceDecayed                 tme_workspace_decayed;
    TMELoadRaw                          tme_q;
    TMELoadRaw tme_k;
    TMELoadGate tme_g;
  };

  // ---------------------------------------------------------------------
  // Hand-off rings.
  //
  // This line's mutlass carries no MP31 warp-specialised pipeline, so the four
  // hand-offs K1 needs are named-barrier rings, spelled the way the recurrence
  // collective spells its input rings: a producer waits its ring's empty pair
  // before it writes a stage and publishes the full pair after, and a consumer
  // waits the full pair and releases the empty one after its last read.
  //
  // Warp-role synchronization: the norm producers are warps 0/1/2 and the
  // consumers are warps 4-7; the decayed producers are warps 4-7 and the store
  // consumer is the last of them.  Norm is released after the fused pass to
  // overlap the next fill.
  using PrepareBarrier = PrepareBarriers<PipelineStages, RawStages>;
  using NormRing       = NamedBarrierRing<PipelineStages, PrepareBarrier::NormFull, PrepareBarrier::NormEmpty>;
  // The decayed tile has a single stage: the four consumer warps publish its
  // ready side and the TMA store waits it.  Reuse across chunks is ordered by
  // KktReady's rendezvous, so there is no empty side -- this is not a ring.
  struct DecayedReady {
    static MUTLASS_DEVICE constexpr int full(int sequence) {
      return PrepareBarrier::DecayedReady;
    }
    static MUTLASS_DEVICE constexpr uint32_t phase(int sequence) {
      return uint32_t(sequence & 1);
    }
    static MUTLASS_DEVICE void publish(int sequence) {
      arrive_warp(full(sequence));
    }
    static MUTLASS_DEVICE void wait(int sequence) {
      wait_warp(full(sequence), int(phase(sequence)));
    }
  };
  // A raw tile owns one transaction barrier: lane 0 arms it (the
  // arrive_and_expect_tx) and issues the copy, and every lane of that warp waits
  // the same barrier, so a warp normalizes as soon as *its own* tile has landed.
  // The stage it lands in is the norm stage, so the free side is NormEmpty --
  // the caller waits that before it issues (see the producers).
  template <int FullBase, class RawElement = Element> struct RawTmeRing {
    static MUTLASS_DEVICE constexpr int stage(int sequence) {
      return sequence % RawStages;
    }
    static MUTLASS_DEVICE constexpr int full(int sequence) {
      return FullBase + sequence % RawStages;
    }
    static MUTLASS_DEVICE constexpr uint32_t phase(int sequence) {
      return uint32_t((sequence / RawStages) & 1);
    }
    static MUTLASS_DEVICE void arm(int sequence) {
      named_barrier_arrive_and_expect_tx(
          full(sequence), uint32_t(kRawTileElems * sizeof(RawElement)));
    }
    static MUTLASS_DEVICE void wait(int sequence) {
      wait_warp(full(sequence), int(phase(sequence)));
    }
  };
  using RawQRing = RawTmeRing<PrepareBarrier::RawQFull>;
  using RawKRing = RawTmeRing<PrepareBarrier::RawKFull>;
  using RawGRing = RawTmeRing<PrepareBarrier::RawGFull, ElementGate>;

  struct BarrierStorage;

  struct Pipeline {
    // Sequences stand in for the mutlass pipeline states: each role advances its
    // own cursor once per chunk and the ring turns the cursor into a barrier
    // pair, an id and a phase.
    int norm_read;
    int norm_write;
    int decayed_read;
    int decayed_write;
    int load_q_read;
    int load_q_write;
    int load_k_read;
    int load_k_write;
    int load_g_read;
    int load_g_write;
    // Consumer-side release cursors for the raw stages, advanced once per chunk.
    // Phase of the KktReady handshake, tracked by the inverse warp (the only
    // consumer of that barrier): one completion per chunk.
    uint32_t inverse_ready_phase;
    // Phase of the InverseBetaReady handshake, tracked by the consumer squad
    // (its only waiters): the inverse warp arrives once the fold's A operand is
    // staged, which is what the fold SQMMA waits on.  One completion per chunk.
    uint32_t inverse_beta_ready_phase;
    // Phase of the RestoredFree handshake, tracked by the consumer squad (its
    // only waiters): the store warp arrives once the decayed TME store has drained.
    // Starts at 1 for the same reason as inverse_free_phase: chunk 0 has no
    // previous chunk's stores to wait for.
    uint32_t restored_free_phase;
    // Phase of the KktFree handshake, tracked by the KKT warps (its only
    // waiters).  Starts at 1: the barrier is initialised in phase 0, so the first
    // wait (phase 1) passes immediately - chunk 0 has no solve to wait for.
    // Starting at 0 deadlocks, because chunk 0's scatter would be waiting for a
    // publish that depends on that very scatter.
    uint32_t inverse_free_phase;

    MUTLASS_DEVICE explicit Pipeline(BarrierStorage* storage)
        : norm_read(0),
          norm_write(0),
          decayed_read(0),
          decayed_write(0),
          load_q_read(0),
          load_q_write(0),
          load_k_read(0),
          load_k_write(0),
          load_g_read(0),
          load_g_write(0),
          inverse_ready_phase(0),
          inverse_beta_ready_phase(0),
          inverse_free_phase(1),
          restored_free_phase(1) {
      (void)storage;
    }
  };

  struct alignas(1) BarrierStorage {
    // The four handshakes, then every ring pair; one byte per barrier id, so the
    // ids in PrepareBarriers are offsets into this array.
    uint8_t stage[PrepareBarrier::NumBarriers];
  };

  // The fold writes T directly to gmem, so it needs no extra shared-memory
  // rendezvous after the SQMMA.
  static MUTLASS_DEVICE void init_stage_barriers(BarrierStorage* barrier_storage, int tid) {
    (void)barrier_storage;
    if (tid == 0) {
      // All four consumer warps rendezvous over the staged P tile; the same
      // meeting orders reuse of that tile across chunks.
      named_barrier_init(PrepareBarrier::KktReady, 4);
      // All four writers arrive once each after their scatter; the inverse warp
      // consumes that phase before it reads the matrix.  It arrives once after
      // publishing the inverse and the writers wait for that before their next
      // scatter (the decayed release no longer orders it - see consumer_loop).
      named_barrier_init(PrepareBarrier::KktFree, 1);
      // The inverse warp arrives InverseBetaReady once the fold's A operand is
      // staged (one arrival); the consumer squad waits that phase before the
      // fold.  The fold writes T directly to gmem, so no extra shared-memory
      // rendezvous is needed after the SQMMA.
      named_barrier_init(PrepareBarrier::InverseBetaReady, 1);
      named_barrier_init(PrepareBarrier::RestoredFree, 1);

      // K1's hand-off rings.  The counts are the roles that arrive on them:
      //   norm    - the three producer warps publish, the four consumer warps release;
      //   decayed - the four consumer warps publish, the store warp releases;
      //   raw     - lane 0 of the owning producer warp arms the tile (one arrival;
      //             the TME engine owns the completion), the four consumer warps
      //             release.
      MUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < PipelineStages; ++j) {
        named_barrier_init(PrepareBarrier::NormFull + j, uint32_t(NumProducerWarps - 1));
        named_barrier_init(PrepareBarrier::NormEmpty + j, uint32_t(NumConsumerWarps));
        named_barrier_init(PrepareBarrier::DecayedReady + j, uint32_t(NumConsumerWarps));
      }
      MUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < RawStages; ++j) {
        // One arrival per raw tile: its producer warp's lane 0 arms it (the TME
        // engine owns the completion).  The free side is NormEmpty.
        named_barrier_init(PrepareBarrier::RawQFull + j, 1);
        named_barrier_init(PrepareBarrier::RawKFull + j, 1);
        named_barrier_init(PrepareBarrier::RawGFull + j, 1);
      }
    }
  }

  // The (B*T, Hqk|H, kHeadDim) view the three raw TME descriptors are built
  // from.  Only the token and head strides enter the descriptor; the batch is
  // folded into the outer mode, which is exact whenever the batch stride is the
  // token stride times T (true for every (B, T, H, D) view of a batch-major
  // buffer, including a q|k|v-interleaved one).  Anything else would need a
  // fourth descriptor mode; assert instead of silently aliasing batches.
  template <class ProblemSize, class Stride, class RawElement>
  MUTLASS_HOST_DEVICE static auto
  make_raw_gmem_tensor(ProblemSize const &problem_size, Stride const &stride,
                       RawElement const *ptr, int heads) {
    auto outer = get<2>(stride);
    assert(problem_size.B <= 1 || get<1>(outer) == int64_t(problem_size.T) * get<0>(stride));
    return make_tensor(make_gmem_ptr(ptr),
                       make_layout(make_shape(int(problem_size.B) * int(problem_size.T), heads, Int<kHeadDim>{}),
                                   make_stride(get<0>(stride), get<0>(outer), _1{})));
  }

  template <class ProblemSize>
  static Params to_underlying_arguments(ProblemSize const& problem_size, Arguments const& args) {
    int32_t records           = Components::workspace_records(problem_size);
    int32_t outer             = records * problem_size.H;
    auto    workspace_decayed = Workspace::make_decayed_tensor(args.workspace.decayed, outer);
    // (kChunk, kHeadDim, outer) row-major view of k_restored: same flat
    // offsets as Workspace::KRestoredLayout's (kHeadDim, kChunk, outer), with
    // the chunk-major record stride (not the 2048-element field size).
    auto gmem_q = make_raw_gmem_tensor(problem_size, args.stride_q, args.ptr_q, problem_size.Hqk);
    auto gmem_k = make_raw_gmem_tensor(problem_size, args.stride_k, args.ptr_k, problem_size.Hqk);
    auto gmem_g = make_raw_gmem_tensor(problem_size, args.stride_g, args.ptr_g, problem_size.H);
    return Params{
        args.ptr_q,
        args.ptr_k,
        args.ptr_g,
        args.ptr_beta,
        args.ptr_A_log,
        args.ptr_dt_bias,
        args.stride_q,
        args.stride_k,
        args.stride_g,
        args.scale,
        args.lower_bound,
        args.workspace,
        make_tme_copy(MP31_TME_STORE{}, workspace_decayed, take<0, 2>(SmemLayoutDecayed{})),
        make_tme_copy(MP31_TME_LOAD{}, gmem_q, SmemLayoutRaw{}),
        make_tme_copy(MP31_TME_LOAD{}, gmem_k, SmemLayoutRaw{}),
        make_tme_copy(MP31_TME_LOAD{}, gmem_g, SmemLayoutRaw{}),
    };
  }

  static MUTLASS_DEVICE float sigmoid_fast(float x) {
    return 1.0f / (1.0f + ::mate::flat::detail::fast_exp2(-x * kLog2E));
  }

  static MUTLASS_DEVICE ElementGate inactive_gate_raw_element() {
    if constexpr (HasGateParams) {
      return -std::numeric_limits<ElementGate>::infinity();
    } else {
      return ElementGate(GateIsLinear ? 1.f : 0.f);
    }
  }

  // simd::vrcp expresses this as a real <4 x float> division so the backend can
  // fold it into one AF32.RCP.BST4; see that helper for why four scalar
  // __musa_rcp_rn_f calls cannot reach the same combine.
  static MUTLASS_DEVICE float4 rcp_native(float4 const& x) {
    return simd::vrcp(x);
  }

  template <class TensorQK>
  static MUTLASS_DEVICE float4 load_qk_vec4(TensorQK&& sQk, int row, int col_base) {
    return simd::load_packed4_as_float4(&sQk(make_coord(row, col_base)));
  }

  template <class TensorQK>
  static MUTLASS_DEVICE void store_qk_vec4(TensorQK&& sQk, int row, int col_base, float4 value) {
    simd::store_float4_to_packed4_rn(&sQk(make_coord(row, col_base)), value);
  }

  // Reduce over SubWarpSize lanes.  MP31 has a native SHUFFLE.IDX
  // instruction, but the butterfly form (__shfl_xor_sync) is emulated
  // through a global-memory buffer; express the reduction as index shuffles
  // with an absolute source lane so it lowers to the native instruction.
  template <int SubWarpSize>
  static MUTLASS_DEVICE void subwarp_reduce_sum(float& value) {
    int lane = mutlass::canonical_lane_idx();
    MUTE_UNROLL
    for (int offset = SubWarpSize / 2; offset > 0; offset >>= 1) {
      value += __shfl_sync(uint32_t(-1), value, lane ^ offset, SubWarpSize);
    }
  }

  template <class ProblemSize, class WorkDesc>
  static MUTLASS_DEVICE float load_beta_register(Params const&      params,
                                                 ProblemSize const& problem_size,
                                                 WorkDesc const&    work_desc,
                                                 int                chunk_idx,
                                                 int                lane,
                                                 bool               is_not_full_chunk) {
    bool    token_valid = lane < kChunk && (!is_not_full_chunk || lane < work_desc.actual_len(chunk_idx));
    int64_t offset =
        (int64_t(work_desc.tme_batch()) * problem_size.T + int64_t(work_desc.tme_token(chunk_idx) + lane)) *
            problem_size.H +
        int64_t(work_desc.head_idx);
    float raw_beta =
        token_valid ? static_cast<float>(params.ptr_beta[offset]) : -std::numeric_limits<float>::infinity();
    if (!token_valid)
      return 0.f;
    if constexpr (UseBetaSigmoid)
      return (AllowNegEigval ? 2.f : 1.f) * sigmoid_fast(raw_beta);
    else
      return raw_beta;
  }

  // -------------------------------------------------------------------------
  // TME raw-tile loads (Q/K/G)
  // -------------------------------------------------------------------------

  // The folded (batch, token) coordinate of a chunk's first row in the
  // (B*T, Hqk, kHeadDim) view the raw descriptors were built from.
  template <class ProblemSize, class WorkDesc>
  MUTLASS_DEVICE static int raw_tokens(ProblemSize const& problem_size, WorkDesc const& work_desc, int chunk_idx) {
    return int(int64_t(work_desc.tme_batch()) * int64_t(problem_size.T) + work_desc.tme_token(chunk_idx));
  }

  // Issue one TMA box -- the whole (kChunk, kHeadDim) raw tile of a
  // (token, head) pair -- and wait for it.  Lane 0 waits the ring's empty pair
  // and arms the transaction barrier (that is the tile's arrive_and_expect_tx)
  // and issues the copy; every lane then phase-waits the same barrier, so the
  // whole warp is ordered against the TMA's smem writes and a warp's normalize
  // starts as soon as its own tile has landed.
  //
  // The box origin is an absolute coordinate, not a tile index: TMA tile mode
  // takes an arbitrary start row, so a chunk does not have to be 16-row
  // aligned in the folded (B*T) axis.  Out-of-range rows are zero-filled by
  // the hardware, which is exactly what the partial-chunk path wants: the
  // normalize pass zeroes the invalid rows anyway, and the cumsum reads the
  // inactive-gate sentinel instead of the loaded byte.
  template <class Ring, class TmeLoad, class RawElement>
  MUTLASS_DEVICE static void
  tme_load_raw_issue(TmeLoad const &tme_load, int tokens, int tokens_total,
                     int head, int heads, RawElement *s_raw, int sequence) {
    int lane = mutlass::canonical_lane_idx();
    // s_raw is already offset to this chunk's stage by the caller.
    // TME_LOAD_Unpack only ever reads the box origin out of the source
    // (`src.data().coord_`), and `copy(Copy_Atom, ...)` only dispatches when
    // both operands are rank-1 (otherwise it falls into the vectorising
    // per-mode loop) with a static size equal to the atom's box.  So: slice
    // the coordinate tensor down to the box origin, then re-label that origin
    // with a flat kChunk*kHeadDim layout.
    constexpr int kBoxElems = kChunk * kHeadDim;
    auto          m_all     = tme_load.get_tme_tensor(make_shape(tokens_total, heads, Int<kHeadDim>{}));
    auto          box_start = m_all(make_coord(tokens, head, _));
    auto          src       = make_tensor(box_start.data(), make_layout(Int<kBoxElems>{}));
    auto          dst       = make_tensor(make_smem_ptr(s_raw), make_layout(Int<kBoxElems>{}));
    // The caller has already waited the stage free (NormEmpty).
    if (lane == 0) {
      Ring::arm(sequence);
      copy(tme_load.with(named_barrier_id(Ring::full(sequence))), src, dst);
    }
  }

  // The owning warp waits its own tile; lanes that are not lane 0 wait through
  // wait_warp's rendezvous.
  template <class Ring>
  MUTLASS_DEVICE static void tme_load_raw_wait(int sequence) {
    Ring::wait(sequence);
  }

  // -------------------------------------------------------------------------
  // Producer squad (global warps 0-3)
  // -------------------------------------------------------------------------

  // Producer-squad warps 0/1: normalize the raw Q/K tile in smem in place.
  // The raw tile arrives through TME, so its tail rows are hardware-zero
  // rather than descriptor-safe garbage; either way the invalid rows are
  // masked out here.
  // Each pass covers RowsPerPass rows at once: a row is split over LanesPerRow
  // lanes and the warp's remaining lanes take the following rows.  Widening the
  // per-lane element count is what pays here, because the two per-row serial
  // costs - the shuffle reduction and the correctly-rounded rsqrt sequence -
  // are issued once per pass and shared by every row in it.  Eight lanes per
  // row puts four rows in flight and costs three shuffle levels instead of
  // five, which is roughly half the normalize instructions.
  static MUTLASS_DEVICE void normalize_qk(Element* s_norm_data, float scale, int actual_len, bool is_not_full_chunk) {
    constexpr int LanesPerRow = 8;
    constexpr int ElemPerLane = kHeadDim / LanesPerRow;                    // 16
    constexpr int VecPerLane  = ElemPerLane / VecSize;                     // 4
    constexpr int RowsPerPass = mutlass::NumThreadsPerWarp / LanesPerRow;  // 4
    int           lane        = mutlass::canonical_lane_idx();
    int           lane_in_row = lane % LanesPerRow;
    int           row_in_pass = lane / LanesPerRow;
    int           col_base    = lane_in_row * ElemPerLane;
    auto          sNorm       = make_tensor(make_smem_ptr(s_norm_data), NormLayout{});
    MUTLASS_PRAGMA_UNROLL
    for (int pass = 0; pass < kChunk / RowsPerPass; ++pass) {
      int    row = pass * RowsPerPass + row_in_pass;
      float4 v[VecPerLane];
      MUTE_UNROLL
      for (int i = 0; i < VecPerLane; ++i) {
        v[i] = load_qk_vec4(sNorm, row, col_base + i * VecSize);
      }
      float sum = 0.0f;
      if constexpr (NormalizeQK) {
        // Independent partial chains, then a short tree: dot4 is a serial
        // four-deep fma chain and four of them in a row would make the
        // reduction latency-bound instead of throughput-bound.
        float part[VecPerLane];
        MUTE_UNROLL
        for (int i = 0; i < VecPerLane; ++i) {
          part[i] = 0.0f;
          simd::dot4(v[i], v[i], part[i]);
        }
        MUTE_UNROLL
        for (int i = 0; i < VecPerLane; ++i) {
          sum += part[i];
        }
        subwarp_reduce_sum<LanesPerRow>(sum);
        sum = __musa_rsqrt_rn_f(sum + 1.0e-6f);
      } else {
        sum = 1.0f;
      }
      // Robust-copy tail rows hold garbage: zero them like the sync-load path.
      bool  valid  = !is_not_full_chunk || row < actual_len;
      float factor = sum * scale;
      MUTE_UNROLL
      for (int i = 0; i < VecPerLane; ++i) {
        store_qk_vec4(sNorm, row, col_base + i * VecSize, valid ? simd::vmul(v[i], factor) : simd::splat4(0.0f));
      }
    }
  }

  // Producer-squad warp 2: gate cumsum straight from gmem.  Each lane owns one column
  // strip and keeps the running prefix; the fp32 exp2(prefix) is published
  // to smem, the final value to workspace.total.
  template <class TensorG>
  MUTLASS_DEVICE static float4 load_gate_vec4(TensorG const &t, int row,
                                              int col) {
    if constexpr (!SeparateGate)
      return load_qk_vec4(t, row, col);
    else {
      auto raw = *reinterpret_cast<mutlass::Array<ElementGate, 4> const *>(
          &t(make_coord(row, col)));
      auto v = mutlass::NumericArrayConverter<float, ElementGate, 4>{}(raw);
      return make_float4(v[0], v[1], v[2], v[3]);
    }
  }

  template <class ProblemSize, class WorkDesc>
  static MUTLASS_DEVICE void
  compute_g_cumsum(SharedStorage &shared_storage, ElementGate *s_g_cumsum_raw,
                   Params const &params, ProblemSize const &problem_size,
                   WorkDesc const &work_desc, int chunk_idx) {
    auto sGCumsum = make_tensor(make_smem_ptr(s_g_cumsum_raw), CumsumLayout{});
    auto sGOut = make_tensor(make_smem_ptr(out_gate_data(shared_storage)),
                             CumsumLayout{});
    int   lane       = mutlass::canonical_lane_idx();
    int   col_base   = lane * VecSize;
    int   actual_len = work_desc.actual_len(chunk_idx);
    float a_log_exp  = 1.0f;
    if constexpr (HasGateParams) {
      if (lane == 0) {
        a_log_exp = ::mate::flat::detail::fast_exp2(params.ptr_A_log[work_desc.head_idx] * kLog2E);
      }
      a_log_exp = __shfl_sync(uint32_t(-1), a_log_exp, 0, mutlass::NumThreadsPerWarp);
    }
    float4 dt = simd::splat4(0.0f);
    if constexpr (HasGateParams) {
      dt = simd::load_float4(params.ptr_dt_bias + int64_t(work_desc.head_idx) * kHeadDim + col_base);
    }
    float  gate_scale = -a_log_exp * kLog2E;
    float  gate_coeff = params.lower_bound * kLog2E;
    float4 cumsum     = simd::splat4(0.0f);
    // The cumsum is updated in place, so the compiler keeps every row's load
    // behind the previous row's store; load a group of rows up front instead.
    constexpr int RowsPerGroup = 4;
    static_assert(kChunk % RowsPerGroup == 0);
    MUTE_UNROLL
    for (int row0 = 0; row0 < kChunk; row0 += RowsPerGroup) {
      float4 raw_g[RowsPerGroup];
      MUTE_UNROLL
      for (int i = 0; i < RowsPerGroup; ++i) {
        raw_g[i] = simd::splat4(static_cast<float>(inactive_gate_raw_element()));
        if (row0 + i < actual_len) {
          raw_g[i] = load_gate_vec4(sGCumsum, row0 + i, col_base);
        }
      }
      MUTE_UNROLL
      for (int i = 0; i < RowsPerGroup; ++i) {
        float4 gate = raw_g[i];
        if constexpr (HasGateParams) {
          float4 exp_arg = simd::vmul(simd::vadd(raw_g[i], dt), gate_scale);
          gate =
              simd::vmul(rcp_native(simd::vadd(simd::fast_exp2(exp_arg), 1.0f)),
                         gate_coeff);
        } else if constexpr (GateIsLinear) {
          gate = make_float4(::log2f(fmaxf(gate.x,1e-10f)), ::log2f(fmaxf(gate.y,1e-10f)), ::log2f(fmaxf(gate.z,1e-10f)),
                             ::log2f(fmaxf(gate.w,1e-10f)));
        } else if constexpr (GateIsLog)
          gate = simd::vmul(gate, kLog2E);
        cumsum = simd::vadd(cumsum, gate);
        store_qk_vec4(sGOut, row0 + i, col_base, simd::fast_exp2(cumsum));
      }
    }
    int    record = Workspace::record(problem_size, work_desc, chunk_idx);
    float4 total  = simd::fast_exp2(cumsum);
    // Two copies of the same decay product, and they exist for different readers.
    //
    // `total` is the Element row the MP32 recurrence and this kernel's folded
    // epilogue consume: one rounding to the workspace type here, right at the
    // chunk's own decay, which is what those paths have always produced.
    //
    // `decay` is the fp32 row the MP31 recurrence and MP31's backward scan use
    // to decay the running state.  A bf16 decay factor is not accurate enough
    // for that: one rounding is ~0.4 %, coarser than the factor itself when the
    // gate sits near 1, so every chunk multiplies the state by the same biased
    // value and the error grows linearly with the chunk count -- 4.7 % at 8192
    // tokens, 19.5 % at 32768.  fp32 keeps it at the summation error (~4e-4 %).
    using TotalVec = mutlass::Array<float, 4>;
    mutlass::NumericArrayConverter<Element, float, 4, mutlass::FloatRoundStyle::round_to_nearest> round_total;
    *reinterpret_cast<mutlass::Array<Element, 4>*>(params.workspace.total + int64_t(record) * Workspace::RecordStride +
                                                   col_base) = round_total(*reinterpret_cast<TotalVec const*>(&total));
    *reinterpret_cast<float4*>(Workspace::decay_row(params.workspace.total, record) + col_base) = total;
  }

  // Consumer-squad store warp: TME store of the decayed tile, issued as soon
  // as all four consumer warps have published it - long before the fold, so
  // the transfer overlaps the KKT+P gemm and the whole epilogue.
  template <class ProblemSize, class WorkDesc>
  static MUTLASS_DEVICE void store_decayed_tile(SharedStorage&     shared_storage,
                                                Params const&      params,
                                                ProblemSize const& problem_size,
                                                WorkDesc const&    work_desc,
                                                int                chunk_idx,
                                                int                outer) {
    if (mutlass::canonical_lane_idx() != 0) {
      return;
    }
    int  record    = Workspace::record(problem_size, work_desc, chunk_idx);
    auto g_decayed = params.tme_workspace_decayed.get_tme_tensor(make_shape(Int<2 * kChunk>{}, Int<kHeadDim>{}, outer));
    auto s_decayed = make_tensor(make_smem_ptr(shared_storage.smem_decayed.data()), SmemLayoutDecayed{});
    auto thr_copy_s2g = params.tme_workspace_decayed.get_slice(_0{});
    auto bSG_sDecayed = group_modes<0, 3>(thr_copy_s2g.partition_S(s_decayed));
    auto bSG_gDecayed = group_modes<0, 3>(thr_copy_s2g.partition_D(g_decayed));
    copy(params.tme_workspace_decayed, bSG_sDecayed(_, 0), bSG_gDecayed(_, int32_t(record)));

    tme_store_arrive();
  }

  // -------------------------------------------------------------------------
  // Consumer warps
  // -------------------------------------------------------------------------

  // All four consumer warps: gate the producer's normalized tiles and
  // produce decayed + k_inverse in the SQMMA operand layouts, in one fused
  // pass over the rows (each gate row is loaded once and applied to both
  // its Q and K rows).  k_restored = k_inverse * total is computed here
  // too: total is the last row of the gate cumsum tile, so no separate
  // g_total buffer is needed.  No validity checks: the producers already
  // publish zeros for inactive norm rows and zero gates for inactive
  // cumsum rows; the reciprocal epsilon keeps 0 * rcp(0) finite.
  template <class TensorK,
            class TensorQ,
            class TensorG,
            class TensorDecayed,
            class TensorKInverse,
            class TensorKRestored>
  static MUTLASS_DEVICE void scale_normalized_qk(TensorK&&         sK,
                                                 TensorQ&&         sQ,
                                                 TensorG&&         sGCumsum,
                                                 TensorDecayed&&   sDecayed,
                                                 TensorKInverse&&  sKInverse,
                                                 TensorKRestored&& sKRestored,
                                                 int               local_tid) {
    constexpr int   ThreadsPerRow     = 32;
    constexpr int   RowsPerWarp       = mutlass::NumThreadsPerWarp / ThreadsPerRow;
    constexpr int   ElementsPerThread = kHeadDim / ThreadsPerRow;
    constexpr int   VectorsPerThread  = ElementsPerThread / VecSize;
    constexpr int   RowsPerWave       = NumConsumerWarps * RowsPerWarp;
    constexpr float kReciprocalEps    = 1.0e-38f;
    static_assert(kChunk % RowsPerWave == 0);
    int warp_idx      = mutlass::canonical_warp_idx() - ConsumerWarpBase;
    int lane_idx      = mutlass::canonical_lane_idx();
    int row_in_warp   = lane_idx / ThreadsPerRow;
    int lane_in_row   = lane_idx % ThreadsPerRow;
    int thread_col_lo = lane_in_row * ElementsPerThread;

    constexpr int RowsPerThread = kChunk / RowsPerWave;

    // g_total = exp2(final cumsum) is the same for every row of the chunk, so
    // it is read once per thread instead of once per row.
    float4 g_total[VectorsPerThread];
    MUTE_UNROLL
    for (int vec_iter = 0; vec_iter < VectorsPerThread; ++vec_iter) {
      int vec_slot      = (vec_iter + row_in_warp) % VectorsPerThread;
      g_total[vec_iter] = load_qk_vec4(sGCumsum, kChunk - 1, thread_col_lo + vec_slot * VecSize);
    }

    // Issue every row's loads before the first store: the compiler cannot
    // prove the source and destination smem tiles disjoint, so a per-row
    // load/store loop serializes one smem round trip per row.
    float4 q_in[RowsPerThread][VectorsPerThread];
    float4 k_in[RowsPerThread][VectorsPerThread];
    float4 gate_in[RowsPerThread][VectorsPerThread];
    MUTE_UNROLL
    for (int r = 0; r < RowsPerThread; ++r) {
      int row = r * RowsPerWave + warp_idx * RowsPerWarp + row_in_warp;
      MUTE_UNROLL
      for (int vec_iter = 0; vec_iter < VectorsPerThread; ++vec_iter) {
        int col_base         = thread_col_lo + ((vec_iter + row_in_warp) % VectorsPerThread) * VecSize;
        q_in[r][vec_iter]    = load_qk_vec4(sQ, row, col_base);
        k_in[r][vec_iter]    = load_qk_vec4(sK, row, col_base);
        gate_in[r][vec_iter] = load_qk_vec4(sGCumsum, row, col_base);
      }
    }

    MUTE_UNROLL
    for (int r = 0; r < RowsPerThread; ++r) {
      int row = r * RowsPerWave + warp_idx * RowsPerWarp + row_in_warp;
      // Row layout of the 32-row decayed operand tile, i.e. the K1/K2
      // workspace contract for this architecture.
      //   MP31: blocked by row class - K 0-15 in physical rows 0..15 and the
      //     matching Q 0-15 in physical rows 16..31.  Because the MP31 32x32
      //     C ownership pairs physical rows r/r+16 inside one warp, every warp
      //     then owns four KKT rows (tCrAcc[0..3]) and four P rows (tCrAcc[4..7]);
      //     see compute_inverse_and_p for the re-derived scatter.
      //   MP32 (M32N32, C rows [0,16) to warps 0/1, [16,32) to warps 2/3):
      //     Q 0-15 in the top half and K 0-15 in the bottom half, which gives
      //     warps 0/1 the P rows and warps 2/3 the KKT rows.
      int q_row = row + kChunk;
      int k_row = row;
      MUTE_UNROLL
      for (int vec_iter = 0; vec_iter < VectorsPerThread; ++vec_iter) {
        int vec_slot = (vec_iter + row_in_warp) % VectorsPerThread;
        int col_base = thread_col_lo + vec_slot * VecSize;

        float4 q    = q_in[r][vec_iter];
        float4 k    = k_in[r][vec_iter];
        float4 gate = gate_in[r][vec_iter];

        // 0 * rcp(0 + eps) stays finite for the inactive rows.
        float4 k_inverse  = simd::vmul(k, rcp_native(simd::vadd(gate, simd::splat4(kReciprocalEps))));
        float4 k_restored = simd::vmul(k_inverse, g_total[vec_iter]);

        store_qk_vec4(sDecayed, q_row, col_base, simd::vmul(q, gate));
        store_qk_vec4(sDecayed, k_row, col_base, simd::vmul(k, gate));
        store_qk_vec4(sKInverse, row, col_base, k_inverse);
        // k_restored is only ever read by the fold, so it is staged directly in
        // the fold's B layout (head axis contiguous), NOT in the workspace tile
        // frame - T is what reaches gmem, written back after the fold.  It is
        // staged in NATURAL head order: a thread's four consecutive columns are
        // address-contiguous there (verified against the layout for every
        // lane/row), so the store vectorizes exactly like the decayed writes.
        // The workspace's LogicalToSlot permutation moves to the fold's T
        // writeback, which is scalar there either way.
        store_qk_vec4(sKRestored, col_base, row, k_restored);
      }
    }
  }

  // All four consumer warps: the chunk epilogue - scatter the KKT matrix and
  // stage P, hand both to warp 3 (KktReady), and once warp 3 has staged the
  // solved inverse into the fold's A operand (InverseBetaReady), fold it into
  // k_restored (T = inverse_beta^T * k_restored) with one 32x128x16 SQMMA.
  // Warp 3's solve, its A_rec wmma and the inverse publish all overlap the
  // consumer squad's fold and the next chunk's scale.  Under the blocked decayed row layout
  // (K 0-15 in physical rows 0..15, Q 0-15 in physical rows 16..31) the MP31
  // 32x32 C ownership gives every lane tCrAcc[0..3] on one of the K rows
  // (M = l/8 + 4*warp) and tCrAcc[4..7] on the matching Q row (M = l/8 + 4*warp +
  // 16), so each warp owns four KKT rows and four P rows and every warp writes
  // into both matrices.
  template <class ProblemSize, class WorkDesc>
  MUTLASS_DEVICE static void compute_inverse_and_p(SharedStorage&     shared_storage,
                                                   Params const&      params,
                                                   ProblemSize const& problem_size,
                                                   WorkDesc const&    work_desc,
                                                   int                chunk_idx,
                                                   int                local_tid,
                                                   uint32_t&          inverse_beta_ready_phase,
                                                   uint32_t&          inverse_free_phase) {
    auto sDecayed  = make_tensor(make_smem_ptr(shared_storage.smem_decayed.data()), SmemLayoutDecayed{})(_, _, 0);
    auto sKInverse = make_tensor(make_smem_ptr(shared_storage.smem_k_inverse.data()), SmemLayoutKInverse{})(_, _, 0);

    TiledMmaQK tiled_mma;
    auto       thr_mma = tiled_mma.get_thread_slice(local_tid);
    auto       tCrAcc  = partition_fragment_C(tiled_mma, make_shape(Int<2 * kChunk>{}, Int<2 * kChunk>{}));
    static_assert(decltype(size(tCrAcc))::value == 8);
    clear(tCrAcc);

    auto tCsDecayed  = thr_mma.partition_A(sDecayed);
    auto tCsKInverse = thr_mma.partition_B(sKInverse);
    auto tCrDecayed  = thr_mma.make_fragment_A(tCsDecayed);
    auto tCrKInverse = thr_mma.make_fragment_B(tCsKInverse);

    // Only K-inverse rows 0..15 are initialized.  They produce the valid C
    // columns 0..15; columns 16..31 are intentionally computed and discarded.
    gemm(tiled_mma, tCrAcc, tCrDecayed, tCrKInverse, tCrAcc);
    mate::warpsquad_commit_batch();
    mate::warpsquad_wait<0>();

    int lane = local_tid % mutlass::NumThreadsPerWarp;

    auto matrix = make_tensor(make_smem_ptr(shared_storage.smem_inverse_workspace.matrix),
                              typename CollectiveInverseNxN::SmemLayout{});
    auto cC     = make_identity_tensor(make_shape(Int<2 * kChunk>{}, Int<2 * kChunk>{}));
    auto tCc    = thr_mma.partition_C(cC);
    // MP31 32x32 C ownership pairs physical rows r and r+16 inside one warp.
    // With the blocked decayed layout those two rows are one K row and the
    // matching Q row, so every lane holds tCrAcc[0..3] on a KKT row and tCrAcc[4..7]
    // on a P row.  tCrAcc[0]/tCrAcc[1] (tCrAcc[4]/tCrAcc[5]) are the two valid columns
    // l%8 and l%8+8; tCrAcc[2..3] (tCrAcc[6..7]) are columns >= 16 and are dropped.
    // Scatter plan (identity tensor dump of partition_C):
    //   warp w, lane l: KKT row = l/8 + 4w (0..15), P row = l/8 + 4w (0..15),
    //   columns l%8 and l%8+8.  Warp w therefore covers rows 4w..4w+3 of both
    //   16x16 tiles, l/8 selects the row inside the warp and l%8 the column
    //   pair, so each tile is written exactly once across all 128 lanes.
    {
      // The matrix is single-buffered and the inverse warp reads it until it
      // publishes; the decayed release no longer orders that (it is a consumer
      // squad handshake now), so every writer closes it explicitly here.  Placed
      // after the gemm above and far below the decayed producer_commit in
      // consumer_loop, which is what keeps the store warp's full barrier
      // satisfiable while the warps block.
      named_barrier_wait(PrepareBarrier::KktFree, inverse_free_phase);
      inverse_free_phase ^= 1u;

      int k_row  = int(get<0>(tCc(0)));
      int col_lo = int(get<1>(tCc(0)));
      int col_hi = int(get<1>(tCc(1)));
      // The writers commit the bare KKT product: beta is no longer folded in
      // here.  The solve warp stages beta on the matrix diagonal right before
      // the substitution (see CollectiveLowerTriangularInverseNxN), which takes
      // the beta global load, its sigmoid and the per-element beta shuffle out
      // of this scatter's dependency chain.  The diagonal and the upper triangle
      // are dead - the substitution reads the strictly lower part - but the
      // float4 reads that span them have to stay finite, so they are zeroed.
      matrix(make_coord(k_row, col_lo)) = k_row > col_lo ? tCrAcc(0) : 0.0f;
      matrix(make_coord(k_row, col_hi)) = k_row > col_hi ? tCrAcc(1) : 0.0f;

      // The solve itself now runs on the inverse warp (producer_inverse_loop).
      // All four writers rendezvous, then hand the matrix over and wait for
      // KktFree before their next scatter (see above).
    }

    {
      // Staging P in smem lets warp 3 read whole rows for its A_rec = P @ inv
      // wmma; the p field is entirely warp 3's job now, so there is no
      // consumer-side drain at all.  P rows come from tCrAcc[4..7] (Q rows
      // physical 16..31).
      auto    sP       = make_tensor(make_smem_ptr(shared_storage.smem_p_stage.data()), PStagingLayout{});
      int     p_row    = int(get<0>(tCc(4))) - kChunk;
      int     p_col_lo = int(get<1>(tCc(4)));
      int     p_col_hi = int(get<1>(tCc(5)));
      float4  values   = *reinterpret_cast<float4 const*>(&tCrAcc(4));
      Element packed[VecSize];
      simd::store_float4_to_packed4_rn(packed, values);
      // No clearing pass is needed (and one is actively harmful): the fragment
      // covers every one of the 256 cells, writing the explicit ternary zero
      // for the upper triangle, and the four warps land in each other's slices.
      // A "clear my own slice first" pass therefore races with another warp's
      // scatter and can wipe values that were just written.
      // Rows past actual_len are already zero out of normalize_qk, so the QK product is
      // zero there; only the causal mask is needed here.
      sP(make_coord(p_row, p_col_lo)) = p_row >= p_col_lo ? packed[0] : Element(0.0f);
      sP(make_coord(p_row, p_col_hi)) = p_row >= p_col_hi ? packed[1] : Element(0.0f);

      // Hand the matrix and the staged P tile to warp 3: the rendezvous
      // completes only when all four warps have arrived, so the solve sees
      // every writer's scatter.  The squad then waits for warp 3 to have the
      // fold's A operand staged.
      __syncwarp();
      if (lane == 0) {
        named_barrier_arrive_and_wait(PrepareBarrier::KktReady);
      }
      __syncwarp();
      named_barrier_wait(PrepareBarrier::InverseBetaReady, inverse_beta_ready_phase);
      inverse_beta_ready_phase ^= 1u;
    }

    // ---- the fold: T = inverse_beta^T * k_restored on the SQMMA --------------
    // One 16x128x16 tile in one 32x128x16 atom: M = the chunk token, N = the
    // head axis (all 128 in one atom), K = the contraction.  C's coordinates are
    // (chunk token, workspace slot), so it lands in the tile directly.
    {
      auto sInverseBeta = make_tensor(make_smem_ptr(shared_storage.smem_inverse_beta.data()), SmemLayoutInverseBeta{});
      auto sKRestore    = make_tensor(make_smem_ptr(shared_storage.smem_k_restored.data()), SmemLayoutKRestore{});
      TiledMmaKRInv tiled_mma_kr_inv;
      auto          thr_mma_kr_inv = tiled_mma_kr_inv.get_thread_slice(local_tid);
      auto          tCrA           = thr_mma_kr_inv.make_fragment_A(thr_mma_kr_inv.partition_A(sInverseBeta));
      auto          tCrB           = thr_mma_kr_inv.make_fragment_B(thr_mma_kr_inv.partition_B(sKRestore));
      // Backward-facing copy of the *unfolded* k_restored, written in the slot
      // frame K4 reads (slot = 8*(d & 15) + (d >> 4)); the fold below overwrites
      // this smem tile with T, so this is the only place the unfolded value is
      // still available.
      {
        int32_t const rec_b = int32_t(Workspace::record(problem_size, work_desc, chunk_idx));
        Element*      gKB   = params.workspace.k_restored_backward + int64_t(rec_b) * Workspace::RecordStride;
        // Eight consecutive slots per 16 B store: slot0 = 8*(d & 15) is a
        // multiple of eight, so slots slot0..slot0+7 map back to the head
        // columns d = 16*j + slot0/8 (j = 0..7).  The gather is smem-only; the
        // gmem side is two uint4 stores per thread instead of 16 scalar ones.
        MUTLASS_PRAGMA_UNROLL
        for (int r = 0; r < 2; ++r) {
          int const t     = int(local_tid) / 16 + r * 8;
          int const slot0 = (int(local_tid) % 16) * 8;
          Element   v[8];
          MUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < 8; ++j) v[j] = static_cast<Element>(sKRestore(make_coord(16 * j + (slot0 >> 3), t)));
          *reinterpret_cast<uint4*>(gKB + t * kHeadDim + slot0) = *reinterpret_cast<uint4 const*>(v);
        }
      }
      auto tCrAccKRInv = partition_fragment_C(tiled_mma_kr_inv, make_shape(Int<kNPadKRInv>{}, Int<kHeadDim>{}));
      static_assert(decltype(size(tCrAccKRInv))::value ==
                    kNPadKRInv * kHeadDim / (NumConsumerWarps * mutlass::NumThreadsPerWarp));
      auto cKRInv   = make_identity_tensor(make_shape(Int<kNPadKRInv>{}, Int<kHeadDim>{}));
      auto tCcKRInv = thr_mma_kr_inv.partition_C(cKRInv);
      clear(tCrAccKRInv);
      gemm(tiled_mma_kr_inv, tCrAccKRInv, tCrA, tCrB, tCrAccKRInv);
      mate::warpsquad_commit_batch();
      mate::warpsquad_wait<0>();
      // C's columns are NATURAL head indices (the B tile is staged in natural
      // order), and the workspace's LogicalToSlot permutation is applied at
      // the gmem writeback.  With C's ownership
      // (probed) m = (t/8)%4 + 4*(t/32), n = (t%8) + 8*v0 + 16*v1 in the tiled
      // MMA's own thread index t = local_tid, and slot(n) = 8*(n%16) + n/16 =
      // 8*t0 + 64*v0 + v1, so each thread's sixteen valid values land in TWO
      // runs of eight consecutive slots, and the writeback is two packed
      // 16-byte gmem stores per thread.  That is the PermuteN contract doing
      // double duty: the frame that gives K2's state lane 16 consecutive slots
      // also gives this writeback 8+8.
      {
        // The tiled MMA's thread index is local_tid (0..127), NOT the lane:
        // t = t0 + 8*t1 + 32*t2, m = t1 + 4*t2, n = t0 + 8*v0 + 16*v1, and
        // slot(n) = 8*(n%16) + n/16 = 8*t0 + 64*v0 + v1.
        int const m     = ((local_tid >> 3) & 3) + 4 * (local_tid >> 5);
        int const slot0 = 8 * (local_tid & 7);
        // tCrAcc(2*v1) / tCrAcc(2*v1+1) are (v0=0) / (v0=1): run A covers slots
        // slot0..slot0+7, run B slots slot0+64..slot0+71.
        alignas(16) Element run_a[kChunk / 2];
        alignas(16) Element run_b[kChunk / 2];
        MUTLASS_PRAGMA_UNROLL
        for (int v1 = 0; v1 < kChunk / 2; ++v1) {
          run_a[v1] = Element(tCrAccKRInv(2 * v1));
          run_b[v1] = Element(tCrAccKRInv(2 * v1 + 1));
        }
        // Plain 64-bit pointer arithmetic, NOT a mute layout: the workspace is
        // chunk-major, so a long sequence's record count times the 18688 B record
        // stride is larger than 2 GiB, and a 32-bit byte offset wraps there.
        // Records past the wrap were silently never written (flat record 114913
        // on a 16224-token/128-head case), and K2 then read the garbage.
        int64_t const record = int64_t(Workspace::record(problem_size, work_desc, chunk_idx));
        Element*      gT     = params.workspace.k_restored + record * int64_t(Workspace::RecordStride);
        *reinterpret_cast<uint4*>(gT + int64_t(m) * kHeadDim + slot0)      = *reinterpret_cast<uint4 const*>(run_a);
        *reinterpret_cast<uint4*>(gT + int64_t(m) * kHeadDim + slot0 + 64) = *reinterpret_cast<uint4 const*>(run_b);
      }
    }
  }

  // Consumer warps: fused pass + gemm + epilogue for every chunk.
  template <class ProblemSize, class WorkDesc>
  MUTLASS_DEVICE static void consumer_loop(SharedStorage&     shared_storage,
                                           Params const&      params,
                                           ProblemSize const& problem_size,
                                           WorkDesc const&    work_desc,
                                           Pipeline&          pipeline,
                                           int                thread_idx,
                                           int                chunk_idx) {
    int lane = mutlass::canonical_lane_idx();

    // The raw stage in flight: the producer's write cursor and the consumers'
    // release cursor advance over the same RawStages ring, one step per chunk.
    int const raw_stage = 0;  // RawStages == 1: the raw tile is the norm tile
    auto sGCumsum = make_tensor(make_smem_ptr(out_gate_data(shared_storage) +
                                              raw_stage * kRawTileElems),
                                CumsumLayout{});
    auto sNormQ =
        make_tensor(make_smem_ptr(shared_storage.smem_norm_q.data() + raw_stage * kRawTileElems), NormLayout{});
    auto sNormK =
        make_tensor(make_smem_ptr(shared_storage.smem_norm_k.data() + raw_stage * kRawTileElems), NormLayout{});
    auto sDecayed  = make_tensor(make_smem_ptr(shared_storage.smem_decayed.data()), SmemLayoutDecayed{})(_, _, 0);
    auto sKInverse = make_tensor(make_smem_ptr(shared_storage.smem_k_inverse.data()), SmemLayoutKInverse{})(_, _, 0);
    // The scale pass stages k_restored in the fold's B layout; the fold writes
    // T directly to the workspace gmem frame.
    auto sKRestored = make_tensor(make_smem_ptr(shared_storage.smem_k_restored.data()), SmemLayoutKRestore{});

    {
      // The decayed tile is reusable once the store warp's TMA has drained;
      // k_restored is consumed by the fold and written directly to gmem.
      named_barrier_wait(PrepareBarrier::RestoredFree, pipeline.restored_free_phase);
      pipeline.restored_free_phase ^= 1u;
      NormRing::wait(pipeline.norm_read);
      scale_normalized_qk(sNormK, sNormQ, sGCumsum, sDecayed, sKInverse, sKRestored, thread_idx);

      // The fused pass is done reading the norm tiles, so release them and
      // publish the decayed tile.  That release is also what frees the raw
      // tiles: they ARE the norm tiles, normalized in place, so the producer's
      // NormRing::wait_empty covers the raw stage as well.
      NormRing::release(pipeline.norm_read);
      DecayedReady::publish(pipeline.decayed_write);
      ++pipeline.norm_read;
      ++pipeline.decayed_write;

      // The consumer squad's store warp issues the decayed TMA store as soon
      // as all four warps have published the tile, then runs the epilogue
      // while the TMA runs.  The fold writes k_restored directly to gmem; only
      // the decayed TME transaction needs to drain before reuse.
      int warp_idx_c = mutlass::canonical_warp_idx() - ConsumerWarpBase;
      if (warp_idx_c == StoreConsumerWarp) {
        DecayedReady::wait(pipeline.decayed_read);
        int32_t records = Components::workspace_records(problem_size);
        store_decayed_tile(shared_storage, params, problem_size, work_desc, chunk_idx, records * problem_size.H);
      }

      compute_inverse_and_p(shared_storage,
                            params,
                            problem_size,
                            work_desc,
                            chunk_idx,
                            thread_idx,
                            pipeline.inverse_beta_ready_phase,
                            pipeline.inverse_free_phase);

      if (warp_idx_c == StoreConsumerWarp) {
        __syncwarp();
        if (lane == 0) {
          tme_store_wait();
          named_barrier_arrive(PrepareBarrier::RestoredFree);
        }
        __syncwarp();
        ++pipeline.decayed_read;
      }
    }
  }

  // Producer warp 0: TME-load Q, then normalize it in place.
  template <class ProblemSize, class WorkDesc>
  MUTLASS_DEVICE static void producer_normalize_q_loop(SharedStorage&     shared_storage,
                                                       Params const&      params,
                                                       ProblemSize const& problem_size,
                                                       WorkDesc const&    work_desc,
                                                       Pipeline&          pipeline,
                                                       int                thread_idx,
                                                       int                chunk_idx) {
    (void)thread_idx;
    {
      int  actual_len        = work_desc.actual_len(chunk_idx);
      bool is_not_full_chunk = actual_len < kChunk;
      NormRing::wait_empty(pipeline.norm_write);
      Element* s_raw = shared_storage.smem_norm_q.data() + RawQRing::stage(pipeline.load_q_write) * kRawTileElems;
      tme_load_raw_issue<RawQRing>(params.tme_q,
                                   raw_tokens(problem_size, work_desc, chunk_idx),
                                   int(problem_size.B) * int(problem_size.T),
                                   work_desc.qk_head_idx,
                                   problem_size.Hqk,
                                   s_raw,
                                   pipeline.load_q_write);
      tme_load_raw_wait<RawQRing>(pipeline.load_q_read);

      normalize_qk(s_raw, params.scale, actual_len, is_not_full_chunk);

      // Publish the norm tile once this warp's fill is visible to the others.
      __syncwarp();
      NormRing::publish(pipeline.norm_write);
      __syncwarp();
      ++pipeline.load_q_write;
      ++pipeline.load_q_read;
      ++pipeline.norm_write;
    }
  }

  // Producer warp 1: TME-load K, then normalize it in place.
  template <class ProblemSize, class WorkDesc>
  MUTLASS_DEVICE static void producer_normalize_k_loop(SharedStorage&     shared_storage,
                                                       Params const&      params,
                                                       ProblemSize const& problem_size,
                                                       WorkDesc const&    work_desc,
                                                       Pipeline&          pipeline,
                                                       int                thread_idx,
                                                       int                chunk_idx) {
    (void)thread_idx;
    {
      int  actual_len        = work_desc.actual_len(chunk_idx);
      bool is_not_full_chunk = actual_len < kChunk;
      NormRing::wait_empty(pipeline.norm_write);
      Element* s_raw = shared_storage.smem_norm_k.data() + RawKRing::stage(pipeline.load_k_write) * kRawTileElems;
      tme_load_raw_issue<RawKRing>(params.tme_k,
                                   raw_tokens(problem_size, work_desc, chunk_idx),
                                   int(problem_size.B) * int(problem_size.T),
                                   work_desc.qk_head_idx,
                                   problem_size.Hqk,
                                   s_raw,
                                   pipeline.load_k_write);
      tme_load_raw_wait<RawKRing>(pipeline.load_k_read);

      normalize_qk(s_raw, 1.0f, actual_len, is_not_full_chunk);

      // Publish the norm tile once this warp's fill is visible to the others.
      __syncwarp();
      NormRing::publish(pipeline.norm_write);
      __syncwarp();
      ++pipeline.load_k_write;
      ++pipeline.load_k_read;
      ++pipeline.norm_write;
    }
  }

  // Producer warp: gate cumsum for every chunk.
  template <class ProblemSize, class WorkDesc>
  MUTLASS_DEVICE static void producer_gate_loop(SharedStorage&     shared_storage,
                                                Params const&      params,
                                                ProblemSize const& problem_size,
                                                WorkDesc const&    work_desc,
                                                Pipeline&          pipeline,
                                                int                thread_idx,
                                                int                chunk_idx) {
    (void)thread_idx;
    {
      NormRing::wait_empty(pipeline.norm_write);
      ElementGate *s_raw =
          raw_gate_data(shared_storage) +
          RawGRing::stage(pipeline.load_g_write) * kRawTileElems;
      tme_load_raw_issue<RawGRing>(params.tme_g,
                                   raw_tokens(problem_size, work_desc, chunk_idx),
                                   int(problem_size.B) * int(problem_size.T),
                                   work_desc.head_idx,
                                   problem_size.H,
                                   s_raw,
                                   pipeline.load_g_write);
      tme_load_raw_wait<RawGRing>(pipeline.load_g_read);

      compute_g_cumsum(shared_storage, s_raw, params, problem_size, work_desc, chunk_idx);

      // Publish the norm tile once this warp's fill is visible to the others.
      __syncwarp();
      NormRing::publish(pipeline.norm_write);
      __syncwarp();
      ++pipeline.load_g_write;
      ++pipeline.load_g_read;
      ++pipeline.norm_write;
    }
  }

  // Producer squad warp 3 (global warp 3): the KKT inverse solve, the fold's
  // A-operand staging, the P-side epilogue (A_rec = P @ inv on ONE warp's
  // wmma) and the inverse publish - and nothing else.  The consumer squad
  // hands the KKT matrix and the staged P tile over through KktReady; this
  // warp solves, stages inverse_beta for the consumer squad's fold
  // (InverseBetaReady), then computes A_rec while the squad folds, and hands
  // the single-buffered smem back with KktFree.
  template <class ProblemSize, class WorkDesc>
  MUTLASS_DEVICE static void producer_inverse_loop(SharedStorage&     shared_storage,
                                                   Params const&      params,
                                                   ProblemSize const& problem_size,
                                                   WorkDesc const&    work_desc,
                                                   Pipeline&          pipeline,
                                                   int                thread_idx,
                                                   int                chunk_idx) {
    (void)thread_idx;
    int lane = mutlass::canonical_lane_idx();
    {
      bool is_not_full_chunk = work_desc.actual_len(chunk_idx) < kChunk;
      // Issued before the handshake wait: the wait below then covers the global
      // load's latency, and this warp's own lane indexing is the one the solve's
      // row scaling wants (lane l holds beta of token l for l < kChunk).
      float beta_lane = load_beta_register(params, problem_size, work_desc, chunk_idx, lane, is_not_full_chunk);

      CollectiveInverseNxN inverse_nxn;
      int64_t              record      = Workspace::record(problem_size, work_desc, chunk_idx);
      Element*             gP          = params.workspace.p + record * Workspace::RecordStride;
      Element*             gInverseOut = params.workspace.inverse + record * Workspace::RecordStride;
      // The hand-off wait lives here, not inside the collective: all four
      // consumer warps arrive KktReady once each after their scatter, and every
      // lane of this warp waits that phase (a phase test is non-destructive,
      // exactly like the pipeline's full-barrier waits), so no extra __syncwarp
      // is needed before the lanes that read the matrix.  solve_columns then
      // stages this lane's beta on its own diagonal before the substitution.
      named_barrier_wait(PrepareBarrier::KktReady, pipeline.inverse_ready_phase);
      pipeline.inverse_ready_phase ^= 1u;
      // lane, not the block-wide tid: only lanes [0, 16) are in the solve.
      inverse_nxn.solve_columns(shared_storage.smem_inverse_workspace, beta_lane, lane);

      // Only this warp has the finished inverse in its own write order, so it
      // stages the whole 16x16 tile itself (256 elements over 32 lanes); the
      // InverseBetaReady arrival below publishes it to the consumer squad's
      // fold.
      {
        auto sInverseBeta =
            make_tensor(make_smem_ptr(shared_storage.smem_inverse_beta.data()), SmemLayoutInverseBeta{});
        float const*  sol         = shared_storage.smem_inverse_workspace.solution;
        constexpr int kStageIters = kChunk * kChunk / mutlass::NumThreadsPerWarp;
        // Read the whole tile before the first store so the loads are not
        // serialized behind stores the compiler cannot disambiguate.
        float staged[kStageIters];
        MUTLASS_PRAGMA_UNROLL
        for (int t = 0; t < kStageIters; ++t) {
          int const e = lane + t * mutlass::NumThreadsPerWarp;
          int const l = e / kChunk;  // A's M: row l of inverse_beta^T (rows >= kChunk are dropped)
          int const i = e % kChunk;  // A's K: the contraction index
          staged[t]   = l < kChunk ? sol[l * CollectiveInverseNxN::SolutionStride + i] : 0.0f;
        }
        MUTLASS_PRAGMA_UNROLL
        for (int t = 0; t < kStageIters; ++t) {
          int const e                                      = lane + t * mutlass::NumThreadsPerWarp;
          sInverseBeta(make_coord(e / kChunk, e % kChunk)) = static_cast<Element>(staged[t]);
        }
      }
      __syncwarp();
      if (lane == 0) {
        named_barrier_arrive(PrepareBarrier::InverseBetaReady);
      }

      // ---- A_rec = P @ inv on one warp's wmma (m16n16k16, layout probed) -----
      {
        using AtomPInv   = conditional_t<is_same_v<Element, mutlass::bfloat16_t>,
                                         mute::MP31_16x16x16_F32BF16BF16F32<mute::TCE::Major::K, mute::TCE::Major::K>,
                                         mute::MP31_16x16x16_F32F16F16F32<mute::TCE::Major::K, mute::TCE::Major::K>>;
        auto         sP  = make_tensor(make_smem_ptr(shared_storage.smem_p_stage.data()), PStagingLayout{});
        float const* sol = shared_storage.smem_inverse_workspace.solution;
        int const    m0  = lane >> 2;
        int const    k0  = (lane & 3) * 2;
        // A = P, K-major: lane owns rows m0 and m0+8, the k pairs (k0, k0+8) -
        // four packed 4-byte loads straight out of the staged bf16 tile (P is
        // cast once, at scatter time, not here).
        uint32_t a_frg[4];
        a_frg[0] = *reinterpret_cast<uint32_t const*>(&sP(make_coord(m0, k0)));
        a_frg[1] = *reinterpret_cast<uint32_t const*>(&sP(make_coord(m0, k0 + 8)));
        a_frg[2] = *reinterpret_cast<uint32_t const*>(&sP(make_coord(m0 + 8, k0)));
        a_frg[3] = *reinterpret_cast<uint32_t const*>(&sP(make_coord(m0 + 8, k0 + 8)));
        // B = inv, K-major: the same ownership on (n, k).  The solution's row n
        // is the inverse's column n, so a k pair is one contiguous float2 -
        // one bf16 convert on the way into the wmma, the same rounding the
        // fold's A operand sees.
        uint32_t b_frg[4];
        MUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 4; ++i) {
          int const          n = m0 + (i >> 1) * 8;
          int const          k = k0 + (i & 1) * 8;
          float2 const       f = *reinterpret_cast<float2 const*>(sol + n * CollectiveInverseNxN::SolutionStride + k);
          alignas(4) Element e[2] = {static_cast<Element>(f.x), static_cast<Element>(f.y)};
          b_frg[i]                = *reinterpret_cast<uint32_t const*>(e);
        }
        float c_frg[8] = {};
        AtomPInv::fma(reinterpret_cast<int32_t*>(c_frg),
                      reinterpret_cast<int32_t const*>(a_frg),
                      reinterpret_cast<int32_t const*>(b_frg),
                      reinterpret_cast<int32_t const*>(c_frg));
        // C ownership: m = lane/8 + 4*v1, n = lane%8 + 8*v0.  Stage back into
        // the P tile (its values are already in registers above), then drain
        // whole rows with one 16-byte store per lane.
        int const ct0 = lane & 7;
        int const ct1 = lane >> 3;
        // Backward-facing copy: `sP` still holds the *unfolded* P here (the loop
        // below overwrites it with A_rec).  K3/K4 need P, not A_rec, so publish
        // it to the workspace field the fold left alone.
        {
          Element* gPB = params.workspace.p_backward + record * Workspace::RecordStride;
          int const jb  = lane >> 1;
          int const lb0 = (lane & 1) * (kChunk / 2);
          *reinterpret_cast<uint4*>(gPB + jb * kChunk + lb0) =
              *reinterpret_cast<uint4 const*>(&sP(make_coord(jb, lb0)));
        }
        MUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < 8; ++v) {
          sP(make_coord(ct1 + (v >> 1) * 4, ct0 + (v & 1) * 8)) = static_cast<Element>(c_frg[v]);
        }
        __syncwarp();
        int const j                                     = lane >> 1;
        int const l0                                    = (lane & 1) * (kChunk / 2);
        *reinterpret_cast<uint4*>(gP + j * kChunk + l0) = *reinterpret_cast<uint4 const*>(&sP(make_coord(j, l0)));
      }

      // Workspace::MatrixLayout keeps the contraction axis contiguous, so the
      // lane-per-column drain is the strided (K-major) form.  The M-major form
      // vectorizes that drain and becomes usable once the recurrence stops
      // reading these rows (the fold's IP removal) or its staged [P; inverse]
      // copy is taught the transposed frame.
      static_assert(mute::is_constant<1, decltype(mute::stride<1>(typename Workspace::MatrixLayout{}))>::value,
                    "publish_solution's major must match Workspace::MatrixLayout");
      CollectiveInverseNxN::template publish_solution<mute::TCE::Major::K>(
          shared_storage.smem_inverse_workspace, gInverseOut, lane);

      // Hand the single-buffered inverse workspace back to the KKT warps.
      __syncwarp();
      if (lane == 0) {
        named_barrier_arrive(PrepareBarrier::KktFree);
      }
      __syncwarp();
    }
  }
};

}  // namespace prepare

}  // namespace mate::flat::kda
