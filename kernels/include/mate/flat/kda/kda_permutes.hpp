#pragma once

#include <mutlass/arch/arch.h>

#include <mute/layout.hpp>

namespace mate::flat::kda {

// The KDA axis permutations: the compile-time layout isomorphisms that the two
// kernels -- or a kernel and the workspace ABI -- have to interpret the same
// way.
//
// Each is a mute layout pinned by a bidirectional static_assert, because the
// alternative is a closed form on one side and a comment restating it on the
// other, and nothing makes the two move together.
//
// The names read as "permute <axis>": PermuteN permutes the GEMM's N axis, and
// its parameter is that axis's width (a value-axis PermuteV would sit beside it).

// The order of the headDim axis of the k_restored workspace field.  headDim is
// the UK GEMM's N axis, so each architecture's frame is a permutation of N.
//
// Declared but not defined: an architecture without a frame is a hard error
// rather than a silent fallback.
template <class ArchTag, int HeadDim>
struct PermuteN;

// ---------------------------------------------------------------------------
// MP31
//
// K1 (prepare) writes logical key h of k_restored at slot
//
//     slot = G^-1(h) = 8 * (h % 16) + h / 16        (h < 128)
//
// so that K2's state squad lane holds 16 CONSECUTIVE smem slots per row -- one
// 32 B vector store instead of 16 stride-8 scalar stores.  Writing h = 16*a + b
// with a < 8, b < 16 gives slot = a + 8*b: a transpose of the (a, b) 8x16 block.
//
// K2 does not evaluate this -- the permutation is baked into its UK operand
// layout -- which is why the frame has one caller rather than two.  It still
// belongs in a frame: it is the one place the order is written down, and K1
// used to spell it out inline.
template <int HeadDim>
struct PermuteN<mutlass::arch::Mp31, HeadDim> {
  static_assert(HeadDim == 128, "the MP31 PermuteN is the 8x16 decomposition of a 128-wide headDim");

  // mute's idx2crd is column-major (mode 0 varies fastest) and Step assigns
  // strides in increasing step order, so the domain is Shape<_16, _8>: h = 16*a
  // + b lands on (b, a) and the strides below give 8*b + a.
  using LogicalToSlot =
      decltype(mute::make_ordered_layout(mute::Shape<mute::_16, mute::_8>{}, mute::Step<mute::_8, mute::_1>{}));
  using SlotToLogical = decltype(mute::right_inverse(LogicalToSlot{}));

  // h = 19 -> (a, b) = (1, 3) -> 25;  h = 61 -> (a, b) = (3, 13) -> 107.
  static_assert(LogicalToSlot{}(19) == 25 && LogicalToSlot{}(61) == 107, "MP31 PermuteN moved");
  static_assert(SlotToLogical{}(25) == 19 && SlotToLogical{}(107) == 61, "MP31 PermuteN moved");
};

// ---------------------------------------------------------------------------
// MP32
//
// The one thing the workspace record cannot state about the k_restored field:
// the ORDER of its headDim axis.  MP32's prepare kernel (K1) writes a logical key
// through LogicalToSlot, and MP32's recurrence (K2) reads the logical key back
// through SlotToLogical.  MP31 needs neither: there the same permutation falls
// out of its k_restored smem tile layout and the TME store.
//
// K2's UK SQMMA reads k_restored as its B operand, so that accumulator's N axis
// IS this frame and its C layout hands a lane the four-strided slots
// (lane % 4) + 4*i.  Storing the logical key at the slot below is what turns
// those slots into a lane's eight CONSECUTIVE logical keys, which is what makes
// the state merge one 16 B store per group instead of a scatter.
//
//   logical_k = 32*a + 8*u + v  ->  slot = 32*a + 4*v + u   (a < 4, u < 4, v < 8)
//
// i.e. the (u,v) 4x8 block is transposed.  Each side used to carry its own closed
// form; the checks below pin the mapping down so the two kernels cannot drift.

// ---------------------------------------------------------------------------
// The k axis.
//
// The relabel that makes an MP32 fp32 accumulator usable directly as the
// register A operand of an RS-SQMMA (MUTE_ARCH_SQMMA_MP32_ENABLED).
//
// For an [M,N] accumulator of any MP32 bf16 atom, thread t owns columns
//
//     n = (t % 4) + 4 * (v % 2) + 8 * ((v / 2) % (N / 8))
//
// while the A operand of an [M,K] RS atom must supply
//
//     k = 2 * (t % 4) + (v % 2) + 8 * ((v / 2) % (K / 8)).
//
// The row index m is literally the same expression on both sides, so the two
// fragments are related register for register -- but the SAME register of the
// SAME lane means column n on one side and column k on the other.  With N == K
// the (v % 2) and (v / 2) digits agree, and the mismatch is exactly
//
//     n = r + 4*a + 8*b    ->    k = 2*r + a + 8*b      (r = t % 4 < 4, a = v % 2)
//
// i.e. a relabelling of the k axis: the accumulator's bit 2 becomes the A
// operand's bit 0, and its bits 0..1 become bits 1..2.  As a closed form,
//
//     pi(k) = 2 * (k % 4) + ((k / 4) % 2) + (k - k % 8)
//
// a ROTATION of the low three bits (bit 0 <- bit 2 <- bit 1 <- bit 0), so it is a
// bijection but NOT an involution: pi^3 is the identity, pi^2 is not.  Bits 3 and
// up are untouched, so the 32-column chunk index that separates one [M,32] A
// fragment from the next is preserved.
//
// Consequence: put logical key pi(c) at state column c -- equivalently, lay the B
// operand's k rows out in pi order -- and every k step of the recurrence's state
// update converts acc into A with zero register moves and zero shuffles, because
// the register already holds what that register of the A operand has to contain.
// The workspace fields that contract over the state's key axis have to be written
// in the same frame (see the MP32 prepare drain).
//
// Declared but not defined, like PermuteN: MP31 has no register-A atom at all, so
// it has no such frame, and asking for one is a hard error rather than a silent
// identity.
template <class ArchTag, int D>
struct PermuteKPi;

// MP32.  D is the width of the state's key axis; the frame is one K = 32 chunk
// wide, so D has to be a whole number of them.

}  // namespace mate::flat::kda

// ---------------------------------------------------------------------------
// The value axis.
//
// Which tile row holds a given value column, and what an mma atom's C
// coordinate means on this axis.  The two architectures answer differently --
// see the specialisations -- and that difference is what makes each arch's
// packed 16 B store legal.
//
// A map is a bijection as soon as its extents multiply out to ValueTile:
// make_ordered_layout, filter and right_inverse all yield compact layouts.  So
// each frame asserts the coverage it is responsible for -- that its reshape is
// exactly the value axis -- and pins the concrete order with literal values.
//
// Declared but not defined, like PermuteN.
template <class ArchTag, int ValueTile>
struct PermuteV;

// ---------------------------------------------------------------------------
// MP31
//
// The SQMMA enumerates a lane's value columns in 8x8-transposed blocks, and the
// state tile stores the logical order.  An atom coordinate therefore reaches the
// tile row through that transpose, and a logical column IS its tile row.
template <int ValueTile>
struct PermuteV<mutlass::arch::Mp31, ValueTile> {
  static_assert(ValueTile == 32 || ValueTile % 64 == 0,
                "MP31's value permutation is a whole number of 64-column blocks, or 32");

  using Natural = mute::Layout<mute::Shape<mute::Int<ValueTile>>, mute::Stride<mute::_1>>;
  // The third mode is 1 for ValueTile == 64 and gets filtered away.  It is also 1
  // at 32, where the conditional below picks Natural instead.
  using Transposed = decltype(mute::filter(
      mute::make_ordered_layout(mute::Shape<mute::_8, mute::_8, mute::Int<(ValueTile + 63) / 64>>{},
                                mute::Step<mute::_2, mute::_1, mute::_3>{})));

  static constexpr bool kTransposed = ValueTile != 32;

  using TileRowFromInternal = std::conditional_t<kTransposed, Transposed, Natural>;
  using TileRowFromLogical  = Natural;
  using LogicalFromTileRow  = Natural;
  using LogicalFromInternal = TileRowFromInternal;

  // At 32 columns the frame is Natural, so only the transpose needs checking.
  static_assert(!kTransposed || mute::size(Transposed{}) == ValueTile,
                "MP31's value transpose must cover exactly ValueTile columns");
  // 64 columns: value 1 sits at row 8 and value 8 at row 1 (the 8x8 block transpose).
  static_assert(!kTransposed || (TileRowFromInternal{}(1) == 8 && TileRowFromInternal{}(8) == 1),
                "MP31's value transpose moved");
};

// ---------------------------------------------------------------------------
// MP32
//
// A lane's DS/IP/PU register r sits on tile row (lane % 4) + 4r, so the atom's C
// column already IS the tile row and TileRowFromInternal is the identity.  The
// tile stores logical column v at
//
//   physical(v) = 4 * (v % (ValueTile / 4)) + v / (ValueTile / 4)
//
// -- the (ValueTile / 4, 4) reshape transposed -- which is what turns those
// stride-4 rows back into CONSECUTIVE logical columns for the lane:
//
//   natural((lane % 4) + 4r) = (ValueTile / 4) * (lane % 4) + r
//
// For ValueTile == 64 that is natural(p) = 16 * (p % 4) + p / 4 and its inverse
// physical(v) = 4 * (v % 16) + v / 16; for 32 it is natural(p) = 8 * (p % 4) + p / 4.
