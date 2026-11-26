#pragma once

#include <cstddef>

namespace mate::flat::kda {

// Resource and traversal choices, independent of tensor/problem representation.
// K1 and K2 select these at compile time; they are not runtime tuning knobs.
template <int         PrepareCtasPerMp_,
          int         RecurrenceCtasPerMp_,
          int         RecurrenceValueTile_,
          int         MetadataThreads_,
          std::size_t MetadataSharedLimit_>
struct ChunkKdaPrefillSchedulePolicy {
  static constexpr int         PrepareCtasPerMp    = PrepareCtasPerMp_;
  static constexpr int         RecurrenceCtasPerMp = RecurrenceCtasPerMp_;
  static constexpr int         RecurrenceValueTile = RecurrenceValueTile_;
  static constexpr int         MetadataThreads     = MetadataThreads_;
  static constexpr std::size_t MetadataSharedLimit = MetadataSharedLimit_;

  static_assert(PrepareCtasPerMp > 0);
  static_assert(RecurrenceCtasPerMp > 0);
  static_assert(RecurrenceValueTile > 0);
  static_assert(MetadataThreads > 0);
  static_assert(MetadataSharedLimit > 0);
};

}  // namespace mate::flat::kda
