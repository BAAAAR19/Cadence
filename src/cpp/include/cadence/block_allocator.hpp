// Paged KV block allocation, ported from cadence/engine/kv/block_manager.py.
//
// The Python file remains the specification: it is the readable statement of
// the semantics and the reference this is fuzzed against
// (tests/test_kv_parity.py). Where the two differ in structure they must not
// differ in behaviour, and every deliberate difference is commented here.
//
// The design is the same one the Python has, for the same reasons, and it
// happens to be the design that is fast in C++ as well:
//
//   * the free list is a std::vector used as a stack, so allocation is a pop
//     with no node chasing and no allocation of its own in steady state;
//   * refcounts are a flat vector indexed by block id rather than a hash map,
//     which is cache-friendly and bounded by the pool size;
//   * every operation is O(1) in the size of the pool.
//
// Nothing here touches a Python object, so nothing here needs the GIL. The
// bindings hold it for the length of each call anyway; see bindings.cpp.

#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace cadence {

using BlockId = std::int32_t;
inline constexpr BlockId kInvalidBlock = -1;

// Translated by the bindings into cadence.engine.kv.block_manager.OutOfBlocks,
// which is the exception the scheduler already catches. A new exception type
// that happened to have the same name would be silently uncaught.
struct OutOfBlocks : std::runtime_error {
  OutOfBlocks(std::int32_t wanted_, std::int32_t available_)
      : std::runtime_error("out of KV blocks"), wanted(wanted_), available(available_) {}
  std::int32_t wanted;
  std::int32_t available;
};

inline std::int32_t ceil_div(std::int32_t a, std::int32_t b) noexcept {
  return (a + b - 1) / b;
}

class BlockAllocator {
 public:
  BlockAllocator(std::int32_t n_blocks, std::int32_t block_size)
      // Validated in the member-initialiser list, not the body: `refcount_` is
      // sized from `n_blocks` and is constructed first, so a negative value
      // checked in the body would already have been cast to a huge size_t and
      // thrown a length_error instead of the ValueError the Python reference
      // raises. Same input, same exception, in both implementations.
      : n_blocks_(require_positive(n_blocks, block_size)),
        block_size_(block_size),
        refcount_(static_cast<std::size_t>(n_blocks), 0) {
    free_.reserve(static_cast<std::size_t>(n_blocks));
    for (BlockId b = n_blocks - 1; b >= 0; --b) free_.push_back(b);
  }
  virtual ~BlockAllocator() = default;

  // --- accounting -------------------------------------------------------
  [[nodiscard]] std::int32_t n_blocks() const noexcept { return n_blocks_; }
  [[nodiscard]] std::int32_t block_size() const noexcept { return block_size_; }
  [[nodiscard]] std::int32_t free_blocks() const noexcept {
    return static_cast<std::int32_t>(free_.size());
  }
  [[nodiscard]] std::int32_t used_blocks() const noexcept {
    return n_blocks_ - free_blocks();
  }
  [[nodiscard]] std::int64_t n_copy_on_write() const noexcept { return n_cow_; }

  // Worst-case footprint: what the request would occupy at its full token
  // budget. The admission gate, deliberately not what gets allocated.
  [[nodiscard]] virtual std::int32_t blocks_needed(std::int32_t n_tokens,
                                                   std::int32_t max_new_tokens,
                                                   std::int32_t cached) const {
    const std::int32_t total = n_tokens + max_new_tokens;
    return std::max(0, ceil_div(total, block_size_) - ceil_div(cached, block_size_));
  }

  // Blocks to allocate at admission: enough for the prompt and no more.
  [[nodiscard]] virtual std::int32_t initial_blocks(std::int32_t n_prompt_tokens,
                                                    std::int32_t cached) const {
    return std::max(0, ceil_div(n_prompt_tokens, block_size_) - ceil_div(cached, block_size_));
  }

  // --- allocation -------------------------------------------------------
  // Strong exception guarantee: on failure nothing is consumed.
  std::vector<BlockId> alloc(std::int32_t n) {
    if (n < 0) throw std::invalid_argument("n must be >= 0");
    if (n > free_blocks()) throw OutOfBlocks(n, free_blocks());
    std::vector<BlockId> out;
    out.reserve(static_cast<std::size_t>(n));
    for (std::int32_t i = 0; i < n; ++i) {
      const BlockId b = free_.back();
      free_.pop_back();
      refcount_[static_cast<std::size_t>(b)] = 1;
      out.push_back(b);
    }
    return out;
  }

  // One pass, applying as it goes, and therefore leaving a partially applied
  // mutation behind if it throws half way through. That is what the Python
  // reference does, and matching it is deliberate: both of these throw only
  // when the caller has already lost track of its own block table, the
  // scheduler treats either as fatal, and a divergence here would be a
  // divergence the differential test could not see past.
  void share(const std::vector<BlockId>& ids) {
    for (BlockId b : ids) {
      if (at(b) <= 0) throw std::runtime_error("cannot share free block " + std::to_string(b));
      ++refcount_[static_cast<std::size_t>(b)];
    }
  }

  void release(const std::vector<BlockId>& ids) {
    for (BlockId b : ids) {
      if (at(b) <= 0) throw std::runtime_error("double free of block " + std::to_string(b));
      if (--refcount_[static_cast<std::size_t>(b)] == 0) free_.push_back(b);
    }
  }

  [[nodiscard]] std::int32_t refcount(BlockId b) const { return at(b); }
  [[nodiscard]] bool is_shared(BlockId b) const { return at(b) > 1; }

  // --- copy-on-write ----------------------------------------------------
  // Privatise a shared block that is about to be appended to. The caller is
  // responsible for copying the block's contents; here it is pure
  // bookkeeping, which is what makes it testable.
  BlockId prepare_write(BlockId old) {
    if (at(old) == 1) return old;
    const BlockId fresh = alloc(1).front();  // throws OutOfBlocks, consuming nothing
    ++n_cow_;
    release({old});
    return fresh;
  }

  // --- growth -----------------------------------------------------------
  [[nodiscard]] std::int32_t slots_in(std::int32_t n_table) const noexcept {
    return n_table * block_size_;
  }

  // Can `n` more tokens be written, given the free list as it stands?
  //
  // `shared_at_idx` is whether the block the next token would land in is
  // shared; the caller reads it out of the block table, because the table
  // lives on the Python side. Spare room in that block is not sufficient on
  // its own: writing into a shared block first requires privatising it, and
  // that copy needs a free block like any other allocation.
  [[nodiscard]] bool can_append(std::int32_t n_table, std::int32_t n_filled, std::int32_t n,
                                bool shared_at_idx) const noexcept {
    const std::int32_t spare = slots_in(n_table) - n_filled;
    const std::int32_t cow = shared_at_idx ? 1 : 0;
    if (spare >= n) return cow <= free_blocks();
    return ceil_div(n - spare, block_size_) + cow <= free_blocks();
  }

  // --- metrics ----------------------------------------------------------
  [[nodiscard]] double fragmentation_ratio(std::int64_t live_tokens) const noexcept {
    const std::int32_t used = used_blocks();
    if (used == 0) return 1.0;
    return static_cast<double>(live_tokens) /
           (static_cast<double>(used) * static_cast<double>(block_size_));
  }

  [[nodiscard]] const std::vector<std::int32_t>& refcounts() const noexcept { return refcount_; }
  [[nodiscard]] const std::vector<BlockId>& free_list() const noexcept { return free_; }

 protected:
  static std::int32_t require_positive(std::int32_t n_blocks, std::int32_t block_size) {
    if (n_blocks <= 0 || block_size <= 0) {
      throw std::invalid_argument("n_blocks and block_size must be positive");
    }
    return n_blocks;
  }

  // Block ids are opaque tokens handed out by alloc(), so they are
  // non-negative by construction; an id outside the pool means the caller has
  // lost track of its own block table, which is an IndexError in Python and is
  // one here too.
  [[nodiscard]] std::int32_t at(BlockId b) const {
    if (b < 0 || b >= n_blocks_) throw std::out_of_range("block id out of range");
    return refcount_[static_cast<std::size_t>(b)];
  }

  std::int32_t n_blocks_;
  std::int32_t block_size_;
  std::vector<std::int32_t> refcount_;
  std::vector<BlockId> free_;  // LIFO: the most recently freed block is warm
  std::int64_t n_cow_ = 0;
};

// The allocator paged KV replaces, kept so the ablation compares reservation
// *policies* rather than two codebases. A sequence gets a slab sized for the
// server-wide output cap, so every request that stops early leaves the tail of
// its slab reserved and unusable.
class ContiguousBlockAllocator : public BlockAllocator {
 public:
  ContiguousBlockAllocator(std::int32_t n_blocks, std::int32_t block_size,
                           std::int32_t reserve_tokens)
      : BlockAllocator(n_blocks, block_size), reserve_tokens_(reserve_tokens) {}

  [[nodiscard]] std::int32_t blocks_needed(std::int32_t n_tokens, std::int32_t /*max_new*/,
                                           std::int32_t cached) const override {
    const std::int32_t total = n_tokens + reserve_tokens_;
    return std::max(0, ceil_div(total, block_size_) - ceil_div(cached, block_size_));
  }
  [[nodiscard]] std::int32_t initial_blocks(std::int32_t n_prompt_tokens,
                                            std::int32_t cached) const override {
    return blocks_needed(n_prompt_tokens, 0, cached);
  }
  [[nodiscard]] std::int32_t reserve_tokens() const noexcept { return reserve_tokens_; }

 private:
  std::int32_t reserve_tokens_;
};

}  // namespace cadence
