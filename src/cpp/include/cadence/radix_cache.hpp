// The radix prefix cache, ported from cadence/engine/kv/radix_cache.py.
//
// Same tree, same four rules, same owner-sequence bookkeeping; see the Python
// module docstring for why each of them exists. What follows is only what is
// specific to the port.
//
// **Children are keyed by a whole block, not by a first token.** The key type
// is therefore a short vector of tokens (block_size of them), hashed with a
// 64-bit FNV-1a. Keying on the first token would put two prompts that share
// only a ChatML header into the same child slot, where they could not be
// separated: a block is the unit of sharing and cannot be split.
//
// **Node identity is a generation-stamped handle, not a pointer.** Python
// hands the scheduler a `Node` and lets the garbage collector make a
// use-after-free impossible; C++ has no such backstop, and the scheduler does
// hold a node across steps (`Live.prefix_node`). A raw `Node*` in a Python
// object would become a dangling pointer the moment that node were evicted --
// and evicting a node the scheduler still held is precisely the bug this
// project found in the Python version (see ContinuousScheduler._pinned_match).
// So handles are integers looked up in a registry, and a stale one raises a
// Python exception instead of corrupting the heap.
//
// **Thread safety comes from the GIL, and that is why the GIL is not
// released.** Every binding here holds the GIL for the whole of its call, so
// each operation is atomic with respect to the /stats endpoint reading
// `n_nodes` from the API thread. Releasing it around `match` -- which the
// build guide suggests -- would turn that concurrent read into a genuine data
// race over a tree another thread is splitting, in exchange for saving less
// time than the release and re-acquire themselves cost at this call duration.
// The microbenchmark measures both; see bench/bench_core.py.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <unordered_map>
#include <vector>

#include "cadence/block_allocator.hpp"

namespace cadence {

using Token = std::int32_t;
using Key = std::vector<Token>;

struct KeyHash {
  std::size_t operator()(const Key& k) const noexcept {
    std::uint64_t h = 1469598103934665603ULL;  // FNV-1a
    for (Token t : k) {
      h ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(t));
      h *= 1099511628211ULL;
    }
    return static_cast<std::size_t>(h);
  }
};

struct Node {
  Key tokens;                 // edge label
  std::vector<BlockId> blocks;  // covers `tokens`, block-aligned
  std::unordered_map<Key, std::unique_ptr<Node>, KeyHash> children;
  Node* parent = nullptr;
  std::int32_t refs = 0;
  std::uint64_t last_used = 0;  // logical clock, never wall time
  std::int32_t owner_seq = -1;
  std::int32_t depth_tokens = 0;
  std::uint64_t handle = 0;

  [[nodiscard]] bool is_leaf() const noexcept { return children.empty(); }
};

// Shared between the cache and every handle it has handed out, so that a
// handle can tell "evicted" from "the cache itself is gone" without either of
// them owning the other.
struct Registry {
  std::unordered_map<std::uint64_t, Node*> live;
};

struct MatchResult {
  std::int32_t n_tokens = 0;
  std::vector<BlockId> block_ids;
  std::uint64_t handle = 0;  // 0 means "no node": handles start at 1
  std::int32_t owner_seq = -1;
};

class RadixCache {
 public:
  RadixCache(BlockAllocator& alloc, std::int32_t block_size)
      : alloc_(alloc), block_size_(block_size), reg_(std::make_shared<Registry>()) {
    if (block_size <= 0) throw std::invalid_argument("block_size must be positive");
    root_ = std::make_unique<Node>();
    root_->refs = 1;  // the root is never evictable
  }

  // --- counters, mutable because the scheduler advances them once per
  // admission rather than once per lookup (see _match_prefix) -------------
  std::uint64_t query_tokens = 0;
  std::uint64_t hit_tokens = 0;
  std::uint64_t n_evictions = 0;

  [[nodiscard]] std::int32_t block_size() const noexcept { return block_size_; }
  [[nodiscard]] BlockAllocator& blocks() const noexcept { return alloc_; }
  [[nodiscard]] std::shared_ptr<Registry> registry() const noexcept { return reg_; }
  [[nodiscard]] double hit_rate() const noexcept {
    return query_tokens ? static_cast<double>(hit_tokens) / static_cast<double>(query_tokens) : 0.0;
  }

  [[nodiscard]] std::int32_t n_nodes() const {
    std::int32_t n = 0;
    std::vector<const Node*> stack{root_.get()};
    while (!stack.empty()) {
      const Node* cur = stack.back();
      stack.pop_back();
      ++n;
      for (const auto& kv : cur->children) stack.push_back(kv.second.get());
    }
    return n;
  }

  // Every root-to-node token path. Test-facing, not on any hot path: it is
  // what lets the tree-shape assertions in tests/test_radix_cache.py run
  // against this implementation as well as the Python one.
  [[nodiscard]] std::vector<std::vector<Token>> paths() const {
    std::vector<std::vector<Token>> out;
    std::vector<std::pair<const Node*, std::vector<Token>>> stack;
    stack.emplace_back(root_.get(), std::vector<Token>{});
    while (!stack.empty()) {
      auto [node, acc] = std::move(stack.back());
      stack.pop_back();
      for (const auto& kv : node->children) {
        std::vector<Token> path = acc;
        path.insert(path.end(), kv.second->tokens.begin(), kv.second->tokens.end());
        stack.emplace_back(kv.second.get(), path);
        out.push_back(std::move(path));
      }
    }
    return out;
  }

  // --- lookup -----------------------------------------------------------
  MatchResult match(const std::vector<Token>& token_ids, bool count) {
    if (count) query_tokens += token_ids.size();
    // At least one token is always left to prefill: a forward pass needs a
    // token to produce logits from, so a 100% hit would leave nothing to
    // sample.
    const std::int32_t limit =
        std::max(0, static_cast<std::int32_t>(token_ids.size()) - 1);

    Node* node = root_.get();
    Node* best = nullptr;
    std::int32_t i = 0;
    while (i + block_size_ <= limit) {
      auto it = node->children.find(key_at(token_ids, i));
      if (it == node->children.end()) break;
      Node* child = it->second.get();
      const std::int32_t n = common_prefix_len(child->tokens, token_ids, i, limit);
      if (n < static_cast<std::int32_t>(child->tokens.size())) {
        // Partial match inside the edge: split at the last block boundary the
        // two sequences agree on, so the shared head becomes a node in its own
        // right and can be handed out.
        const std::int32_t k = n - (n % block_size_);
        if (k > 0 && child->owner_seq >= 0) {
          split(child, k);
          Node* head = node->children.find(key_at(token_ids, i))->second.get();
          head->last_used = tick();
          best = head;
        }
        break;
      }
      i += n;
      node = child;
      node->last_used = tick();
      if (node->owner_seq >= 0) best = node;
    }

    if (best == nullptr) return {};
    std::int32_t matched = (best->depth_tokens / block_size_) * block_size_;
    matched = std::min(matched, limit);
    matched = (matched / block_size_) * block_size_;
    if (matched == 0) return {};

    MatchResult out;
    out.n_tokens = matched;
    out.block_ids = path_blocks(best, matched / block_size_);
    out.handle = best->handle;
    out.owner_seq = best->owner_seq;
    if (count) hit_tokens += static_cast<std::uint64_t>(matched);
    return out;
  }

  // --- pinning ----------------------------------------------------------
  void acquire(Node* node) {
    for (Node* cur = node; cur != nullptr; cur = cur->parent) {
      ++cur->refs;
      cur->last_used = tick();
    }
  }

  void release_node(Node* node) {
    for (Node* cur = node; cur != nullptr; cur = cur->parent) {
      --cur->refs;
      cur->last_used = tick();
    }
  }

  // --- insertion --------------------------------------------------------
  // Returns the node the path ends at, or nullptr when there was nothing
  // whole-block to store.
  Node* insert(const std::vector<Token>& token_ids, const std::vector<BlockId>& block_ids,
               std::int32_t owner_seq) {
    std::int32_t n = static_cast<std::int32_t>(
        std::min(token_ids.size(), block_ids.size() * static_cast<std::size_t>(block_size_)));
    n = (n / block_size_) * block_size_;
    if (n == 0) return nullptr;

    Node* node = root_.get();
    std::int32_t i = 0;
    while (i < n) {
      auto it = node->children.find(key_at(token_ids, i));
      if (it == node->children.end()) {
        auto fresh = std::make_unique<Node>();
        fresh->tokens.assign(token_ids.begin() + i, token_ids.begin() + n);
        fresh->blocks.assign(block_ids.begin() + i / block_size_,
                             block_ids.begin() + n / block_size_);
        fresh->parent = node;
        fresh->last_used = tick();
        fresh->depth_tokens = n;
        alloc_.share(fresh->blocks);  // the cache takes its own reference
        retain_seq(owner_seq);
        fresh->owner_seq = owner_seq;
        Node* raw = fresh.get();
        adopt(std::move(fresh), node, key_at(token_ids, i));
        return raw;
      }
      Node* child = it->second.get();
      std::int32_t m = common_prefix_len(child->tokens, token_ids, i, n);
      if (m < static_cast<std::int32_t>(child->tokens.size())) {
        const std::int32_t k = m - (m % block_size_);
        if (k == 0) {
          // Unreachable while children are block-keyed: a key match means the
          // first block agrees, so m >= block_size. Kept in code form because
          // descending here would graft this prompt's tail under a node it
          // does not follow.
          return node == root_.get() ? nullptr : node;
        }
        split(child, k);
        child = node->children.find(key_at(token_ids, i))->second.get();
        m = k;
      }
      i += m;
      node = child;
      node->last_used = tick();
    }

    // Exact path already present. Give it an owner if it has none, so that a
    // later request can actually reuse it.
    if (node->owner_seq < 0) {
      retain_seq(owner_seq);
      node->owner_seq = owner_seq;
    }
    return node;
  }

  // --- eviction ---------------------------------------------------------
  // `released` collects owner sequences whose last naming node has gone; the
  // caller fires the Python callback for them once the tree is consistent
  // again, so a re-entrant callback cannot observe a half-evicted cache.
  std::int32_t evict(std::int32_t n_blocks_needed, std::vector<std::int32_t>* released) {
    std::int32_t freed = 0;
    while (freed < n_blocks_needed) {
      Node* victim = lru_leaf();
      if (victim == nullptr) break;
      freed += evict_node(victim, released);
      ++n_evictions;
    }
    return freed;
  }

  std::int32_t evict_nodes(std::int32_t n, std::vector<std::int32_t>* released) {
    std::int32_t evicted = 0;
    for (std::int32_t i = 0; i < std::max(0, n); ++i) {
      Node* victim = lru_leaf();
      if (victim == nullptr) break;
      evict_node(victim, released);
      ++n_evictions;
      ++evicted;
    }
    return evicted;
  }

  Node* resolve(std::uint64_t handle) const {
    auto it = reg_->live.find(handle);
    return it == reg_->live.end() ? nullptr : it->second;
  }

  // The Python callback is invoked by the bindings, not here: this class does
  // not know what a Python object is.
  [[nodiscard]] const std::unordered_map<std::int32_t, std::int32_t>& seq_refs() const noexcept {
    return seq_refs_;
  }

 private:
  std::uint64_t tick() noexcept { return ++clock_; }

  Key key_at(const std::vector<Token>& toks, std::int32_t i) const {
    const auto begin = toks.begin() + i;
    const auto end = (i + block_size_ <= static_cast<std::int32_t>(toks.size()))
                         ? begin + block_size_
                         : toks.end();
    return Key(begin, end);
  }

  static std::int32_t common_prefix_len(const Key& edge, const std::vector<Token>& toks,
                                        std::int32_t from, std::int32_t to) {
    const std::int32_t n = std::min<std::int32_t>(
        static_cast<std::int32_t>(edge.size()), std::max(0, to - from));
    std::int32_t i = 0;
    while (i < n && edge[static_cast<std::size_t>(i)] == toks[static_cast<std::size_t>(from + i)]) {
      ++i;
    }
    return i;
  }

  std::vector<BlockId> path_blocks(const Node* node, std::int32_t take) const {
    std::vector<const Node*> chain;
    for (const Node* cur = node; cur != nullptr && cur != root_.get(); cur = cur->parent) {
      chain.push_back(cur);
    }
    std::vector<BlockId> out;
    for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
      out.insert(out.end(), (*it)->blocks.begin(), (*it)->blocks.end());
      if (static_cast<std::int32_t>(out.size()) >= take) break;
    }
    out.resize(static_cast<std::size_t>(std::min<std::int32_t>(take, static_cast<std::int32_t>(out.size()))));
    return out;
  }

  void adopt(std::unique_ptr<Node> child, Node* parent, const Key& key) {
    child->handle = ++next_handle_;
    reg_->live[child->handle] = child.get();
    parent->children[key] = std::move(child);
  }

  // Split `node`'s edge after `k` tokens. `k` is a multiple of the block size,
  // so the block list splits cleanly too.
  void split(Node* node, std::int32_t k) {
    k -= k % block_size_;
    if (k == 0 || k >= static_cast<std::int32_t>(node->tokens.size())) return;
    Node* parent = node->parent;

    auto head = std::make_unique<Node>();
    head->tokens.assign(node->tokens.begin(), node->tokens.begin() + k);
    head->blocks.assign(node->blocks.begin(), node->blocks.begin() + k / block_size_);
    head->parent = parent;
    head->refs = node->refs;
    head->last_used = node->last_used;
    head->depth_tokens =
        node->depth_tokens - (static_cast<std::int32_t>(node->tokens.size()) - k);
    // The owner's KV covers the whole path, so it also covers any prefix of
    // it: the head can safely name the same sequence.
    if (node->owner_seq >= 0) {
      retain_seq(node->owner_seq);
      head->owner_seq = node->owner_seq;
    }

    // Take ownership of `node` out of the parent before rewiring, so that
    // assigning the head into the same slot cannot destroy it.
    const Key old_key = key_at(node->tokens, 0);
    auto it = parent->children.find(old_key);
    std::unique_ptr<Node> owned = std::move(it->second);
    parent->children.erase(it);

    node->tokens.erase(node->tokens.begin(), node->tokens.begin() + k);
    node->blocks.erase(node->blocks.begin(), node->blocks.begin() + k / block_size_);
    node->parent = head.get();

    const Key child_key = key_at(node->tokens, 0);
    const Key head_key = key_at(head->tokens, 0);
    head->children[child_key] = std::move(owned);
    // The head's first block is the old node's first block, so this overwrites
    // the slot the old node occupied rather than leaking one.
    adopt(std::move(head), parent, head_key);
  }

  // Least-recently-used unreferenced leaf, or nullptr if every leaf is pinned.
  // Ties are impossible by construction: every assignment of `last_used` takes
  // a fresh tick except the one in split(), and the node that inherits a value
  // there is never a leaf.
  Node* lru_leaf() const {
    Node* best = nullptr;
    std::vector<Node*> stack{root_.get()};
    while (!stack.empty()) {
      Node* cur = stack.back();
      stack.pop_back();
      if (cur != root_.get() && cur->is_leaf() && cur->refs == 0) {
        if (best == nullptr || cur->last_used < best->last_used) best = cur;
      }
      for (const auto& kv : cur->children) stack.push_back(kv.second.get());
    }
    return best;
  }

  std::int32_t evict_node(Node* node, std::vector<std::int32_t>* released) {
    Node* parent = node->parent;
    const std::int32_t n = static_cast<std::int32_t>(node->blocks.size());
    alloc_.release(node->blocks);
    release_seq(node->owner_seq, released);
    reg_->live.erase(node->handle);
    // Erasing from the parent destroys the node, so nothing may touch it
    // afterwards.
    parent->children.erase(key_at(node->tokens, 0));
    return n;
  }

  void retain_seq(std::int32_t seq_id) {
    if (seq_id < 0) return;
    ++seq_refs_[seq_id];
  }

  void release_seq(std::int32_t seq_id, std::vector<std::int32_t>* released) {
    if (seq_id < 0) return;
    auto it = seq_refs_.find(seq_id);
    const std::int32_t n = (it == seq_refs_.end() ? 0 : it->second) - 1;
    if (n <= 0) {
      if (it != seq_refs_.end()) seq_refs_.erase(it);
      if (released != nullptr) released->push_back(seq_id);
    } else {
      it->second = n;
    }
  }

  BlockAllocator& alloc_;
  std::int32_t block_size_;
  std::shared_ptr<Registry> reg_;
  std::unique_ptr<Node> root_;
  std::unordered_map<std::int32_t, std::int32_t> seq_refs_;
  std::uint64_t clock_ = 0;
  std::uint64_t next_handle_ = 0;
};

}  // namespace cadence
