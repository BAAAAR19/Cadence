// pybind11 bindings for the KV core.
//
// The contract is exact drop-in replacement: the scheduler in
// cadence/engine/scheduler/continuous.py must not be able to tell which
// implementation it was handed. That constrains the bindings in three ways
// worth naming.
//
// **Block tables stay Python lists.** `can_append`, `append_slot` and
// `fork_for_write` take the sequence's block table as a `py::list` and mutate
// it in place, because that is what the Python does and what the rest of the
// engine reads (`rq.block_ids`). Converting it to a std::vector on every
// decode step would cost O(len) per sequence per token to save nothing: the
// operations themselves only ever look at the length and one element, so the
// binding does exactly that and stays O(1).
//
// **Exceptions are the Python ones.** `OutOfBlocks` is translated back into
// cadence.engine.kv.block_manager.OutOfBlocks, because the scheduler catches
// that class by identity. A same-named C++ exception would sail straight
// through the `except OutOfBlocks` in `_grow` and kill the engine thread.
//
// **The GIL is held throughout.** See the note at the top of
// radix_cache.hpp: holding it is what makes every operation atomic against
// the /stats reader on the API thread, and at these call durations releasing
// it would cost more than it saves.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "cadence/block_allocator.hpp"
#include "cadence/radix_cache.hpp"

namespace py = pybind11;
using namespace cadence;

namespace {

// --- block tables as Python lists ----------------------------------------
std::int32_t list_len(const py::list& table) {
  return static_cast<std::int32_t>(table.size());
}

BlockId list_at(const py::list& table, std::int32_t i) {
  return table[static_cast<std::size_t>(i)].cast<BlockId>();
}

bool shared_at(const BlockAllocator& self, const py::list& table, std::int32_t idx) {
  if (idx < 0 || idx >= list_len(table)) return false;
  return self.is_shared(list_at(table, idx));
}

// --- the prefix cache's Python-facing node handle -------------------------
//
// A generation-stamped handle rather than a pointer: see radix_cache.hpp. The
// weak_ptr distinguishes "this node was evicted" from "the cache it belonged
// to is gone", and both raise rather than dereferencing anything.
struct NodeRef {
  std::weak_ptr<Registry> reg;
  std::uint64_t handle = 0;

  [[nodiscard]] Node* get() const {
    auto r = reg.lock();
    if (!r) throw std::runtime_error("radix node outlived its cache");
    auto it = r->live.find(handle);
    if (it == r->live.end()) {
      throw std::runtime_error(
          "radix node " + std::to_string(handle) +
          " has been evicted; it was used after the cache dropped it");
    }
    return it->second;
  }
  [[nodiscard]] bool alive() const {
    auto r = reg.lock();
    return r && r->live.count(handle) > 0;
  }
};

py::object wrap_node(const RadixCache& cache, std::uint64_t handle) {
  if (handle == 0) return py::none();
  return py::cast(NodeRef{cache.registry(), handle});
}

Node* node_arg(const py::object& obj) {
  if (obj.is_none()) return nullptr;
  return obj.cast<const NodeRef&>().get();
}

// The Match the scheduler unpacks: `hit.n_tokens`, `hit.block_ids`,
// `hit.node`, `hit.owner_seq`, `hit.hit`.
struct PyMatch {
  std::int32_t n_tokens = 0;
  py::list block_ids;
  py::object node = py::none();
  std::int32_t owner_seq = -1;
};

PyMatch make_match(const RadixCache& cache, const MatchResult& r) {
  PyMatch m;
  m.n_tokens = r.n_tokens;
  for (BlockId b : r.block_ids) m.block_ids.append(b);
  m.node = wrap_node(cache, r.handle);
  m.owner_seq = r.owner_seq;
  return m;
}

// The cache holds the callback; the bindings own firing it, because the core
// has no idea what a Python object is. Fired only once the tree is consistent
// again, so a callback that re-entered the cache would find it in a valid
// state.
struct CacheHolder {
  RadixCache cache;
  py::object on_seq_released = py::none();

  CacheHolder(BlockAllocator& alloc, std::int32_t block_size) : cache(alloc, block_size) {}

  void fire(const std::vector<std::int32_t>& released) {
    if (released.empty() || on_seq_released.is_none()) return;
    for (std::int32_t sid : released) on_seq_released(sid);
  }
};

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "Cadence KV core: paged block allocator + radix prefix cache (C++17)";
  m.attr("__all__") = py::make_tuple("BlockAllocator", "ContiguousBlockAllocator",
                                     "RadixCache", "Match", "NodeRef");

  // Benchmark scaffolding, not API. `bench/bench_core.py` times these two
  // against each other to price the GIL release the build guide recommends
  // around `match`; see the note at the top of radix_cache.hpp for why the
  // answer is that it is not worth taking.
  m.def("_noop", []() {});
  // The floor under every call that takes a prompt: pybind11 copies the
  // Python list into a std::vector before the function body runs. Timing this
  // separates "the tree walk is fast" from "the boundary is not free", which
  // turned out to be the whole story of the match benchmark.
  m.def("_consume_tokens", [](const std::vector<Token>& t) { return t.size(); });
  m.def("_noop_gil_released", []() {}, py::call_guard<py::gil_scoped_release>());

  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const OutOfBlocks& e) {
      // Raise the *Python* class, so `except OutOfBlocks` in the scheduler
      // catches it and the `.wanted` / `.available` attributes are there.
      py::object cls =
          py::module_::import("cadence.engine.kv.block_manager").attr("OutOfBlocks");
      PyErr_SetObject(cls.ptr(), py::make_tuple(e.wanted, e.available).ptr());
    }
  });

  // --- BlockAllocator ----------------------------------------------------
  py::class_<BlockAllocator>(m, "BlockAllocator")
      .def(py::init<std::int32_t, std::int32_t>(), py::arg("n_blocks"),
           py::arg("block_size") = 16)
      .def_property_readonly("n_blocks", &BlockAllocator::n_blocks)
      .def_property_readonly("block_size", &BlockAllocator::block_size)
      .def_property_readonly("n_copy_on_write", &BlockAllocator::n_copy_on_write)
      .def_property_readonly(
          "free",
          [](const BlockAllocator& s) {
            // A copy, not a view: handing out a mutable window into the free
            // list would let Python corrupt the invariant that a block is on
            // the free list exactly when its refcount is zero.
            py::list out;
            for (BlockId b : s.free_list()) out.append(b);
            return out;
          },
          "Snapshot of the free list, LIFO order (last element is allocated next).")
      .def_property_readonly("refcount",
                             [](const BlockAllocator& s) {
                               py::list out;
                               for (std::int32_t rc : s.refcounts()) out.append(rc);
                               return out;
                             })
      .def("free_blocks", &BlockAllocator::free_blocks)
      .def("used_blocks", &BlockAllocator::used_blocks)
      .def("blocks_needed", &BlockAllocator::blocks_needed, py::arg("n_tokens"),
           py::arg("max_new_tokens"), py::arg("cached") = 0)
      .def("initial_blocks", &BlockAllocator::initial_blocks, py::arg("n_prompt_tokens"),
           py::arg("cached") = 0)
      .def("alloc",
           [](BlockAllocator& s, std::int32_t n) {
             py::list out;
             for (BlockId b : s.alloc(n)) out.append(b);
             return out;
           },
           py::arg("n"))
      .def("share", &BlockAllocator::share, py::arg("block_ids"))
      .def("release", &BlockAllocator::release, py::arg("block_ids"))
      .def("is_shared", &BlockAllocator::is_shared, py::arg("block_id"))
      .def("refcount_of", &BlockAllocator::refcount, py::arg("block_id"))
      .def("slots_in",
           [](const BlockAllocator& s, const py::list& table) {
             return s.slots_in(list_len(table));
           },
           py::arg("block_table"))
      .def("can_append",
           [](const BlockAllocator& s, const py::list& table, std::int32_t n_filled,
              std::int32_t n) {
             const std::int32_t idx = n_filled / s.block_size();
             return s.can_append(list_len(table), n_filled, n, shared_at(s, table, idx));
           },
           py::arg("block_table"), py::arg("n_filled"), py::arg("n") = 1)
      .def("fork_for_write",
           [](BlockAllocator& s, py::list table, std::int32_t block_idx) {
             const BlockId old = list_at(table, block_idx);
             const BlockId fresh = s.prepare_write(old);
             if (fresh != old) table[static_cast<std::size_t>(block_idx)] = py::cast(fresh);
             return fresh;
           },
           py::arg("block_table"), py::arg("block_idx"))
      .def("append_slot",
           [](BlockAllocator& s, py::list table, std::int32_t n_filled) -> py::object {
             if (s.slots_in(list_len(table)) > n_filled) {
               const std::int32_t idx = n_filled / s.block_size();
               if (shared_at(s, table, idx)) {
                 const BlockId fresh = s.prepare_write(list_at(table, idx));
                 table[static_cast<std::size_t>(idx)] = py::cast(fresh);
               }
               return py::none();
             }
             const BlockId fresh = s.alloc(1).front();
             table.append(fresh);
             return py::cast(fresh);
           },
           py::arg("block_table"), py::arg("n_filled"),
           "Make room for one more token. Returns a newly allocated block id "
           "if one was needed, else None.")
      .def("fragmentation_ratio",
           [](const BlockAllocator& s, const std::vector<std::int64_t>& live) {
             std::int64_t total = 0;
             for (std::int64_t n : live) total += n;
             return s.fragmentation_ratio(total);
           },
           py::arg("live_token_counts"))
      .def("snapshot", [](const BlockAllocator& s) {
        py::dict d;
        d["total"] = s.n_blocks();
        d["free"] = s.free_blocks();
        d["used"] = s.used_blocks();
        d["cow"] = s.n_copy_on_write();
        return d;
      });

  py::class_<ContiguousBlockAllocator, BlockAllocator>(m, "ContiguousBlockAllocator")
      .def(py::init<std::int32_t, std::int32_t, std::int32_t>(), py::arg("n_blocks"),
           py::arg("block_size") = 16, py::arg("reserve_tokens") = 512)
      .def_property_readonly("reserve_tokens", &ContiguousBlockAllocator::reserve_tokens);

  // --- radix cache -------------------------------------------------------
  py::class_<NodeRef>(m, "NodeRef")
      .def_property_readonly("owner_seq", [](const NodeRef& r) { return r.get()->owner_seq; })
      .def_property_readonly("refs", [](const NodeRef& r) { return r.get()->refs; })
      .def_property_readonly("depth_tokens",
                             [](const NodeRef& r) { return r.get()->depth_tokens; })
      .def_property_readonly("last_used", [](const NodeRef& r) { return r.get()->last_used; })
      .def_property_readonly("is_leaf", [](const NodeRef& r) { return r.get()->is_leaf(); })
      .def_property_readonly("alive", &NodeRef::alive,
                             "False once the node has been evicted. Every other "
                             "attribute raises in that state rather than reading "
                             "freed memory.")
      .def("__eq__",
           [](const NodeRef& a, const py::object& b) {
             if (!py::isinstance<NodeRef>(b)) return false;
             const auto& o = b.cast<const NodeRef&>();
             return a.handle == o.handle && !a.reg.owner_before(o.reg) &&
                    !o.reg.owner_before(a.reg);
           })
      .def("__hash__", [](const NodeRef& r) { return py::hash(py::int_(r.handle)); })
      .def("__repr__", [](const NodeRef& r) {
        return "<NodeRef " + std::to_string(r.handle) + (r.alive() ? ">" : " evicted>");
      });

  py::class_<PyMatch>(m, "Match")
      .def_readonly("n_tokens", &PyMatch::n_tokens)
      .def_readonly("block_ids", &PyMatch::block_ids)
      .def_readonly("node", &PyMatch::node)
      .def_readonly("owner_seq", &PyMatch::owner_seq)
      .def_property_readonly("hit", [](const PyMatch& m) { return m.n_tokens > 0; })
      .def("__repr__", [](const PyMatch& m) {
        return "Match(n_tokens=" + std::to_string(m.n_tokens) + ")";
      });

  py::class_<CacheHolder>(m, "RadixCache")
      .def(py::init<BlockAllocator&, std::int32_t>(), py::arg("blocks"),
           py::arg("block_size") = 16,
           // Without this the collector can free the allocator while the cache
           // still holds a reference to it: an intermittent segfault under
           // load, which is the worst kind of bug to have shipped.
           py::keep_alive<1, 2>())
      .def_readwrite("on_seq_released", &CacheHolder::on_seq_released)
      .def_property(
          "query_tokens", [](const CacheHolder& h) { return h.cache.query_tokens; },
          [](CacheHolder& h, std::uint64_t v) { h.cache.query_tokens = v; })
      .def_property(
          "hit_tokens", [](const CacheHolder& h) { return h.cache.hit_tokens; },
          [](CacheHolder& h, std::uint64_t v) { h.cache.hit_tokens = v; })
      .def_property(
          "n_evictions", [](const CacheHolder& h) { return h.cache.n_evictions; },
          [](CacheHolder& h, std::uint64_t v) { h.cache.n_evictions = v; })
      .def_property_readonly("block_size", [](const CacheHolder& h) { return h.cache.block_size(); })
      .def_property_readonly("blocks", [](CacheHolder& h) { return &h.cache.blocks(); },
                             py::return_value_policy::reference_internal)
      .def_property_readonly("hit_rate", [](const CacheHolder& h) { return h.cache.hit_rate(); })
      .def("n_nodes", [](const CacheHolder& h) { return h.cache.n_nodes(); })
      .def("paths", [](const CacheHolder& h) { return h.cache.paths(); })
      .def("n_owned_sequences",
           [](const CacheHolder& h) { return h.cache.seq_refs().size(); })
      .def("match",
           [](CacheHolder& h, const std::vector<Token>& token_ids, bool count) {
             return make_match(h.cache, h.cache.match(token_ids, count));
           },
           py::arg("token_ids"), py::kw_only(), py::arg("count") = true)
      .def("acquire", [](CacheHolder& h, const py::object& node) { h.cache.acquire(node_arg(node)); },
           py::arg("node"))
      .def("release", [](CacheHolder& h, const py::object& node) { h.cache.release_node(node_arg(node)); },
           py::arg("node"))
      .def("insert",
           [](CacheHolder& h, const std::vector<Token>& token_ids,
              const std::vector<BlockId>& block_ids, std::int32_t owner_seq) {
             Node* n = h.cache.insert(token_ids, block_ids, owner_seq);
             return n == nullptr ? py::none() : wrap_node(h.cache, n->handle);
           },
           py::arg("token_ids"), py::arg("block_ids"), py::arg("owner_seq"))
      .def("evict",
           [](CacheHolder& h, std::int32_t n_blocks_needed) {
             std::vector<std::int32_t> released;
             const std::int32_t freed = h.cache.evict(n_blocks_needed, &released);
             h.fire(released);
             return freed;
           },
           py::arg("n_blocks_needed"))
      .def("evict_nodes",
           [](CacheHolder& h, std::int32_t n) {
             std::vector<std::int32_t> released;
             const std::int32_t evicted = h.cache.evict_nodes(n, &released);
             h.fire(released);
             return evicted;
           },
           py::arg("n"))
      .def("evict_all_unused", [](CacheHolder& h) {
        std::vector<std::int32_t> released;
        const std::int32_t freed = h.cache.evict(h.cache.blocks().n_blocks(), &released);
        h.fire(released);
        return freed;
      });
}
