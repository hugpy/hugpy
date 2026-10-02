// Serving-table layout persistence: the localStorage key + the DEFAULT column
// order (left-to-right, after the fixed select-checkbox). Operator ask
// 2026-07-28 REPLACES the 07-18 order: Model leads (it is now the FROZEN
// identity column — sticky-left during horizontal scroll, and excluded from
// drag-reorder so it always stays first), then the sizing/cost columns
// (Memory, Alloc, Size, Ctx), then the routing identifiers (Task, Engine,
// 4-bit, MoE), then the live/placement columns (State, Seat, Residency, 📌),
// with Actions last — ⛔ block is a button INSIDE the Actions cell, not a
// column of its own, so the requested trailing "block" lands there.
// The order/widths are user-adjustable (drag to reorder, drag a header edge to
// resize) and persisted per browser; useColumnLayout merges this list against a
// stored layout so a future column add/remove needs no version bump (see hook).
// The key is bumped to v2 BECAUSE the default ORDER changed: the hook's merge
// deliberately honours a stored order, so operators carrying a v1 layout would
// otherwise never see the new default.
export const SERV_LAYOUT_KEY = 'hugpy.workers.servtable.layout.v2'

export const SERV_DEFAULT_ORDER = [
  'name', 'memory', 'alloc', 'size', 'ctx', 'task', 'framework',
  'fourbit', 'moe', 'state', 'seat', 'residency', 'pin', 'actions',
]

// The one column that is PINNED left (frozen during horizontal scroll) and
// therefore not a drag-reorder source or drop target. Its sort click still works.
// It is ALSO the fluid column (operator ask 2026-07-28): no manual resize, width
// is a clamp() in CSS, and any persisted px width for it is ignored (see the
// <colgroup> below) — deliberately WITHOUT bumping SERV_LAYOUT_KEY, so every
// other column's stored width survives.
export const SERV_FROZEN_COL = 'name'

// ── Compact (narrow-viewport) mode ───────────────────────────────────────────
// Below SERV_COMPACT_PX of TABLE-WRAPPER width the serving table drops to three
// columns and moves everything else into a per-row drawer. The trigger is the
// WRAPPER's width, not the viewport's (a ResizeObserver, not matchMedia): the
// panel can be narrow inside a wide window (a split tab pane, a narrow card),
// and a viewport media query would leave that case unusably wide-scrolling.
export const SERV_COMPACT_PX = 720

// The only columns that stay in the table in compact mode, left-to-right. Every
// other column def is rendered as a labeled chip in the row drawer instead.
export const SERV_COMPACT_COLS = ['name', 'state', 'memory']

// One concise allocation editor: a mode dropdown + explicit per-model budgets
// (VRAM / RAM / cores) in custom mode — the model's resource contract.
// The worker's EFFECTIVE capacity (bytes) for a resource — the SAME universe the
// honest bars use (limit when set, else physical): VRAM = limits.gpu_mem_gib
// else vram_total; RAM = limits.ram_max_gib else ram_total. This is the honest
// WHOLE the explicit-budget percentages resolve against (never a mixed
// denominator). Returns {bytes, gib, basis} — basis names what the whole is.
export const WP_GIB = 2 ** 30

// ── The FIVE flat allocation modes (k37, Slice B) ───────────────────────────
// Operator ask 2026-07-24: the alloc value is picked IN PLACE — a compact menu
// anchored at the click site. Four of the five modes are ZERO-KNOB (apply the
// instant they're picked); ONLY "explicit" opens extra chrome (its knobs).
//
// These are the operator-facing NAMES (managers/alloc_modes.ALLOC_MODES). The
// wire encoding is handled server-side (/assign runs normalize_spill), so the
// UI just sends {alloc_mode: <name>, …} and central rewrites the coarse trio
// onto the legacy n_gpu_layers wire — the UI never touches n_gpu_layers again.
//   gpu-only  all layers on the GPU, no spill (old console "Max GPU", -1)
//   ram-only  all in RAM, never the GPU              (old console "CPU only")
//   max-gpu   as much GPU as fits, spill the rest — THE DEFAULT (old "autofit")
//   max-ram   as much RAM as fits, spill the rest to GPU              (NEW)
//   explicit  target VRAM/RAM budgets + leniency%% + device priority  (NEW)
export const ALLOC_MODE_OPTIONS = [
  ['gpu-only', '🖥 GPU only', 'all layers on the GPU, no spill — won’t fit the GPU (after evict) → refused'],
  ['ram-only', '🧠 RAM only', 'all in host RAM, never the GPU (binds CPU even with a GPU present)'],
  ['max-gpu',  '⚡ Max GPU',  'as much GPU as fits, spill the rest to RAM — the DEFAULT (serves-and-spills, never OOMs)'],
  ['max-ram',  '💾 Max RAM',  'as much RAM as fits, spill the rest to the GPU'],
  ['explicit', '🎛 Explicit…', 'target VRAM/RAM budgets + a leniency %% + a device priority — the only mode with knobs'],
]

// Modes a non-GGUF (transformers/comfy) model may pick. The four non-explicit
// modes have a working non-GGUF meaning: the coarse trio rides accelerate, and
// max-ram was opened for non-GGUF 2026-07-24 (transformers RAM-priority
// max_memory, diffusers cpu-offload — Slice C wired the loaders). Only
// ``explicit`` stays GGUF-only (its banded leniency floor has no transformers
// analogue), so the menu DISABLES only explicit for a known non-gguf model.
// Mirrors alloc_modes.NONGGUF_ALLOWED_MODES.
export const NONGGUF_ALLOC_MODES = new Set(['gpu-only', 'ram-only', 'max-gpu', 'max-ram'])

export const GGUF_ONLY_MODE_TIP = 'explicit is GGUF-only — banded leniency has no transformers analogue'
