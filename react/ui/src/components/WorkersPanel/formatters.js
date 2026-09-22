// midTrunc — MIDDLE ellipsis for a model name. Model keys differentiate at BOTH
// ends ("DavidAU~MN-GRAND-23.5B-…-NEO-Imatrix-GGUF" vs its non-NEO sibling), so
// a plain CSS tail-ellipsis hides exactly the distinguishing part. The budget is
// a CHARACTER count derived from the column (not a measured pixel width): cheap,
// deterministic, and stable across re-renders. No-ops when the name already fits
// (head + tail + 2 chars — never replaces 1–2 characters with a 1-char ellipsis).
// The FULL name always rides the cell's title, and the compact drawer shows it
// untruncated.
//
// BUDGET: the column caps at 220px; at the table's 12px font (~6.2px/char) less
// 16px of cell padding that is ~32 characters, so 12 + 1 + 14 = 27 sits inside
// it with margin and the CSS ellipsis backstop never fires. The TAIL is the
// longer half on purpose — the operator's own case is
// "…-NEO-Imatrix-GGUF" vs "…-Imatrix-GGUF", where the discriminator is ~16
// characters from the end; a 10-char tail would show "atrix-GGUF" for both.
export function midTrunc(name, head = 12, tail = 14) {
  const s = String(name ?? '')
  if (s.length <= head + tail + 2) return s
  return `${s.slice(0, head)}…${s.slice(-tail)}`
}

// Honest binary units (ruling 5, 2026-07-24): this formatter divides by 1024
// (binary), so the LABEL must be the binary unit (KiB/MiB/GiB), not the decimal
// SI one (KB/MB/GB). Before, a 45.09 GiB model (effective_bytes 48.41e9 —
// exactly the sum of coder-next's shards) rendered "45.1 GB", conflating the
// two universes: the number was GiB, the suffix said GB. The whole stack speaks
// GiB (effective quant, budgets, box caps all say "GiB"), so labeling the /1024
// math as GiB makes every call site (~80 of them) honest at once. The reported
// "41.6 GB" was NOT this formatter's output for the size cell — it is the model's
// MoE non-expert GPU-split figure (feasibility prices the card against that, not
// the full file); the size cell always showed the full effective_bytes, which is
// correct data, only mislabeled by one unit-suffix character each.
export function fmtBytes(n) {
  if (n == null) return '?'
  const u = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
  let v = Number(n), i = 0
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++ }
  return `${v.toFixed(1)} ${u[i]}`
}

// last-served relative time from the central per-(worker,model) `last_picked`
// epoch (seconds). null / 0 = never routed through central — the coldest, and
// exactly the never-served leftovers the eviction proposal frees first.
export function fmtServed(epoch) {
  if (!epoch) return 'never served'
  const secs = Math.max(0, Date.now() / 1000 - Number(epoch))
  if (secs < 45) return 'just now'
  const m = secs / 60
  if (m < 60) return `${Math.round(m)}m ago`
  const h = m / 60
  if (h < 24) return `${Math.round(h)}h ago`
  const d = h / 24
  if (d < 30) return `${Math.round(d)}d ago`
  return `${Math.round(d / 30)}mo ago`
}
