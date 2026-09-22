// Protection/eviction badge for a per-model storage row. Reuses the Serving-row
// state-pill palette. Protected models CANNOT be deleted; evictable ones can,
// and any in the current proposal are flagged for freeing.
export function storageBadge(m, proposed) {
  // REFUSED (storage): the pull never started — even a full FIFO of the cold,
  // unprotected models couldn't free enough room under this worker's budget.
  // Doctrine "defaults are promises": a model that cannot fit must read as
  // MISSING with a reason, never as a pull stuck at 7%. `title` carries the
  // worker's own honest reason string (needs / budget / reclaimable / blocked).
  if (m.refused) {
    return {
      pill: 'wp-pill-refused', glyph: '⊘ missing',
      title: `won't fit on this worker — the download was REFUSED before it `
        + `started (nothing was deleted, no partial file). ${m.refused.reason || ''}`
        + `\n\nRaise this worker's storage allocation (disk_cache_gib), free a `
        + `protected model, or route this model to another box.`,
    }
  }
  if (proposed) return { pill: 'wp-pill-evict', glyph: '🗑 will free', title: 'in the eviction proposal — freed on approval. 📌 A pinned model can appear here: pin keeps its allocation/routing, not its files.' }
  // SHARED CATALOG / UNREAPABLE STORE (k60, 2026-07-31). These bytes are on a
  // store this worker may NEVER delete from, so they are shown but charged to
  // nothing: they contribute 0 to used/over-budget and can never be proposed.
  // Checked ahead of every other state — it is a filesystem fact, not a policy
  // label, and it outranks whatever else the row happens to be.
  if (m.store === 'shared') return { pill: 'wp-pill-shared', glyph: '🔗 shared', title: 'on the SHARED central catalog (the fleet\'s source-of-truth copies, read through from here). Never evicted from this worker and never counted against its storage budget — it is not this box\'s cache.' }
  if (m.store === 'unreapable') return { pill: 'wp-pill-shared', glyph: '🔗 unreapable store', title: 'on a model store this box has not declared local & disposable (HUGPY_MODEL_STORE_REAPABLE unset), so nothing here can be reaped. Shown for visibility; never counted against this worker\'s storage budget.' }
  // NOTE (2026-07-17): 📌 pin no longer protects files, so the PROTECTIVE
  // states are checked FIRST — a pinned model that is also static/loaded/…
  // shows that (real) protection. A bare pinned model falls through to the
  // attribution-only badge below (protected:false → still a candidate).
  if (m.why === 'static') return { pill: 'wp-pill-loaded', glyph: '🔒 static', title: 'static residency — a locked seat, never evicted, files kept on disk (the only tier that blocks eviction/reaping)' }
  if (m.loaded) return { pill: 'wp-pill-serving', glyph: '🔥 loaded', title: 'resident/serving right now — protected' }
  if (m.loading) return { pill: 'wp-pill-heating', glyph: '🔶 heating', title: 'weights loading — protected' }
  // Only a GENUINELY LIVE pull renders as a transfer. Central derives this
  // read-side (owner alive AND bytes moving) — a dead-owner/stalled entry
  // never reaches us as `provisioning`, so it can't show a phantom ⏳ forever.
  if (m.provisioning) return { pill: 'wp-pill-pulling', glyph: '⏳ pulling', title: 'files transferring right now — protected while the bytes land' }
  if (m.assigned) return { pill: 'wp-pill-idle', glyph: '📎 assigned', title: 'designated to this worker — protected in the operator-gated bulk reaper' }
  if (m.pinned) return { pill: 'wp-pill-loaded', glyph: '📌 pinned', title: 'this allocation survives restarts (routing to this worker is durable). Does NOT download the model and does NOT protect its files from eviction — bytes arrive on call and can be evicted to make room (routing is unaffected). Only 🔒 static keeps files on disk.' }
  if (m.protected) return { pill: 'wp-pill-idle', glyph: '🛡 protected', title: m.why || 'protected' }
  return { pill: 'wp-pill-cold', glyph: '○ evictable', title: 'on disk, unassigned & cold — reclaimable' }
}
