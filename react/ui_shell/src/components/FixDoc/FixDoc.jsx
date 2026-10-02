// FixDoc — a tiny reusable "how do I fix this?" affordance for warning/error
// labels. Renders a 📖 that click-throughs (new tab, so console state stays
// put) to the console Docs page at the anchor that documents the fix.
//
// Docs deep links are hash-routed (pages/Docs/Docs.jsx): /docs#<page> or
// /docs#<page>/<section>. Every entry below MUST point at an anchor that
// already exists in Docs.jsx (its NAV/TOC/section ids) — never invent one;
// a warning with no matching docs section stays unlinked (that absence is a
// docs gap to fill, not a link to fake).
//
// Doctrine (operator, 2026-07-04): a tagged warning/error must deep-link to
// the section that remedies IT SPECIFICALLY, not a nearby install/architecture
// tour. And convention forward: a NEW warning/error badge ships WITH its
// troubleshooting section + its FIX_DOCS entry, in the same PR — never add a
// badge that has no fix to point at.
import './FixDoc.css'

export const FIX_DOCS = {
  // GPU box whose llama-cpp-python is a CPU-only build (engine.supports_gpu_offload
  // === false). Points at the remedy: the Troubleshooting "CPU-only engine"
  // entry with the diagnostic one-liner, the two traps, the working CUDA/ROCm
  // rebuild recipes, and the verify step — NOT the fresh-install page.
  'engine-cpu-only': {
    href: '/docs#troubleshooting/engine-cpu-only',
    hint: 'CPU-only engine, VRAM idle — the diagnostic, the two traps, and the CUDA/ROCm rebuild that fixes it',
  },
  // Worker stuck pending/blocked at the admission gate. Points at the remedy
  // (Admit / enrollment token / unblock), not the architecture-tour callout.
  'worker-admission': {
    href: '/docs#worker-troubleshooting/admission',
    hint: 'Worker stuck pending/blocked — click Admit, or check the enrollment token / unblock it',
  },
  // The join command shows an address a worker box can't reach. The install
  // "Add workers" section IS the remedy (use an address the box can reach, not
  // localhost), so it stays targeted there.
  'worker-join': {
    href: '/docs#installation/workers',
    hint: 'Add workers — point hugpy worker at a central address the box can actually reach (not localhost)',
  },
  // Worker's local model cache is past its disk-cache ceiling (WorkersPanel
  // ⚠ over budget badge).
  'worker-over-budget': {
    href: '/docs#worker-troubleshooting/over-budget',
    hint: 'Worker over its disk-cache budget — approve the eviction proposal, raise the ceiling, or free disk',
  },
  // Central's /health ping to the worker failed (WorkersPanel ✗ unreachable badge).
  'worker-unreachable': {
    href: '/docs#worker-troubleshooting/unreachable',
    hint: 'Worker shows ✗ unreachable — agent running? port 9100 reachable? one agent per port? registered URL dialable?',
  },
  // Assigned model whose files are absent on the worker (WorkersPanel ✗ missing pill).
  'worker-model-missing': {
    href: '/docs#worker-troubleshooting/missing-files',
    hint: 'Model shows ○ missing — normal for an assigned-but-never-called model (lazy download: weights transfer on first call). What to check when it stays missing AFTER a call (disk, mmproj)',
  },
  // A panel's periodic poll of central failed (console ↔ central hop, not the workers).
  'registry-error': {
    href: '/docs#worker-troubleshooting/registry-error',
    hint: "Console shows registry error — the panel's poll of central failed; hover the chip, then check central/auth/proxy",
  },
}

export function fixDocsHref(key) {
  return FIX_DOCS[key]?.href || null
}

export default function FixDoc({ doc }) {
  const d = FIX_DOCS[doc]
  if (!d) return null
  return (
    <a className="fixdoc" href={d.href} target="_blank" rel="noreferrer"
       title={`Docs: ${d.hint}`}
       onClick={e => e.stopPropagation()}>
      📖
    </a>
  )
}
