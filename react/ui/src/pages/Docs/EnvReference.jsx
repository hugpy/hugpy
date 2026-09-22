// EnvReference — filterable, collapsible-by-subsystem wrapper for the
// "4 · Environment variables" reference (2026-07-15, operator-approved).
//
// WHY a wrapper (not a rewrite): the env reference is ~281 variable rows across
// 22 subsystem groups. We do NOT retype that content — each group's existing
// <table className="docs-table docs-table--env"> markup is passed VERBATIM as
// children of <EnvGroup>. Filtering reads the RENDERED DOM (tr.textContent) so
// no row text is ever transcribed, and defaults/purpose/values cells stay
// byte-identical to what ReferencePages.jsx already shipped.
//
// LOAD-BEARING CONSTRAINTS (see Docs.jsx):
//   * Deep-links + the right-hand TOC scrollspy key on the id="env-*" anchors.
//     Each EnvGroup keeps its id on the ALWAYS-RENDERED <h3> heading — only the
//     table body is collapsed — so #architecture/env-vision and the scrollspy
//     IntersectionObserver never lose their target.
//   * Expand-on-navigate: on mount and on every `hashchange`, if the URL hash's
//     section segment names a group id, that group is forced open and scrolled
//     into view. This piggybacks on the SAME hash mechanism Docs.jsx uses
//     (window.location.hash → "#slug/section"); we do not invent a parallel
//     router, we just read the same hash and react to the same event.
import { useState, useEffect, useRef, useCallback, Children, isValidElement, cloneElement } from 'react'

// Section segment of the docs hash, e.g. "#architecture/env-vision" -> "env-vision".
function hashSection() {
  const h = (window.location.hash || '').replace(/^#\/?/, '')
  const parts = h.split('/')
  return parts[1] || ''
}

/* ---- one collapsible subsystem group -------------------------------------
   Renders:  <h3 id> (always in the DOM, carries the anchor + is the toggle)
             <div>   collapsible region holding the verbatim <table> children
   Filtering: when `filter` is non-empty we walk the rendered <tbody> rows,
   hide the non-matching ones, and report the match count up via onCount. */
function EnvGroup({ id, label, defaultOpen, filter, forceOpenSignal, onCount, children }) {
  const [open, setOpen] = useState(!!defaultOpen)
  const regionRef = useRef(null)
  const headingRef = useRef(null)
  const regionId = `${id}-body`

  const filtering = filter.trim().length > 0
  const needle = filter.trim().toLowerCase()

  // Apply the filter against the rendered rows; returns the match count.
  const applyFilter = useCallback(() => {
    const region = regionRef.current
    if (!region) return 0
    const rows = region.querySelectorAll('tbody > tr')
    if (!filtering) {
      rows.forEach((r) => r.classList.remove('env-row-hidden'))
      return rows.length
    }
    let matched = 0
    rows.forEach((r) => {
      // A spanning note row (single colSpan cell) rides along with its group:
      // only counts as a "match" implicitly, never hides a real hit. We hide it
      // while filtering so the group shows just the variable hits.
      const isNote = r.querySelector('td[colspan]') && r.children.length === 1
      const hit = !isNote && (r.textContent || '').toLowerCase().includes(needle)
      if (hit) { matched += 1; r.classList.remove('env-row-hidden') }
      else { r.classList.add('env-row-hidden') }
    })
    return matched
  }, [filtering, needle])

  const [count, setCount] = useState(0)

  // Re-run filtering whenever the needle changes (and on first mount). We read
  // the DOM after render, which is exactly what a layout effect is for.
  useEffect(() => {
    const c = applyFilter()
    setCount(c)
    if (onCount) onCount(id, filtering ? c : null)
  }, [applyFilter, filtering, id])

  // Expand-on-navigate: when this group's id is targeted by the hash, open it
  // and scroll it into view. Bound to the SAME hashchange Docs.jsx listens to.
  useEffect(() => {
    const maybeOpen = () => {
      if (hashSection() === id) {
        setOpen(true)
        // let the region paint open before scrolling to the (stable) heading
        const el = headingRef.current
        if (el) setTimeout(() => el.scrollIntoView(), 80)
      }
    }
    maybeOpen()
    window.addEventListener('hashchange', maybeOpen)
    return () => window.removeEventListener('hashchange', maybeOpen)
  }, [id])

  // Expand/Collapse-all broadcasts a signal ({want:true|false, n}) from parent.
  useEffect(() => {
    if (forceOpenSignal && typeof forceOpenSignal.want === 'boolean') {
      setOpen(forceOpenSignal.want)
    }
  }, [forceOpenSignal])

  // While filtering, a group with matches is force-shown-open; a group with
  // zero matches is hidden entirely (heading included) — deep-links aren't
  // expected mid-filter, and this gives the "show only the hits" behavior.
  const hiddenByFilter = filtering && count === 0
  const effectiveOpen = filtering ? true : open

  if (hiddenByFilter) return null

  return (
    <div className="env-group" data-env-group={id}>
      <h3 id={id} ref={headingRef} className="docs-h3 env-group-h3">
        <button
          type="button"
          className="env-group-toggle"
          aria-expanded={effectiveOpen}
          aria-controls={regionId}
          onClick={() => !filtering && setOpen((v) => !v)}
          disabled={filtering}
          title={filtering ? 'Showing filtered matches' : (effectiveOpen ? 'Collapse' : 'Expand')}
        >
          <span className="env-group-caret" aria-hidden="true">{effectiveOpen ? '▾' : '▸'}</span>
          <span className="env-group-label">{label}</span>
          <span className="env-group-count">
            {filtering ? `${count} match${count === 1 ? '' : 'es'}` : `${count} variable${count === 1 ? '' : 's'}`}
          </span>
        </button>
      </h3>
      <div
        id={regionId}
        ref={regionRef}
        className="env-group-body"
        role="region"
        aria-labelledby={id}
        hidden={!effectiveOpen}
      >
        {children}
      </div>
    </div>
  )
}

/* ---- the whole section: filter box + expand/collapse-all + the groups ------
   Children are <EnvGroup> elements (defined inline in ReferencePages.jsx). We
   clone them to inject the shared `filter` + expand/collapse signal, and to
   collect per-group match counts for the running total. */
export default function EnvReference({ children }) {
  const [filter, setFilter] = useState('')
  const [signal, setSignal] = useState(null)     // { want:boolean, n } broadcast to groups
  const [counts, setCounts] = useState({})        // id -> match count (only while filtering)

  const filtering = filter.trim().length > 0

  const onCount = useCallback((id, c) => {
    setCounts((prev) => {
      if (c === null) { if (!(id in prev)) return prev; const n = { ...prev }; delete n[id]; return n }
      if (prev[id] === c) return prev
      return { ...prev, [id]: c }
    })
  }, [])

  // When the filter clears, drop the counts map so default collapse state
  // (all-collapsed-except-first, decided per-group by defaultOpen) is restored.
  useEffect(() => { if (!filtering) setCounts({}) }, [filtering])

  const groupList = Children.toArray(children).filter(isValidElement)
  const groupCount = groupList.length
  const kids = groupList.map((child, i) =>
    cloneElement(child, {
      filter,
      forceOpenSignal: signal,
      onCount,
      // default collapse state: all collapsed EXCEPT the first group
      defaultOpen: child.props.defaultOpen ?? i === 0,
    })
  )

  const totalMatches = filtering
    ? Object.values(counts).reduce((a, b) => a + b, 0)
    : null
  const matchedGroups = filtering ? Object.values(counts).filter((c) => c > 0).length : null

  return (
    <div className="env-ref">
      <div className="env-ref-toolbar">
        <label className="env-ref-search">
          <span className="env-ref-search-icon" aria-hidden="true">⌕</span>
          <input
            type="search"
            className="env-ref-input"
            placeholder="Filter variables — name, purpose or values…"
            aria-label="Filter environment variables by name, purpose or values"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
          {filtering && (
            <button
              type="button"
              className="env-ref-clear"
              onClick={() => setFilter('')}
              aria-label="Clear filter"
              title="Clear filter"
            >×</button>
          )}
        </label>
        <div className="env-ref-actions">
          {filtering ? (
            <span className="env-ref-summary" role="status">
              {totalMatches} match{totalMatches === 1 ? '' : 'es'} in {matchedGroups} group{matchedGroups === 1 ? '' : 's'}
            </span>
          ) : (
            <>
              <button type="button" className="env-ref-btn" onClick={() => setSignal({ want: true, n: Date.now() })}>
                Expand all
              </button>
              <button type="button" className="env-ref-btn" onClick={() => setSignal({ want: false, n: Date.now() })}>
                Collapse all
              </button>
              <span className="env-ref-summary env-ref-summary--muted">{groupCount} groups</span>
            </>
          )}
        </div>
      </div>
      {filtering && totalMatches === 0 && (
        <p className="env-ref-empty docs-note">No variables match “{filter.trim()}”.</p>
      )}
      {kids}
    </div>
  )
}

export { EnvGroup }
