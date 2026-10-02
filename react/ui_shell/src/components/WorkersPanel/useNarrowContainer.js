import { useEffect, useState } from 'react'

// useNarrowContainer — true while the observed element is narrower than `px`.
// Degrades to false (= desktop layout) wherever ResizeObserver is missing, so
// an old browser gets the full table rather than a broken one.
export function useNarrowContainer(ref, px) {
  const [narrow, setNarrow] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el || typeof ResizeObserver === 'undefined') return undefined
    const ro = new ResizeObserver(entries => {
      for (const e of entries) {
        const w = e.contentRect ? e.contentRect.width : el.clientWidth
        setNarrow(w > 0 && w < px)
      }
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [ref, px])
  return narrow
}
