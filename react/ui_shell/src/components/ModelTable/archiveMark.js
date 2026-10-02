import { fmtAgo } from './modelStatus.js'

// The operator's ARCHIVE MARK on a catalog row — `archived: {marked, at, by,
// reason}` from /models (hugpy.json "archive"; server: hugpy_storage.archive_mark).
// A marked model is refused by every placement write and by the resolver, so
// every picker greys it with the SAME text, built only from the recorded facts.

export function archiveMark(m) {
  const a = m?.archived
  return a && typeof a === 'object' && a.marked ? a : null
}

export function archiveText(a) {
  if (!a) return ''
  const t = a.at ? Date.parse(a.at) : NaN
  const when = Number.isNaN(t) ? `at ${a.at}` : fmtAgo(t / 1000)
  return `marked for archive by ${a.by} ${when}`
    + (a.reason ? `: ${a.reason}` : ' (no reason was given when it was marked)')
}
