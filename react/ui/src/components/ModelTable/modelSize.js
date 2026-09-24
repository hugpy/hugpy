// A model's SIZE as every picker renders it (2026-09-23: "size must never be
// blank"). The /models feed carries size_bytes (effective quant for GGUF, the
// single-format footprint otherwise), size_source (marker / manifest /
// dir_walk / gguf_effective), size_note, and — when the server could not size
// it — size_reason. Nothing here guesses: unknown renders the recorded reason.

export function sizeBytesOf(m) {
  const s = m?.size_bytes != null ? m.size_bytes : m?.effective_bytes
  return s != null && Number.isFinite(Number(s)) ? Number(s) : null
}

// {text, title}: the bytes via `fmt`, or the server's reason when unknown.
export function sizeView(m, fmt) {
  const b = sizeBytesOf(m)
  if (b != null) {
    const src = m?.size_source ? ` (source: ${m.size_source})` : ''
    return { text: fmt(b), title: `${fmt(b)}${src}${m?.size_note ? ` — ${m.size_note}` : ''}`, known: true }
  }
  const why = m?.size_reason
    || (m?.status && m.status !== 'installed'
      ? `not installed on central (status ${m.status})`
      : 'the /models feed carries no size_bytes and no size_reason for this row (central predates size reasons)')
  return { text: `size unknown: ${why}`, title: why, known: false }
}
