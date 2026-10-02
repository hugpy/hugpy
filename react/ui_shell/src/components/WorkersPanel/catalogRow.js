// Map a WORKER's model key to the catalog row from /api/models.
//
// Why this is not a plain equality (2026-09-10): discovery keys a model by its
// bare directory name and only qualifies it as `owner~name` while two owners
// ship the same name. That qualification comes and goes as copies appear and
// disappear (an _archive copy, a sibling quant), but a worker's ASSIGNMENT keeps
// whichever spelling it was given. On ae 29 of 95 assigned keys were
// `owner~name` while the catalog had gone back to `name` — every one of those
// rows rendered as dashes ("—  —  —") because nothing matched exactly, even
// though the model was on disk and fully described in the catalog. Both
// spellings derive from hub_id `owner/name`, so resolve through that.
export function findCatalogRow(models, key) {
  if (!key || !Array.isArray(models) || !models.length) return undefined
  const keyOf = m => m.model_key ?? m.key
  let m = models.find(mm => keyOf(mm) === key)
  if (m) return m
  const at = key.indexOf('~')
  if (at > 0) {
    // owner~name → the row whose hub_id is owner/name, else the unique bare name
    const asHub = key.slice(0, at) + '/' + key.slice(at + 1)
    m = models.find(mm => mm.hub_id === asHub)
    if (m) return m
    const bare = key.slice(at + 1)
    const bares = models.filter(mm => keyOf(mm) === bare)
    return bares.length === 1 ? bares[0] : undefined
  }
  // bare name → the unique owner-qualified row, or the unique hub_id ending in /name
  const qualified = models.filter(mm => {
    const k = keyOf(mm) || ''
    return k.endsWith('~' + key) || (mm.hub_id || '').endsWith('/' + key)
  })
  return qualified.length === 1 ? qualified[0] : undefined
}
