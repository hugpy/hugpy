// Workers that HOLD this model (catalog workers[] with bytes on disk), with
// online/hot state. Only these can take an explicit pin.
export function holdingWorkers(model, roster = []) {
  const hot = new Set(model?.hot_workers || [])
  const held = Array.isArray(model?.workers) ? model.workers : []
  const byId = new Map(held.filter(w => w?.worker_id).map(w => [String(w.worker_id), w]))
  const byName = new Map(held.filter(w => w?.worker).map(w => [w.worker, w]))
  const source = Array.isArray(roster) && roster.length
    ? roster.map(w => {
      const h = byId.get(String(w.id)) || byName.get(w.name) || {}
      return { name: w.name, id: w.id || '', online: w.status === 'online',
        hot: hot.has(w.name) || !!h.loaded || !!h.serving,
        onDisk: Number(h.on_disk_bytes) > 0 }
    })
    : held.filter(w => w?.worker).map(w => ({ name: w.worker, id: w.worker_id || '',
      online: w.status === 'online', hot: hot.has(w.worker) || !!w.loaded || !!w.serving,
      onDisk: Number(w.on_disk_bytes) > 0 }))
  return source.filter(w => w.name)
    .sort((a, b) => (b.hot - a.hot) || (b.online - a.online) || a.name.localeCompare(b.name))
}
