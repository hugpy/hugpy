// Workers that HOLD this model (catalog workers[] with bytes on disk), with
// online/hot state. Only these can take an explicit pin.
export function holdingWorkers(model) {
  const hot = new Set(model?.hot_workers || [])
  return (Array.isArray(model?.workers) ? model.workers : [])
    .filter(w => w && w.worker && Number(w.on_disk_bytes) > 0)
    .map(w => ({ name: w.worker, id: w.worker_id || '', online: w.status === 'online',
      hot: hot.has(w.worker) || !!w.loaded || !!w.serving }))
    .sort((a, b) => (b.hot - a.hot) || (b.online - a.online) || a.name.localeCompare(b.name))
}

