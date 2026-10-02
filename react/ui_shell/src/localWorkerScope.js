// The copied development console deliberately presents only the worker on
// this host. Requests and operations still go to the real central Python API.
export const LOCAL_WORKER_NAME = 'ae-worker'

export function localWorkerRoster(rows) {
  return Array.isArray(rows)
    ? rows.filter(row => row?.name === LOCAL_WORKER_NAME)
    : rows
}
