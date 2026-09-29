# Worker host model inventory

The worker scans the host filesystem in a background thread on its first
heartbeat and every 15 minutes. The scan excludes virtual and transient trees
such as `/proc`, `/sys`, `/dev`, `/run`, and Hugpy's managed model directory.
Set `HUGPY_WORKER_MODEL_SCAN_ROOTS` to a colon-separated list to limit the
search; the default is `/`.

A complete model outside Hugpy's model directory gets a `hugpy.json` pointer
under `~/.hugpy/worker-models/<path-hash>/hugpy.json`. Set
`HUGPY_WORKER_MODEL_INVENTORY_DIR` to relocate these metadata files. A separate
A pointer records the model identity and its absolute `location`; it does not
copy or alter the weights. A `hugpy.json` beside the weights, if present,
supplies their declared identity and task. A name collision with another model
is qualified by owner.

The scan also recognizes Ollama manifests and their digest-named blobs. It
checks the manifest digest against the local Ollama `/api/tags` response,
verifies that every referenced blob is readable and has the declared size,
and offers models with the `completion` capability. Set `HUGPY_OLLAMA_URL`
if the local API is not at `http://127.0.0.1:11434`. These models use the
worker's Ollama adapter for chat inference and grading. Ollama's `/api/ps`
supplies loaded state and VRAM use; eviction calls Ollama with `keep_alive=0`
and leaves the manifest and blobs intact.

The worker loads an external model from its recorded location. Its heartbeat
publishes the discovered models and all local model keys. Central adds
worker-owned models to the ordinary catalog, grading, and placement views and
routes them only to a worker that reports their files. If Central has no
complete copy, no cold transfer or grading cold reset is attempted. The
worker's Hugpy model directory remains the only disk store eligible for
reaping; external paths may still be unloaded from memory on demand.
