// Sharded serving (multi-GPU across machines) — pool workers' GPUs to serve one
// GGUF model too big for any single card, via llama.cpp RPC. Operator + dev page.
//
// The mechanism: a lead `llama-server --rpc <backends> --tensor-split <split>`
// distributes the model's layers across the lead's local GPU and one or more
// remote `ggml-rpc-server` backends. Central's allocator (managers/resolvers/
// allocator.py) decides when to shard; the lead is spawned by
// managers/llama/runners/src/shard_server.py:ensure_shard_server.

export const SHARDED_TOC = [
  ['shard-overview', 'Overview', []],
  ['shard-req', 'Requirements', []],
  ['shard-how', 'How it works', []],
  ['shard-setup', 'Setup', [['shard-engine', 'Build the engine'], ['shard-backends', 'Run RPC backends']]],
  ['shard-optin', 'Opt a model in', []],
  ['shard-use', 'Using it', [['shard-auto', 'Automatic (allocator)'], ['shard-direct', 'Direct']]],
  ['shard-caveats', 'Caveats & gotchas', []],
  ['shard-trouble', 'Troubleshooting', []],
]

const ENGINE_CODE = `hugpy install-engine --cuda
# or programmatically:
python -c "from abstract_hugpy_dev.engine.build import build_from_source; build_from_source(cuda=True)"`

const BACKEND_CODE = `worker-agent --role rpc --rpc-port 50052`

const OPTIN_CODE = `HUGPY_SHARD_MODELS=Qwen3.6-40B-...-GGUF:40gb,Some-70B-GGUF:42gb`

const AUTO_CODE = `curl -sN https://dev.hugpy.ai/api/v1/chat/completions \\
  -H "Authorization: Bearer $HUGPY_API_KEY" -H "Content-Type: application/json" \\
  -d '{"model":"Some-70B-GGUF","messages":[{"role":"user","content":"hi"}]}'`

const DIRECT_CODE = `HUGPY_RPC_SERVERS=192.168.1.128:50052
HUGPY_TENSOR_SPLIT=0.1,0.9   # order = [rpc-backends..., lead]`

export default function ShardedServingPage() {
  return (
    <article className="docs-article">
      <h1>Sharded serving (multi-GPU across machines)</h1>

      <section id="shard-overview" className="docs-sec">
        <p>
          <strong>Sharded serving</strong> pools the GPUs of several workers to serve a
          single model that is too large for any one card. The model's layers are split
          across a <em>lead</em> worker's local GPU and one or more remote{' '}
          <code>ggml-rpc-server</code> backends over the LAN, using llama.cpp's RPC
          backend. The result is served as one ordinary OpenAI endpoint.
        </p>
        <p>
          Central's deterministic allocator decides when to shard (only when a model
          doesn't fit a single GPU), picks the lead + backends by free VRAM, and routes
          the request to the lead. It is <strong>off by default</strong> — a model shards
          only when opted in via <code>HUGPY_SHARD_MODELS</code>. Planning is
          evict-to-fit-aware: it counts VRAM reclaimable from cold resident models, so a
          busy fleet still shards without a manual evict.
        </p>
      </section>

      <section id="shard-req" className="docs-sec">
        <h2 className="docs-h2">Requirements</h2>
        <ul>
          <li><strong>GGUF only.</strong> This is a llama.cpp feature; it shards GGUF models. Transformers/diffusers models (safetensors) cannot use this path.</li>
          <li><strong>Dense models shard across GPUs; MoE prefers CPU-offload.</strong> A Mixture-of-Experts GGUF (e.g. "A3B") offloads idle experts to host RAM rather than a remote GPU, so it often fits the lead alone. Use a dense model to exercise cross-GPU sharding.</li>
          <li>An RPC-capable engine on each node: <code>llama-server</code> (lead) and <code>ggml-rpc-server</code> (backends), built with <code>-DGGML_RPC=ON</code> (and <code>-DGGML_CUDA=on</code> for GPU).</li>
          <li>Backends registered as <code>role=rpc</code> so the allocator sees their <code>rpc_endpoint</code> and free VRAM.</li>
          <li>Fast interconnect. Per-token activations cross the network, so it is bandwidth-bound — worth it to run a model that otherwise won't fit, not to speed up one that already does.</li>
        </ul>
      </section>

      <section id="shard-how" className="docs-sec">
        <h2 className="docs-h2">How it works</h2>
        <ol>
          <li>Central's allocator (<code>managers/resolvers/allocator.py</code>) sees a shard-eligible model exceed every single GPU (counting evict-to-fit-reclaimable VRAM), and returns a <code>shard</code> placement: a lead + the fewest backends whose summed free VRAM covers it, plus a VRAM-proportional <code>tensor_split</code>.</li>
          <li><code>placement_for_model</code> emits that as a per-request spill (<code>rpc_servers</code> + <code>tensor_split</code>); <code>remote._select</code> routes to the lead.</li>
          <li>The lead worker's <code>_apply_spill</code> sets <code>HUGPY_RPC_SERVERS</code> / <code>HUGPY_TENSOR_SPLIT</code>, which <code>get_llama_runner</code> reads to call <code>ensure_shard_server</code> — spawning <code>llama-server --rpc ... --tensor-split ... -ngl 999</code> and serving over HTTP.</li>
        </ol>
      </section>

      <section id="shard-setup" className="docs-sec">
        <h2 className="docs-h2">Setup</h2>
        <div id="shard-engine" className="docs-h3-block">
          <h3 className="docs-h3">Build the engine</h3>
          <p>Build llama.cpp with RPC (and CUDA) on each node. hugpy's own builder does this and installs it where the resolver finds it:</p>
          <pre><code>{ENGINE_CODE}</code></pre>
          <p>It produces <code>llama-server</code> + <code>ggml-rpc-server</code> under <code>engine_dir()/build/bin</code> and persists the location, so <code>server_bin()</code> resolves it with no restart.</p>
        </div>
        <div id="shard-backends" className="docs-h3-block">
          <h3 className="docs-h3">Run RPC backends</h3>
          <p>On each backend node, run the worker agent in RPC-backend mode so it launches <code>ggml-rpc-server</code> and registers <code>rpc_endpoint</code> + free VRAM:</p>
          <pre><code>{BACKEND_CODE}</code></pre>
          <p>The allocator pools a node as a backend once it appears in the registry with <code>role=rpc</code> and non-zero free VRAM.</p>
        </div>
      </section>

      <section id="shard-optin" className="docs-sec">
        <h2 className="docs-h2">Opt a model in</h2>
        <p>
          Set <code>HUGPY_SHARD_MODELS</code> on central to a comma list of{' '}
          <code>key:bytes</code> (a <code>g</code>/<code>gb</code> suffix is allowed). Only listed models
          may shard, and only when the allocator finds they don't fit one GPU:
        </p>
        <pre><code>{OPTIN_CODE}</code></pre>
        <p>Restart central to pick up the env (<code>sudo systemctl restart 7002_hugpy_api</code>). A model that fits a single card routes <em>whole</em>, never sharded.</p>
      </section>

      <section id="shard-use" className="docs-sec">
        <h2 className="docs-h2">Using it</h2>
        <div id="shard-auto" className="docs-h3-block">
          <h3 className="docs-h3">Automatic (allocator)</h3>
          <p>With the model opted in and backends registered, just call the API — central shards it transparently:</p>
          <pre><code>{AUTO_CODE}</code></pre>
        </div>
        <div id="shard-direct" className="docs-h3-block">
          <h3 className="docs-h3">Direct</h3>
          <p>To force a specific shard plan (bypassing the allocator), set the spill env on the serving process:</p>
          <pre><code>{DIRECT_CODE}</code></pre>
          <p>Any GGUF request then loads via <code>ensure_shard_server</code>.</p>
        </div>
      </section>

      <section id="shard-caveats" className="docs-sec">
        <h2 className="docs-h2">Caveats &amp; gotchas</h2>
        <ul>
          <li><strong>tensor-split order is <code>[rpc-backends..., lead]</code></strong> — RPC devices first, the lead last. A small backend GPU must get a small share first, e.g. <code>0.1,0.9</code>.</li>
          <li><strong>One client per backend.</strong> <code>ggml-rpc-server</code> serves a single lead at a time; a stale connection blocks a new lead. Restart the backend to clear it.</li>
          <li><strong>MoE self-selects CPU-offload</strong> — an A3B/MoE GGUF spills experts to host RAM, so it may not use the remote GPU at all. Use a dense model for true cross-GPU sharding.</li>
          <li><strong>LAN-bandwidth bound.</strong> Big models over gigabit are slow; the payoff is running a model that otherwise can't load at all.</li>
        </ul>
      </section>

      <section id="shard-trouble" className="docs-sec">
        <h2 className="docs-h2">Troubleshooting</h2>
        <ul>
          <li><code>failed to allocate RPC0[...] buffer</code> — the backend GPU is out of VRAM; give it a smaller (first) tensor-split share.</li>
          <li>Lead never becomes healthy / hangs at load — a stale connection is holding the backend's single slot; restart <code>ggml-rpc-server</code>.</li>
          <li><code>no llama-server binary</code> — run <code>hugpy install-engine</code>; confirm <code>server_bin()</code> resolves it.</li>
          <li>Model routes whole instead of sharding — it fits one GPU, isn't in <code>HUGPY_SHARD_MODELS</code>, no <code>role=rpc</code> backend is registered, or it's MoE (CPU-offloads).</li>
        </ul>
      </section>
    </article>
  )
}
