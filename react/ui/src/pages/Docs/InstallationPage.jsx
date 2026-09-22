// Run it yourself — installation, edited for plain language with "you should
// see" checkpoints. Section ids (requirements/install/engine/run/storage/
// workers) are load-bearing: FixDoc deep-links #installation/engine and
// #installation/workers. Keep them.

export const INSTALL_TOC = [
  ['requirements', 'What you need', []],
  ['install', 'Install the package', []],
  ['engine', 'Build the engine', []],
  ['run', 'Run the server', []],
  ['storage', 'Where things live', []],
  ['workers', 'Add workers', []],
]

export default function InstallationPage() {
  return (
    <>
      <div className="docs-crumbs">Run it yourself <span>/</span> Installation</div>
      <h1>Installation</h1>
      <p className="docs-lede">
        Install hugpy, build the inference engine, and point it at your storage and GPUs. The
        console ships prebuilt inside the package — production needs no Node toolchain.
      </p>

      <section id="requirements" className="docs-sec">
        <h2>What you need</h2>
        <ul className="docs-routes">
          <li><strong>Python 3.10 or newer</strong> — one package runs the server, workers, bot, and CLI.</li>
          <li><strong>An NVIDIA GPU with CUDA</strong> <em>(optional)</em> — for fast inference and for
            splitting oversized models across machines. CPU-only works for smaller GGUF models.</li>
          <li><strong>No Node.js</strong> — the console is prebuilt inside the package and served by the
            same process.</li>
        </ul>
      </section>

      <section id="install" className="docs-sec">
        <h2>Install the package</h2>
        <pre className="docs-block"><code>{`pip install hugpy`}</code></pre>
        <p>
          The base install is deliberately tiny (no torch, no heavy ML libraries) so it installs
          anywhere — even a phone. The heavy pieces live in extras; install what the machine will
          actually do:
        </p>
        <table className="docs-table">
          <thead><tr><th>Extra</th><th>Adds</th></tr></thead>
          <tbody>
            <tr><td><code>[engine]</code></td><td>the in-process GGUF engine (llama-cpp-python)</td></tr>
            <tr><td><code>[server]</code></td><td>everything a full central server needs (engine + bot + GPU tools)</td></tr>
            <tr><td><code>[bot]</code></td><td>the Discord arm</td></tr>
            <tr><td><code>[transformers]</code> / <code>[gpu]</code> / <code>[embed]</code></td><td>torch+transformers / GPU monitoring / embeddings</td></tr>
            <tr><td><code>[vision]</code> / <code>[ocr]</code> / <code>[web]</code></td><td>image analysis / OCR / scraping</td></tr>
            <tr><td><code>[all]</code></td><td>everything — the full desktop/server install</td></tr>
          </tbody>
        </table>
        <p>
          <strong>Checkpoint:</strong> <code>hugpy --help</code> prints the command list. If it doesn&apos;t,
          the install went into a different Python than the one on your PATH.
        </p>
      </section>

      <section id="engine" className="docs-sec">
        <h2>Build the inference engine</h2>
        <p>
          The engine is the native llama.cpp binary set that actually runs GGUF models. One command
          fetches or builds it (with GPU support when you pass <code>--cuda</code>):
        </p>
        <pre className="docs-block"><code>{`pip install 'hugpy[engine]'    # in-process GGUF engine
hugpy install-engine --cuda    # native llama.cpp + rpc-server`}</code></pre>
        <p>
          <strong>Checkpoint:</strong> the build ends by reporting where the binaries landed;
          <code> llama-server --version</code> from that <code>build/bin</code> directory should run.
          The build also produces <code>rpc-server</code>, which is only needed if this box will lend
          its GPU to the cross-machine shard pool (a failure at that final target is usually
          cosmetic — see <a className="docs-inline-link" href="#worker-troubleshooting/install-engine-rpc">the
          fix-it entry</a>).
        </p>
        <p className="docs-note">
          Pin a specific llama.cpp release with <code>HUGPY_ENGINE_REPO</code> / <code>HUGPY_ENGINE_TAG</code>.
        </p>
      </section>

      <section id="run" className="docs-sec">
        <h2>Run the server</h2>
        <pre className="docs-block"><code>{`hugpy serve --port 7002`}</code></pre>
        <p>
          <strong>You should see</strong> startup logs naming the server it chose (gunicorn, waitress,
          or the dev server) and the console answering at <code>http://&lt;this-box&gt;:7002</code>. One
          process serves both the API and the console — no nginx required.
        </p>
      </section>

      <section id="storage" className="docs-sec">
        <h2>Where things live</h2>
        <p>
          Model weights, the registry, download jobs, and API keys all live under one storage root.
          Set it before first run if you have a big disk for models:
        </p>
        <table className="docs-table">
          <thead><tr><th>Env var</th><th>Holds</th></tr></thead>
          <tbody>
            <tr><td><code>DEFAULT_ROOT</code> <span className="docs-note">(legacy <code>MODELS_HOME</code>)</span></td><td>The storage root — weights, registry, uploads, jobs, keys (falls back to <code>/mnt/llm_storage</code>, then a per-user data dir)</td></tr>
            <tr><td><code>HUGPY_DATA_DIR</code></td><td>Per-OS user-data dir</td></tr>
            <tr><td><code>HUGPY_CONFIG_DIR</code></td><td>Config + state (placement, bindings, toggles)</td></tr>
            <tr><td><code>HUGPY_CACHE_DIR</code></td><td>Per-OS cache dir</td></tr>
            <tr><td><code>HUGPY_ENGINE_DIR</code> <span className="docs-note">(or <code>LLAMA_CPP_DIR</code>)</span></td><td>The built / fetched engine binaries</td></tr>
          </tbody>
        </table>
      </section>

      <section id="workers" className="docs-sec">
        <h2>Add workers</h2>
        <p>
          The easy path is the console&apos;s one-line install command
          (<a className="docs-inline-link" href="#console-workers/join">Put models on workers</a>). By
          hand, it&apos;s the same two commands on each box — the address after <code>--central</code> must
          be one <em>that box</em> can reach (not <code>localhost</code>):
        </p>
        <pre className="docs-block"><code>{`pip install hugpy
hugpy worker --central http://your-hugpy:7002   # full GPU worker
hugpy worker --role rpc                          # shard backend only`}</code></pre>
        <p>
          <strong>You should see</strong> the box appear in the console&apos;s Compute tab as
          <em> pending</em> within seconds — click <strong>✓ admit</strong> to let it serve. Workers
          keep themselves up to date from central afterwards (self-update on the version central
          advertises; point air-gapped workers at central&apos;s own package index with
          <code> --pkg-index &lt;central&gt;/api/llm/pip/simple</code>).
        </p>
      </section>

      <div className="docs-pager">
        <a className="docs-pager-card" href="#worker-troubleshooting"><span className="d">Next</span><span className="t">Worker Troubleshooting →</span></a>
      </div>
    </>
  )
}
