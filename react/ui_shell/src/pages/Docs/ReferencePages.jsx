// Reference — the architecture tour and CLI pages, demoted from the docs
// front door (2026-07-05) but kept intact: developers and integrators still
// need them. Section ids are load-bearing (FixDoc links
// #architecture/admission; old deep links hit #architecture/* and #cli/*).
import { CodeTabs, CopyBlock, C, S, K } from './docParts'
import EnvReference, { EnvGroup } from './EnvReference'

export const ARCH_TOC = [
  ['what', '1 · What hugpy is', []],
  ['architecture', '2 · The architecture', []],
  ['sections', '3 · Section-by-section', [
    ['flask-central', 'Flask central API'],
    ['managers', 'Managers · execution'],
    ['worker-fleet', 'Worker fleet & pools'],
    ['pools', 'Dedicated worker pools'],
    ['admission', 'Admission & enrollment'],
    ['discord-bot', 'Discord bot'],
    ['react-console', 'React console'],
  ]],
  ['env', '4 · Environment variables', [
    ['env-central', 'Central address'],
    ['env-auth', 'Auth & security'],
    ['env-comms', 'Comms, jobs & settings storage'],
    ['env-serving', 'Serving posture, residency & slots'],
    ['env-spill', 'GPU/CPU offload & sharding'],
    ['env-storage', 'Model storage & cache tiers'],
    ['env-discovery', 'Discovery, Hugging Face & Civitai'],
    ['env-model-defaults', 'Per-task model defaults & overrides'],
    ['env-dispatch', 'Chat/dispatch continuation & streaming'],
    ['env-uploads', 'Uploads, CORS & downloads'],
    ['env-vision', 'Vision, image-gen & ComfyUI'],
    ['env-pools', 'ML task routing & worker pools'],
    ['env-worker-identity', 'Worker — identity & heartbeat'],
    ['env-worker-update', 'Worker — self-update & enrollment'],
    ['env-worker-caps', 'Worker — resource caps & lifecycle'],
    ['env-engine', 'Native engine (llama.cpp)'],
    ['env-platform', 'Cross-platform paths & gguf_worker'],
    ['env-phonebrick', 'Phone-brick fleet'],
    ['env-studio', 'Identity-render & Studio video'],
    ['env-misc', 'Discord bot, keeper REPL & peers'],
    ['env-internal', 'Internal schema constants'],
    ['env-frontend', 'React console apps (build-time)'],
  ]],
]

export const CLI_TOC = [
  ['serve', 'serve', []],
  ['worker', 'worker', []],
  ['bot', 'bot', []],
  ['install-engine', 'install-engine', []],
  ['keeper', 'keeper', []],
]

/* =====================================================================
   PAGE · Architecture tour
   ===================================================================== */
export function ArchitecturePage() {
  return (
    <>
      <div className="docs-crumbs">Reference <span>/</span> Architecture tour</div>
          <h1>Architecture tour</h1>
          <p className="docs-lede">
            A comprehensive map of the hugpy project: what each section does and how the pieces tie
            together.
          </p>
          <p className="docs-note">
            This page is for developers and integrators working on or against hugpy&apos;s internals.
            Using the console day-to-day needs none of it — start at{' '}
            <a className="docs-inline-link" href="#quickstart">Start here</a> instead.
          </p>

          <section id="what" className="docs-sec">
            <h2>1 · What hugpy is</h2>
            <p>
              hugpy is a <strong>self-hosted LLM platform</strong>: a single-process product that bundles a
              model registry, downloader, streaming chat, an OpenAI-compatible <code>/v1</code> API with
              on-site API keys, and a <strong>distributed GPU / mobile worker fleet</strong> with cross-machine
              GGUF sharding. You run your own models, on your own boxes, behind one console.
            </p>

            <CodeTabs tabs={[
              { label: 'Shell', lang: 'bash', code: <>{C('# console + API in one server process')}{'\n'}pip install hugpy{'\n'}hugpy serve --port {K('7002')}{'\n\n'}{C('# join a GPU box to the fleet')}{'\n'}hugpy worker --central {S('http://your-hugpy:7002')}</> },
              { label: 'Python', lang: 'python', code: <>{C('# OpenAI-compatible — point any SDK at your box')}{'\n'}{K('from')} openai {K('import')} OpenAI{'\n\n'}client = OpenAI({'\n'}    base_url={S('"http://localhost:7002/api/v1"')},{'\n'}    api_key={S('"hp_…"')},  {C('# minted in the console; or open mode')}{'\n'}){'\n\n'}client.chat.completions.create(model={S('"…"')}, messages=[…], stream={K('True')})</> },
            ]} />

            <h3 className="docs-h3-plain">Mono-repo layout</h3>
            <table className="docs-table">
              <thead><tr><th>Path</th><th>What it is</th></tr></thead>
              <tbody>
                <tr><td><code>api/</code></td><td>The <code>hugpy</code> Python package — Flask “central” server, model-execution layer, worker agents, Discord bot, CLI. Published to PyPI (dev distribution: <code>abstract_hugpy_dev</code>).</td></tr>
                <tr><td><code>ui/</code></td><td>React console — also published as the <code>@hugpy/ui</code> npm package. Built into <code>console_dist/</code> and shipped <em>inside the wheel</em>, so a production install needs no Node toolchain.</td></tr>
              </tbody>
            </table>
            <p>
              Deployment is deliberately simple: <strong>one server process</strong> (<code>hugpy serve</code>)
              serves the API <em>and</em> the built React console at the same origin (<code>:7002</code>). No
              nginx or separate webpack is required in production.
            </p>
          </section>

          <section id="architecture" className="docs-sec">
            <h2>2 · The architecture — how it ties together</h2>
            <p>
              Everything is <strong>hub-and-spoke around the Flask “central” server.</strong> Central owns the
              model registry, the on-disk weights, the worker registry, API keys, and the Discord bridge state.
              Every other component is a spoke that talks to central over HTTP.
            </p>
            <pre className="docs-ascii"><code>{`                        ┌─────────────────────────────────────┐
   React console ◄─────►│                                     │◄────► GPU workers (worker_agent)
   (@hugpy/ui)          │      FLASK CENTRAL  (:7002)         │       - role=worker (lead)
                        │                                     │       - role=rpc (shard backend)
   Discord bot   ◄─────►│  registry · downloads · /chat/stream│
   (hugpy bot)          │  /v1 (OpenAI) · API keys · workers  │◄────► Thin GGUF workers (Termux/ARM)
                        │  discord bridge · phone-brick pool  │
   OpenAI SDK    ◄─────►│                                     │◄────► Phone-brick fleet (Android/ONNX)
   clients              │     ▼ managers (execution) ▼        │
                        │  resolve() → allocator → runners    │◄────► Peer centrals (placement.json)
                        └─────────────────────────────────────┘`}</code></pre>
            <p>
              The <strong>managers layer</strong> is central’s internal “execution brain.” A single
              <code>resolve()</code> function is the one authority that maps <code>(model_key, task)</code> →
              <code>(framework, builder, runner)</code> and decides <em>where</em> a request runs: local
              in-process, a remote GPU worker, a peer central, or a cross-machine shard pool.
            </p>
            <h3 className="docs-h3-plain">Canonical request flow (chat)</h3>
            <ol className="docs-steps-list">
              <li>UI / bot / SDK → <code>POST /chat/stream</code> (or <code>/v1/chat/completions</code>) on central.</li>
              <li><code>dispatch.execute_chat_stream()</code> → <code>resolve()</code> picks framework + placement.</li>
              <li>Placement cascade: forced-local → static peer (<code>placement.json</code>) → live GPU worker pool (pool-reserved + capability-ranked) → local in-process runner.</li>
              <li>If a worker is chosen, central relays the worker’s SSE token stream back to the client and emits a keepalive heartbeat (<code>HUGPY_SSE_HEARTBEAT_SECS</code>, default 15s); on a worker error before first token it falls back to local. The console’s chat banner names which worker (or “local”) served the request.</li>
              <li>Auto-continuation loops past the per-pass token cap until <code>finish_reason ≠ "length"</code> (or <code>HUGPY_MAX_CONTINUATIONS</code>), de-duplicating the overlap at each seam.</li>
            </ol>
          </section>

          <section id="sections" className="docs-sec">
            <h2>3 · Section-by-section</h2>

            <h3 id="flask-central" className="docs-h3">A. Flask central API <span className="docs-path">api/hugpy/flask_app/</span></h3>
            <p>
              The orchestration hub. Blueprints are dual-mounted at both <code>/…</code> and <code>/api/…</code>
              (via <code>ApiPrefixMiddleware</code>) so it works behind nginx <em>or</em> bare gunicorn, and the
              built console is mounted from the wheel’s <code>console_dist/</code> (override
              <code>HUGPY_UI_DIST</code>; dev fallback <code>ui/dist</code>) when present.
            </p>
            <p className="docs-eyebrow">Key route groups</p>
            <ul className="docs-routes">
              <li><code>/chat/stream</code> — SSE token streaming with keepalive heartbeats and graceful teardown of the worker bridge connection.</li>
              <li><code>/v1/*</code> — OpenAI-compatible <code>models</code> + <code>chat/completions</code>; on-site <strong>API keys</strong> (<code>hp_…</code>, stored hashed in <code>api_keys.json</code>, shown once, with a site-wide <code>require_key</code> toggle, optional per-key <strong>pool</strong> binding); <code>/auth/config</code> exposes <code>open</code> vs <code>external</code> auth mode.</li>
              <li><code>/prompt</code>, <code>/prompt/tasks</code> — one verb for <em>every</em> non-chat task (embeddings, summarize, whisper, vision, text-to-image), dispatched by explicit <code>task</code> → model’s <code>primary_task</code> → input media type → chat default.</li>
              <li><code>/models</code>, <code>/jobs</code> — registry listing + resumable download jobs (stall detection, retry, cancel; marker file <code>hugpy.json</code> on completion).</li>
              <li><code>/llm/workers/*</code> — register / heartbeat, assign / unassign / unload, VRAM <strong>probe</strong>, admission (<code>/admit</code>, <code>/block</code>, <code>/admission</code>), per-worker <code>/pool</code>, enrollment tokens (<code>/llm/enroll-tokens</code>), and a one-line join script (<code>/llm/workers/install.sh</code>).</li>
              <li><code>/llm/models/&lt;key&gt;/{'{'}manifest,file,archive{'}'}</code> — provision a model to a worker: manifest (sizes + routing), single file (HTTP Range / resumable), or the whole model as a streamed tar. Plus a PEP-503 <strong>pip index</strong> (<code>/llm/pip/simple/…</code>) for worker self-update.</li>
              <li><code>/llm/serving/*</code> (overview / get / set), <code>/llm/slots/*</code> (overview / load / unload / install) — off / systemd / supervised / swap serving modes + the slot pool; the GGUF serving editor (variant dropdown) posts here.</li>
              <li><code>/llm/queue</code> — the live in-flight chat queue (<code>{'{'}active, counts{'}'}</code>) that powers the topbar activity chip.</li>
              <li><code>/search</code>, <code>/hf/spec</code> — HuggingFace discovery + per-repo install options (GGUF quants / transformers).</li>
              <li><code>/discord/*</code> — bindings, bridges, outbox / inbox, channel &amp; user reporting.</li>
              <li><code>/phone-brick/*</code> — phone pool register / heartbeat, run orchestration (SSE progress), Termux bootstrap script + code tarball.</li>
              <li><code>/llm/peers</code> — central + online workers as peer nodes (disk / mount state).</li>
              <li><code>/version</code>, <code>/readiness</code> — ungated probes: <code>/version</code> returns <code>{'{'}name, version, api, auth_mode{'}'}</code> (pin downstream apps to it); <code>/readiness</code> is a never-5xx snapshot backing the Landing onboarding. <code>/uploads</code> accepts media bounded by <code>HUGPY_MAX_UPLOAD_MB</code> (default 100). In external mode <code>/auth-svc/*</code> is the same-origin auth BFF.</li>
            </ul>

            <h3 id="managers" className="docs-h3">B. Managers — model execution <span className="docs-path">api/hugpy/managers/</span></h3>
            <p>Central’s execution brain. Sits between routes and the actual ML backends.</p>
            <ul className="docs-routes">
              <li><code>resolvers/</code> — <code>resolve()</code> (the single routing authority), <code>allocator.py</code> (a <strong>stateless, deterministic</strong> GPU placement function returning <code>WHOLE</code> / <code>SHARD</code> / <code>CPU</code> / <code>NONE</code>), <code>remote.py</code> (<code>DelegatingRunner</code> for workers + <code>PeerRunner</code> for peer centrals).</li>
              <li><code>serve/</code> — four serving modes — <strong>off</strong> (the in-process runner serves it), <strong>systemd</strong> (a Linux always-on unit per model), <strong>supervised</strong> (a portable detached <code>llama-server</code> per model — the cross-OS systemd replacement, default off-Linux), and <strong>swap</strong> (one shared llama-swap proxy, on-demand) — plus a <strong>slot pool</strong> of generic slot services each running one autofitting <code>llama-server</code> child.</li>
              <li><code>llama/runners/</code> — <code>python_runner</code> (in-process <code>llama_cpp</code>), <code>ccp_runner</code> (HTTP to <code>llama-server</code>), <code>shard_server</code> (spawns the <code>--rpc --tensor-split</code> lead for cross-machine sharding, resolving <code>--mmproj</code> for vision GGUFs), <code>chat_runner</code> (singleton-cached adapter).</li>
              <li><code>spill.py</code> — GPU/CPU memory budgeting: autofit GGUF GPU layers, transformers <code>device_map</code> budgets, and RPC/tensor-split env for sharding.</li>
              <li><strong>Modalities</strong> — <code>vision/</code>, <code>video/</code> (frame-by-frame, resume-aware), <code>whisper_model/</code> (ASR), <code>summarizers/</code>, <code>keywords/</code>, embeddings. A shared async runtime (<code>_platform/async_runtime.py</code>) runs all async work on one long-lived loop.</li>
            </ul>
            <div className="docs-callout">
              <div className="docs-callout-t">Cross-machine GGUF sharding (the allocator path)</div>
              <p>Two gates, <strong>both</strong> required:</p>
              <ol>
                <li><strong>Opt-in:</strong> model key listed in <code>HUGPY_SHARD_MODELS</code> (<code>"key:bytes,…"</code>).</li>
                <li><strong>Doesn’t fit whole:</strong> no single worker’s summed VRAM holds <code>bytes × 1.15</code>.</li>
              </ol>
              <p>
                When both hold, the allocator pools a <strong>lead</strong> (biggest <code>role=worker</code> GPU node)
                + the fewest <strong>rpc backends</strong> (<code>role=rpc</code> nodes advertising <code>rpc_endpoint</code>)
                to reach the target, with a VRAM-proportional <code>tensor_split</code>. Implemented for
                GGUF / llama.cpp; the transformers / Petals equivalent is deferred.
              </p>
            </div>

            <h3 id="worker-fleet" className="docs-h3">C. Distributed worker fleet</h3>
            <ul className="docs-routes">
              <li><code>worker_agent/</code> — full GPU worker: registers, heartbeats every ~15s (45s liveness window), GPU/CPU spill, central-first model provisioning (HTTP Range, HF fallback), <strong>self-update</strong> (<code>pip install -U</code> + re-exec, PyPI by default or central’s index), lazy loading (a model loads into VRAM/RAM only when a request for it arrives — assignment, restart, and self-update never warm one), and <code>role=rpc</code> mode running <code>rpc-server</code>. Serves <code>/health</code>, <code>/infer</code>, <code>/infer/stream</code>, <code>/infer/cancel/&lt;request_id&gt;</code>, <code>/probe</code>, <code>/models/unload</code>.</li>
              <li><code>gguf_worker/</code> — slim, import-light, <strong>chat-only GGUF</strong> worker for Termux / ARM; returns HTTP 501 for unsupported tasks <em>before</em> SSE starts, so central’s <code>DelegatingRunner</code> falls back cleanly.</li>
              <li><code>phone_brick/</code> — two halves: (1) orchestrator-free ONNX YOLO PPE detection across cheap Android phones with <strong>plurality consensus</strong> (AGR / DIS / NOD); (2) an LLM contribution path — <code>rpc_backend.py</code> lets a phone join the GPU shard pool as a <code>role=rpc</code> llama.cpp backend advertising its RAM as pseudo-VRAM (gated by <code>PHONE_BRICK_RPC</code>), and <code>analyze.py</code> feeds phone detections into a chat model for natural-language scene/safety reasoning.</li>
              <li><code>engine/</code> — resolve / fetch / build of native llama.cpp binaries (<code>hugpy install-engine --cuda</code>); the source build always sets <code>-DGGML_RPC=ON</code> so <code>rpc-server</code> exists for the shard fleet.</li>
              <li><code>keeper.py</code> — a stdlib-only terminal REPL where an LLM “keeps” a machine / LXD guest via a shell action loop (Agent–Executor pattern).</li>
              <li><code>_platform/</code> — cross-platform GPU detection (nvidia-smi → torch.cuda), RAM probing, path resolution, detached process spawning.</li>
            </ul>

            <div className="docs-callout" id="pools">
              <div className="docs-callout-t">Dedicated worker pools</div>
              <p>
                A worker is reserved to a named pool via the <code>WORKER_POOL</code> env var (empty = the general
                pool); it re-asserts the label on register/heartbeat. A pooled worker serves <strong>only</strong>
                requests tagged for its pool — general traffic never lands on it. A request is tagged either by an
                API key minted with a <code>pool</code> (binds every request from that key) or by an explicit
                per-request <code>pool</code> (honored for keyless/open callers, or keys that permit it). A pool
                request that finds no matching worker falls back to <strong>local</strong>, never the shared pool.
                Operators set or clear a worker’s pool from the workers panel pill or <code>POST /llm/workers/&lt;id&gt;/pool</code>.
              </p>
            </div>

            <div className="docs-callout" id="admission">
              <div className="docs-callout-t">Admission, enrollment &amp; the capability guard</div>
              <p>
                Workers don’t serve on join: a new worker is <code>pending</code> until an operator clicks
                <strong>Admit</strong>. States (pending / approved / blocked) persist across heartbeats; a blocked
                worker gets a 403 and exits cleanly (no respawn). Enrollment tokens
                (<code>/llm/enroll-tokens</code>, shown once, passed as <code>WORKER_ENROLL_TOKEN</code>) gate
                registration when <code>HUGPY_WORKER_ENROLL_REQUIRED</code> is on. Selection is
                <strong>capability-aware</strong>: central skips a worker only when it reports
                <code>engine.installed == false</code>, and ranks the rest soft-version-match → warm → usable-GPU →
                least-recently-picked → stable id.
              </p>
            </div>

            <h3 id="discord-bot" className="docs-h3">D. Discord bot <span className="docs-path">api/hugpy/bot/</span></h3>
            <p>A separate process driving central over HTTP. Cogs:</p>
            <ul className="docs-routes">
              <li><strong>chat</strong> — <code>/chat</code>, <code>/model</code>, <code>/reset</code>, <code>/running</code>, <code>/stop</code>.</li>
              <li><strong>ml</strong> — <code>/embed</code>, <code>/similarity</code>, <code>/imagine</code>, <code>/task</code>, <code>/tasks</code>.</li>
              <li><strong>ops</strong> — <code>/status</code>, <code>/models</code>, <code>/download</code>, <code>/jobs</code>, <code>/canceljob</code>, <code>/hf</code>.</li>
              <li><strong>tools</strong> — <code>/summarize</code>, <code>/keywords</code>, <code>/transcribe</code>, <code>/describe</code>.</li>
            </ul>
            <p>
              <code>MessageStreamer</code> streams tokens into throttled Discord message edits (rolling over at
              the 2000-char limit). The bot drains central’s <code>/discord/outbox</code>, reports visible
              channels/users, and relays bridged channels to <code>/discord/inbox</code>.
            </p>

            <h3 id="react-console" className="docs-h3">E. React console <span className="docs-path">ui/src/</span></h3>
            <p>
              Panels map 1:1 to backend subsystems and all requests route through a configurable
              <code>hugpyFetch</code> / <code>HugpyProvider</code> layer (the whole console is embeddable as
              <code>&lt;HugpyConsole baseUrl=… /&gt;</code> from the <code>@hugpy/ui</code> npm package — distinct
              from <code>@hugpy/console</code>, the separate portable terminal it can lazy-load).
            </p>
            <ul className="docs-routes">
              <li><strong>ChatPanel</strong> — per-model, localStorage-persisted chats; transmitted system prompt + full multi-turn history; inline base64 image upload for <code>image-text-to-text</code> (VL) models; an <strong>allocation banner</strong> naming which worker (<code>served_by</code> name + id) or “local (this node)” served the last request; stop/cancel.</li>
              <li><strong>ModelTable + ServingControl</strong> — registry table, download / probe / assign, and a GGUF <strong>variant dropdown</strong> (downloaded <code>.gguf</code> basenames + “auto”, persisted as <code>gguf_file</code> in the serve override).</li>
              <li><strong>ActivityQueue</strong> (topbar) — a live chip showing <code>N active · N queued</code> with an expandable per-request list (state / model / elapsed / tokens), polling <code>/api/llm/queue</code>.</li>
              <li><strong>HFSearch</strong> — HuggingFace discovery + add-to-local.</li>
              <li><strong>WorkersPanel</strong> — assign / probe / health / spill allocation, residency + pin controls (tiers v3), the admission gate (approve / pending / block), a pool pill (set/clear), and enrollment-token issuing.</li>
              <li><strong>ApiAccess</strong> — mint / revoke API keys, <code>require_key</code> toggle, an optional <code>pool</code> field binding a key to a worker pool, cURL example.</li>
              <li><strong>DiscordPanel</strong> (one-way bindings) + <strong>BridgePanel</strong> (supervised conversations with approve / reject / defer modes).</li>
              <li><strong>PhoneBrickPanel</strong> — run + monitor phone-brick consensus.</li>
              <li><strong>PeersBar</strong> — peer node disk / mount status.</li>
            </ul>
            <p>
              Auth is pluggable: <strong>open</strong> (single-operator, no wall) or <strong>external</strong>
              (delegated login service via the same-origin BFF), resolved from <code>/api/auth/config</code>.
            </p>
          </section>

          <section id="env" className="docs-sec">
            <h2>4 · Environment variables (complete reference)</h2>
            <p className="docs-lede">
              Every environment variable the hugpy codebase actually reads, grouped by subsystem.
              Generated by walking every <code>os.getenv</code> / <code>os.environ.get</code> /
              <code>os.environ[...]</code> call site (incl. the <code>get_env_value</code>/<code>env_value</code>
              wrapper hugpy's cross-platform layer uses) plus the <code>VITE_*</code> build-time vars in the
              standalone arm UIs. <strong>Defaults shown are code-level fallbacks</strong> — a systemd drop-in or
              <code>.env</code> can still override them; never a secret value, only the variable's name and purpose.
            </p>

            <p className="docs-eyebrow">Quick start — copy, edit, run</p>
            <p>
              Two minimal, genuinely-runnable <code>.env</code> examples — a central box and a worker box.
              Everything else in this reference is optional tuning; these are the only variables you typically
              need to set to get a working fleet.
            </p>
            <CopyBlock title=".env — CENTRAL (hugpy serve)" code={`# hugpy central — .env / systemd Environment=
HUGPY_BASE_URL=http://127.0.0.1:7002   # this box's own address; workers/bot/keeper all dial this
HUGPY_AUTH_MODE=open                   # no login wall — fine for a single-operator box on a private network
DEFAULT_ROOT=/srv/hugpy-data           # SET THIS: storage root for models/uploads/identities/datasets — largest writable volume
SLOT_COUNT=2                           # local llama-server slot children spawned at boot, EMPTY (each loads a model only on a request; 0 disables the pool)
HUGPY_MAX_UPLOAD_MB=100                # cap on any single upload/POST body
HUGPY_ALLOWED_ORIGINS=                 # leave unset while developing; set to your UI origin(s) before exposing this box publicly`} />
            <CopyBlock title=".env — WORKER (hugpy worker)" code={`# hugpy worker — .env / systemd Environment=
WORKER_CENTRAL_URL=http://<CENTRAL_HOST>:7002   # the central this worker registers to — MUST be reachable FROM the worker box
                                                # (a LAN IP is safest; a public hostname may fail to resolve/route from inside the LAN)
WORKER_NAME=computron                          # this worker's fleet display name
WORKER_PORT=9100                               # port the worker agent HTTP server binds
DEFAULT_ROOT=/mnt/storage/hugpy-worker/storage # model/data store root — put it on the LARGEST writable volume, never a small root disk
HUGPY_ENGINE_DIR=%h/hugpy-worker/engine        # native llama.cpp engine build dir (fetched/built binaries live here)
DEFAULT_SERVE_MODE=off                         # on-demand serving (not eager on boot)`} />

            <div className="docs-callout">
              <div className="docs-callout-t">Kill-switches at a glance (default state)</div>
              <p>These flip a whole behavior on/off; everything else in the tables below is a tuning knob.</p>
              <ul className="docs-routes">
                <li><code>HUGPY_NO_LOCAL_SERVING</code> — default <strong>off</strong> (this box may serve locally)</li>
                <li><code>HUGPY_LOCAL_FALLBACK</code> — default <strong>off</strong> (never silently fall back to local after a worker fails)</li>
                <li><code>HUGPY_CENTRAL_GATE</code> — default <strong>on</strong> (per-worker-model concurrency gate active)</li>
                <li><code>HUGPY_WORKER_GEN_GATE</code> — default <strong>on</strong> (in-process generation gate active)</li>
                <li><code>HUGPY_VIDEOGEN_LOCAL</code> — default <strong>off</strong> (central refuses local image/scene gen if a fleet exists but no live worker serves it)</li>
                <li><code>HUGPY_CUDA_EXPANDABLE</code> — default <strong>off</strong> (opt-in only; some driver/torch combos crash under it)</li>
                <li><code>HUGPY_TRUST_REMOTE_CODE</code> — default <strong>off</strong> (security-sensitive; never auto-execute a repo's Python)</li>
                <li><code>HUGPY_WORKER_ENROLL_REQUIRED</code> — default <strong>off</strong> (tokenless workers register as pending, not rejected)</li>
                <li><code>HUGPY_MODEL_STORE_REAPABLE</code> / <code>HUGPY_SHARED_MODEL_STORE</code> — default <strong>off</strong> / <strong>off</strong> (safe-by-default: nothing on a worker's disk is ever deleted unless explicitly declared local &amp; disposable)</li>
                <li><code>HUGPY_DISCOVERY_ALLOW_SHRINK</code> — default <strong>off</strong> (a &gt;50% drop in discovered models is refused, not silently accepted)</li>
                <li><code>PHONE_BRICK_ENABLE_SHELL</code> — default <strong>off</strong> (arbitrary shell execution on a phone-brick worker; dangerous, opt-in only)</li>
                <li><code>HUGPY_BOT_MEMBERS_INTENT</code> — default <strong>off</strong> (privileged Discord gateway intent)</li>
                <li><code>WORKER_SELF_UPDATE</code> — default <strong>on</strong> (a worker auto pip-installs central's advertised version)</li>
                <li><code>IDENTITY_FRONT_AUTOSELECT</code> — default <strong>on</strong> (fleet-VLM picks the best front-view reference automatically)</li>
                <li><code>STUDIO_ALLOW_UNPINNED</code> — default <strong>off</strong> (Studio runners require hash-pinned weights)</li>
                <li><code>HUGPY_AUTO_DOWNLOAD</code> — default <strong>on</strong> (a fresh install auto-pulls the curated model fleet on first use)</li>
              </ul>
            </div>

            <EnvReference>
            <EnvGroup id="env-central" label="Central address">
            <table className="docs-table docs-table--env">
              <thead><tr><th>Variable</th><th>Default</th><th>Purpose</th><th>Values</th></tr></thead>
              <tbody>
                <tr><td><code>HUGPY_BASE_URL</code> <span className="docs-note">(aliases <code>HUGPY_CENTRAL</code>, <code>HUGPY_URL</code>, <code>WORKER_CENTRAL_URL</code>, <code>CENTRAL_URL</code>)</span></td><td><code>http://127.0.0.1:7002</code></td><td>Canonical base URL every arm (bot, keeper, worker, slot) dials to reach central. First non-empty of the alias list wins, canonical name first.</td><td>URL</td></tr>
              </tbody>
            </table>
            </EnvGroup>

            <EnvGroup id="env-auth" label="Auth & security">
            <table className="docs-table docs-table--env">
              <thead><tr><th>Variable</th><th>Default</th><th>Purpose</th><th>Values</th></tr></thead>
              <tbody>
                <tr><td><code>HUGPY_AUTH_MODE</code></td><td><code>open</code> (<code>hugpy serve</code>) / <code>external</code> (bare gunicorn)</td><td>open = no login wall (single operator); external = delegate to an upstream login service via the same-origin BFF.</td><td>"open" | "external"</td></tr>
                <tr><td><code>HUGPY_AUTH_BASE</code> / <code>HUGPY_AUTH_UPSTREAM</code></td><td>falls back to a hardcoded upstream</td><td>Upstream auth service base URL the same-origin BFF proxies login/me calls to; also advertised directly to the UI when the proxy is disabled.</td><td>URL</td></tr>
                <tr><td><code>HUGPY_AUTH_PROXY</code></td><td><code>1</code> (on)</td><td>Enables the same-origin cookie-rewriting auth proxy (fixes third-party-cookie login loops in Safari/Firefox).</td><td>0/false/no/off disables</td></tr>
                <tr><td><code>HUGPY_AUTH_PUBLIC_BASE</code></td><td><code>/api/auth-svc</code></td><td>Same-origin base path/URL advertised to the console UI for auth calls.</td><td>path or URL</td></tr>
                <tr><td><code>HUGPY_OPERATOR_TOKEN</code></td><td><code>""</code> (no gate)</td><td>Shared secret compared against the <code>X-Operator-Token</code>/Bearer header to authorize privileged operator-only routes.</td><td>opaque token string</td></tr>
                <tr><td><code>HUGPY_API_KEY</code> / <code>HUGPY_TOKEN</code></td><td><code>""</code></td><td>Bearer credential CLI tools (keeper, hugpy CLI) send to central's own API.</td><td>opaque token string</td></tr>
                <tr><td><code>KEEPER_API_KEY</code></td><td>falls back to <code>HUGPY_API_KEY</code></td><td>API key the keeper REPL/Discord-bridge loop uses to authenticate its calls to central.</td><td>opaque token string</td></tr>
                <tr><td><code>HUGPY_TRUST_REMOTE_CODE</code></td><td>off</td><td><strong>Security-sensitive.</strong> Operator opt-in allowing a Hugging Face repo to run arbitrary Python on load.</td><td>1/true/yes/on</td></tr>
                <tr><td><code>HUGPY_WORKER_ENROLL_REQUIRED</code></td><td>off</td><td>Whether a valid enrollment token is mandatory for a worker to register/heartbeat.</td><td>1/true/yes/on</td></tr>
                <tr><td><code>WORKER_ENROLL_TOKEN</code></td><td>none</td><td>Enrollment bearer token minted by central (<code>POST /llm/enroll-tokens</code>) and passed by a joining worker.</td><td>opaque token (e.g. <code>hpw_…</code>)</td></tr>
              </tbody>
            </table>
            </EnvGroup>

            <EnvGroup id="env-comms" label="Comms, jobs & settings storage">
            <table className="docs-table docs-table--env">
              <thead><tr><th>Variable</th><th>Default</th><th>Purpose</th><th>Values</th></tr></thead>
              <tbody>
                <tr><td><code>HUGPY_COMMS_DB</code></td><td>path under <code>$XDG_RUNTIME_DIR</code> (or <code>/tmp</code>)</td><td>Cross-process SQLite mirror for the jobs store (gunicorn runs multiple worker processes sharing one comms mirror); also its own off-switch.</td><td>filesystem path, or "off"/"none"/"0"/"disabled"</td></tr>
                <tr><td><code>HUGPY_JOB_STALL_SECONDS</code></td><td>90</td><td>Forward-progress silence threshold after which an active job reads as "stalled" on <code>/llm/jobs</code>.</td><td>float seconds</td></tr>
                <tr><td><code>HUGPY_SETTINGS_PATH</code></td><td><code>&lt;PROJECTS_HOME&gt;/settings.json</code></td><td>Explicit override path for the console settings store.</td><td>filesystem path</td></tr>
                <tr><td><code>HUGPY_PRINCIPALS_PATH</code></td><td><code>&lt;PROJECTS_HOME&gt;/…</code></td><td>Explicit override path for the auth principals/tokens store.</td><td>filesystem path</td></tr>
                <tr><td><code>PROJECTS_HOME</code></td><td><code>&lt;DEFAULT_ROOT&gt;/projects</code></td><td>Base dir for settings.json / principals.json / audit.log / placement.json etc.</td><td>filesystem path</td></tr>
                <tr><td><code>SERVE_OVERRIDES_PATH</code></td><td><code>&lt;PROJECTS_HOME&gt;/serve_overrides.json</code></td><td>Per-model console-writable serving-overrides file.</td><td>filesystem path</td></tr>
                <tr><td><code>XDG_RUNTIME_DIR</code></td><td><code>/tmp</code></td><td>Base runtime dir the per-uid comms SQLite DB sits under when <code>HUGPY_COMMS_DB</code> is unset. <span className="docs-note">(Linux standard, not hugpy-specific.)</span></td><td>filesystem path</td></tr>
              </tbody>
            </table>
            </EnvGroup>

            <EnvGroup id="env-serving" label="Serving posture, residency & slot pool">
            <table className="docs-table docs-table--env">
              <thead><tr><th>Variable</th><th>Default</th><th>Purpose</th><th>Values</th></tr></thead>
              <tbody>
                <tr><td><code>HUGPY_NO_LOCAL_SERVING</code></td><td>off</td><td>Per-box opt-out: never host/serve a model in this process; forces routing to a registered worker.</td><td>1/true/yes/on</td></tr>
                <tr><td><code>SLOT_COUNT</code></td><td>2</td><td>Number of always-on llama-server "slot" child processes in the local pool; <code>0</code> disables the pool. The operator's console setting always wins over this env/drop-in value.</td><td>non-negative int</td></tr>
                <tr><td><code>SLOT_ADVERTISE</code> / <code>SLOT_HOST_ADDR</code></td><td><code>127.0.0.1</code></td><td>Host/IP the scheduler advertises to reach a slot.</td><td>hostname/IP</td></tr>
                <tr><td><code>SLOT_PORT_BASE</code></td><td>8101</td><td>Base TCP port; slot N listens on base+(N-1).</td><td>int port</td></tr>
                <tr><td><code>SLOT_HOST</code></td><td><code>0.0.0.0</code></td><td>Bind address for a slot's own control+proxy HTTP server.</td><td>host/IP</td></tr>
                <tr><td><code>SLOT_ID</code></td><td>1</td><td>This slot instance's label/number (systemd template <code>%i</code>); its port derives from this.</td><td>string/int id</td></tr>
                <tr><td><code>SLOT_PORT</code></td><td><code>SLOT_PORT_BASE</code>+<code>SLOT_ID</code>-1</td><td>This slot's own control/proxy port.</td><td>int port</td></tr>
                <tr><td><code>SLOT_CHILD_PORT</code></td><td><code>SLOT_PORT</code>+1000</td><td>Port the child llama-server process listens on.</td><td>int port</td></tr>
                <tr><td><code>SLOT_HEALTH_TIMEOUT</code></td><td>180</td><td>Seconds to wait for a slot's child llama-server to report healthy.</td><td>float seconds</td></tr>
                <tr><td><code>MAIN_GPU</code></td><td>none</td><td>Pins a slot's child llama-server to one GPU index (sets <code>CUDA_VISIBLE_DEVICES</code>).</td><td>int GPU index</td></tr>
                <tr><td><code>LLAMA_SERVICE_USER</code> / <code>LLAMA_SERVICE_GROUP</code></td><td>current user (serve.py) or "solcatcher"/"web" (slot unit template)</td><td><code>User=</code>/<code>Group=</code> for generated per-model or per-slot systemd unit templates.</td><td>username / group string</td></tr>
                <tr><td><code>HUGPY_WARM_COOLDOWN_S</code></td><td>—</td><td>Retired: there are no background warm-probe attempts. A model loads only when a request for it arrives. This variable is ignored.</td><td>(ignored)</td></tr>
                <tr><td><code>HUGPY_VRAM_HEADROOM</code></td><td>1.15</td><td>Multiplier over raw GGUF file size estimating real VRAM need (KV-cache/runtime overhead) for the worker-slot fit preflight.</td><td>float multiplier</td></tr>
                <tr><td><code>HUGPY_CENTRAL_GATE</code></td><td>on</td><td>Per-(worker, model) in-flight concurrency gate. Kill-switch.</td><td>off/0/false/no disables</td></tr>
                <tr><td><code>HUGPY_CENTRAL_GATE_WAIT_S</code></td><td>30</td><td>Bounded wait for a busy gate slot to free before returning a "busy" error.</td><td>float seconds, ≥0</td></tr>
                <tr><td><code>HUGPY_LOCAL_FALLBACK</code></td><td>off</td><td>Whether central may run a worker-selected model locally after the worker path fails (default no, so a worker failure never silently burns central's CPU/RAM).</td><td>always/1/true/yes/on</td></tr>
                <tr><td><code>HUGPY_WORKER_ROOT</code></td><td>none</td><td>Explicit override of a worker's local root dir (profile venvs live under <code>&lt;root&gt;/envs</code>).</td><td>filesystem path</td></tr>
                <tr><td><code>DEFAULT_SERVE_MODE</code></td><td>auto (supervised off-Linux)</td><td>Which serve mode a freshly-served model gets.</td><td>"off" | "systemd" | "supervised" | "swap"</td></tr>
                <tr><td><code>DEFAULT_LLAMA_NGL</code></td><td>-1 (all layers)</td><td>Default GPU layer offload for systemd/supervised/swap-served models.</td><td>int (-1 = all)</td></tr>
                <tr><td><code>DEFAULT_LLAMA_THREADS</code></td><td><code>cpu_count()</code>-1</td><td>Box-wide cap on llama.cpp generation threads.</td><td>int threads</td></tr>
                <tr><td><code>LLAMA_PORT_BASE</code> / <code>LLAMA_PORT_SPAN</code></td><td>7001 / 4000</td><td>Base port and span reserved for per-model systemd/supervised llama-server unit allocation.</td><td>int / int</td></tr>
                <tr><td><code>LLAMA_SWAP_HOST</code> / <code>_PORT</code> / <code>_CONFIG</code> / <code>_UNIT</code> / <code>_TTL</code></td><td>127.0.0.1 / 9292 / /etc/llama-swap/config.yaml / llama-swap.service / 600</td><td>The shared llama-swap proxy's address, generated config path, systemd unit name, and on-demand unload TTL.</td><td>host / int port / path / unit name / int seconds</td></tr>
                <tr><td><code>SYSTEMD_UNIT_DIR</code></td><td><code>/etc/systemd/system</code></td><td>Where generated per-model systemd units are written.</td><td>filesystem path</td></tr>
                <tr><td><code>LLAMA_UNIT_PREFIX</code></td><td>"llama"</td><td>Prefix for generated systemd unit names.</td><td>free-text string (must be a valid systemd unit-name fragment)</td></tr>
                <tr><td><code>LLAMA_HOST</code></td><td><code>http://127.0.0.1</code></td><td>Base scheme+host <code>serve.py</code>'s <code>_bare_host()</code> and the llama-runner config layer fall back to when resolving a served model's own reachable address (systemd/supervised/swap serve modes).</td><td>URL (scheme optional — a bare host is accepted and re-prefixed)</td></tr>
              </tbody>
            </table>
            </EnvGroup>

            <EnvGroup id="env-spill" label="GPU/CPU offload & cross-machine sharding">
            <table className="docs-table docs-table--env">
              <thead><tr><th>Variable</th><th>Default</th><th>Purpose</th><th>Values</th></tr></thead>
              <tbody>
                <tr><td><code>HUGPY_N_GPU_LAYERS</code></td><td>auto (autofit)</td><td>llama.cpp GPU layer-offload mode for a GGUF model.</td><td>"auto" | "off"/"cpu"/"none" (0) | int layer count | -1 (all)</td></tr>
                <tr><td><code>HUGPY_TENSOR_SPLIT</code></td><td>none</td><td>Multi-GPU VRAM split ratio passed to llama.cpp's <code>tensor_split</code>.</td><td>comma floats, e.g. "0.7,0.3"</td></tr>
                <tr><td><code>HUGPY_MAIN_GPU</code></td><td>0</td><td>Primary GPU index for free-VRAM probing and llama.cpp's <code>main_gpu</code>.</td><td>int GPU index</td></tr>
                <tr><td><code>HUGPY_GPU_MEM_GIB</code></td><td>none</td><td>Operator VRAM budget; caps GGUF layer autofit and the transformers <code>max_memory</code> per-device budget. Also used as this box's own operator cap (min of central's limit &amp; this).</td><td>float GiB</td></tr>
                <tr><td><code>HUGPY_CPU_MEM_GIB</code></td><td>none</td><td>CPU/RAM budget for transformers <code>device_map="auto"</code> spill; also a worker-side operator RAM cap.</td><td>float GiB</td></tr>
                <tr><td><code>HUGPY_N_GPU</code></td><td>1</td><td>Number of GPUs to spread the transformers <code>max_memory</code> map across.</td><td>int count</td></tr>
                <tr><td><code>HUGPY_VRAM_RESERVE_GIB</code></td><td>1.0</td><td>VRAM slice withheld from every budget/autofit calc, for other GPU consumers central can't see.</td><td>float GiB</td></tr>
                <tr><td><code>HUGPY_RAM_RESERVE_GIB</code></td><td>4.0</td><td>RAM slice withheld from every budget calc, to avoid OOM-killing the agent.</td><td>float GiB</td></tr>
                <tr><td><code>HUGPY_RAM_MAX_GIB</code></td><td>none</td><td>Hard ceiling on RAM hugpy may use on this box, regardless of actual free RAM.</td><td>float GiB</td></tr>
                <tr><td><code>HUGPY_VRAM_CTX_RESERVE_GIB</code></td><td>2.5</td><td>VRAM reserved for llama_context (KV cache + compute-graph buffers) beyond the flat safety margin.</td><td>float GiB</td></tr>
                <tr><td><code>HUGPY_VRAM_CEILING_CUSHION_GIB</code></td><td>0.5</td><td>Working room the GPU admission gate keeps free after a model lands — the compute-graph/activation residual (measured ~348 MiB), <em>not</em> the KV cache, which the fit need already prices. Un-stacked against <code>HUGPY_VRAM_RESERVE_GIB</code>: the guarantee is that raw free VRAM after the load stays at or above the larger of the two.</td><td>float GiB</td></tr>
                <tr><td><code>HUGPY_VRAM_CEILING_FRAC</code></td><td>unset (cushion governs)</td><td>Legacy override: when set, the admission gate instead demands <code>(1 − frac)</code> of <em>total</em> card VRAM free after the load. Sizing a fixed residual as a share of the card refuses models a big card can genuinely run, so this is an escape hatch, not the default.</td><td>float in (0, 1]</td></tr>
                <tr><td><code>HUGPY_RPC_SERVERS</code></td><td>none</td><td>Comma list of llama.cpp rpc-server targets for cross-machine model sharding; set per-request by central's allocator.</td><td>"host:port,host:port,…"</td></tr>
                <tr><td><code>HUGPY_SHARD_MODELS</code></td><td>"" (none eligible)</td><td>Operator opt-in list of models allowed to shard across worker GPUs, each with a VRAM byte estimate. Requires the model to also not fit whole on any one worker.</td><td>"key:bytes,…" (bytes may use g/gb suffix, e.g. "BigModel:140gb")</td></tr>
                <tr><td><code>HUGPY_SHARD_PORT_BASE</code></td><td>8790</td><td>Base TCP port for on-demand llama-server shard-lead processes.</td><td>int port</td></tr>
                <tr><td><code>HUGPY_SHARD_HEALTH_TIMEOUT</code></td><td>180</td><td>Seconds to wait for a shard-lead subprocess to come up healthy.</td><td>float seconds</td></tr>
                <tr><td><code>HUGPY_VISION_NGL</code></td><td>999</td><td>GPU layer count for the native llama-server serving vision GGUFs with <code>--mmproj</code> (999 ≈ offload all layers).</td><td>int layer count</td></tr>
                <tr><td><code>WORKER_SPILL</code></td><td>"auto"</td><td>Worker CLI: GPU/CPU layer-spill mode (fit as many layers on GPU as VRAM allows, spill the rest to CPU).</td><td>"auto" | "off"</td></tr>
                <tr><td><code>WORKER_N_GPU_LAYERS</code> / <code>WORKER_GPU_MEM_GIB</code> / <code>WORKER_CPU_MEM_GIB</code> / <code>WORKER_TENSOR_SPLIT</code> / <code>WORKER_MAIN_GPU</code></td><td>none</td><td>Worker CLI flags seeding the matching <code>HUGPY_*</code> spill env vars above at boot.</td><td>same types as above</td></tr>
                <tr><td><code>HUGPY_CUDA_EXPANDABLE</code></td><td>off</td><td>Opt-in: set <code>PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True</code> before torch import to defragment the CUDA allocator ahead of large loads. Some driver/torch combos crash under it — opt-in only, never default-on.</td><td>"1" enables; else disabled and a leaked prior value is actively stripped ("detox") at boot</td></tr>
                <tr><td><code>PYTORCH_CUDA_ALLOC_CONF</code></td><td>none</td><td>Standard PyTorch CUDA-allocator config var; hugpy only reads it to detect/purge a leaked <code>expandable_segments:True</code>. <span className="docs-note">(PyTorch-standard, not hugpy-specific.)</span></td><td>allocator-conf string</td></tr>
              </tbody>
            </table>
            </EnvGroup>

            <EnvGroup id="env-storage" label="Model storage & cache tiers">
            <table className="docs-table docs-table--env">
              <thead><tr><th>Variable</th><th>Default</th><th>Purpose</th><th>Values</th></tr></thead>
              <tbody>
                <tr><td><code>DEFAULT_ROOT</code></td><td>per-OS platform default</td><td>Single source of truth for the storage root; every <code>*_HOME</code> below derives from it. Only honoured if the path is actually creatable/writable, else falls back per-user.</td><td>filesystem path</td></tr>
                <tr><td><code>HUGPY_DATA_DIR</code> / <code>HUGPY_CONFIG_DIR</code> / <code>HUGPY_CACHE_DIR</code></td><td>per-OS default (XDG / AppData / Library)</td><td>Override the per-OS user data/config/cache directory roots.</td><td>filesystem path</td></tr>
                <tr><td><code>HUGPY_ENGINE_DIR</code> <span className="docs-note">(alias <code>LLAMA_CPP_DIR</code>)</span></td><td>per-OS default</td><td>Where the fetched/built native llama.cpp engine (llama-server, rpc-server, shared libs) lives.</td><td>filesystem path</td></tr>
                <tr><td><code>MODELS_HOME</code> <span className="docs-note">(alias <code>MODELS_DIR</code>)</span></td><td><code>&lt;DEFAULT_ROOT&gt;/models</code></td><td>Root for downloaded/served model files. A value pointing at an unwritable path is loudly ignored (guards against a stale <code>.env</code> naming a dead mount).</td><td>filesystem path</td></tr>
                <tr><td><code>UPLOADS_HOME</code> <span className="docs-note">(alias <code>CHAT_UPLOAD_DIR</code>)</span></td><td><code>&lt;DEFAULT_ROOT&gt;/uploads</code></td><td>Root for session-scoped chat uploads (subject to a session reaper).</td><td>filesystem path</td></tr>
                <tr><td><code>IDENTITIES_HOME</code> <span className="docs-note">(alias <code>IDENTITIES_DIR</code>)</span></td><td><code>&lt;DEFAULT_ROOT&gt;/identities</code></td><td>Root for persistent identity profiles — deliberately a <em>sibling</em> of <code>UPLOADS_HOME</code>, not a child, so the upload reaper can never reach it.</td><td>filesystem path</td></tr>
                <tr><td><code>DATASETS_HOME</code> <span className="docs-note">(alias <code>DATASETS_DIR</code>)</span></td><td><code>&lt;DEFAULT_ROOT&gt;/datasets</code></td><td>Root for dataset storage.</td><td>filesystem path</td></tr>
                <tr><td><code>HF_CACHE</code></td><td><code>&lt;MODELS_HOME&gt;/cache</code></td><td>Root for the HF/torch/pip cache subdirs below.</td><td>filesystem path</td></tr>
                <tr><td><code>HF_HOME</code> / <code>HF_HUB_CACHE</code> / <code>TORCH_HOME</code> / <code>PIP_CACHE_DIR</code></td><td>under <code>HF_CACHE</code></td><td>HuggingFace / PyTorch / pip cache homes; also exported back into the process env (<code>setdefault</code>) so downstream libraries pick them up.</td><td>filesystem path</td></tr>
                <tr><td><code>HUGPY_MODEL_CACHE</code></td><td><code>/var/cache/hugpy-models</code></td><td>SSD hot-cache directory GGUF models are promoted into from slower storage.</td><td>filesystem path</td></tr>
                <tr><td><code>HUGPY_MODEL_CACHE_MAX_GIB</code></td><td>450</td><td>Byte-budget ceiling of the SSD model cache (LRU-evicted).</td><td>float GiB</td></tr>
                <tr><td><code>HUGPY_HOT_CACHE_ROOT</code></td><td>"" (disabled)</td><td>Box-local NVMe dir the general hot-set LRU cache promotes catalog models into (contention-based LRU, promote-on-call).</td><td>filesystem path</td></tr>
                <tr><td><code>HUGPY_HOT_CACHE_GIB</code></td><td>225</td><td>Byte budget of the hot-cache tier.</td><td>float GiB</td></tr>
                <tr><td><code>HUGPY_HOT_CACHE_MIN_RESIDENCY_S</code></td><td>1800</td><td>Min idle seconds before a hot-cache entry becomes eviction-eligible (anti-thrash).</td><td>float seconds</td></tr>
                <tr><td><code>HUGPY_WORKER_DISK_RESERVE_GIB</code></td><td>50</td><td>Free-space reserve kept on a worker's model-root disk; below this the worker is "over budget" and cold local models become eviction candidates.</td><td>float GiB</td></tr>
                <tr><td><code>HUGPY_DISK_CACHE_MAX_GIB</code></td><td>none</td><td>A worker's stated local-disk delegation to the model cache; clamps a central <code>disk_cache_gib</code> limit to it.</td><td>float GiB</td></tr>
                <tr><td><code>HUGPY_MODEL_STORE_REAPABLE</code></td><td>off</td><td>Opt-in: this box's model store is local &amp; disposable, so stale/corrupt files may be deleted and re-pulled. Only takes effect when not on shared storage.</td><td>1/true/yes/on</td></tr>
                <tr><td><code>HUGPY_SHARED_MODEL_STORE</code></td><td>off</td><td>Declares this box's model root IS the shared/central catalog volume — trips a hard "never delete" guard regardless of the reap flag.</td><td>1/true/yes/on</td></tr>
                <tr><td><code>HUGPY_STAGING_ORPHAN_GRACE_SECONDS</code></td><td>600</td><td>Grace period before an orphaned <code>.tmp-&lt;pid&gt;</code> staging dir (definitely-dead owner pid) is reaped.</td><td>int seconds</td></tr>
                <tr><td><code>HUGPY_DISCOVERY_ALLOW_SHRINK</code></td><td>off (guard active)</td><td>Force-acknowledge a &gt;50% drop in discovered models vs. the prior report; otherwise the drastic report is refused and stashed as <code>.shrunk</code>.</td><td>"1" allows the shrink to overwrite</td></tr>
                <tr><td><code>EXCLUDE_DIR_NAMES</code></td><td>".cache,.git,.locks,snapshots,blobs,refs,1_Pooling,2_Normalize,onnx,legacy"</td><td>Directory names excluded from model-store discovery walks ("legacy" = the operator's data-hoarding store, never indexed).</td><td>comma-separated list</td></tr>
                <tr><td><code>EXCLUDE_DIR_PREFIXES</code></td><td>"models--"</td><td>Directory name prefixes excluded from discovery (HF cache root naming).</td><td>comma-separated list</td></tr>
                <tr><td><code>HUGPY_MARKER</code></td><td>"hugpy.json"</td><td>Filename of the per-model marker/metadata JSON dropped in each model dir.</td><td>filename string</td></tr>
                <tr><td><code>MODELS_DISCOVERY_PATH</code></td><td><code>&lt;PROJECTS_HOME&gt;/model_discovery.json</code></td><td>Path to the model-discovery report JSON.</td><td>filesystem path</td></tr>
                <tr><td><code>MODELS_DICT_PATH</code></td><td><code>&lt;PROJECTS_HOME&gt;/model_manifest.json</code></td><td>Path to the model manifest/dict JSON.</td><td>filesystem path</td></tr>
                <tr><td><code>PROJECTS_PLACEMENT_PATH</code></td><td><code>&lt;PROJECTS_HOME&gt;/placement.json</code></td><td>Path to the static-peer placement-tracking JSON.</td><td>filesystem path</td></tr>
                <tr><td><code>WHISPER_MODEL_DIR</code> <span className="docs-note">(alias <code>WHISPER_DOWNLOAD_ROOT</code>)</span></td><td>none</td><td>Override dir for the cached openai-whisper <code>.pt</code> weights (avoids pointing the loader at the HF transformers safetensors dir).</td><td>filesystem path</td></tr>
                <tr><td><code>DEFAULT_LOCAL_FILES_ONLY</code></td><td>true</td><td>Whether HF loads default to <code>local_files_only</code> (no network) unless explicitly overridden.</td><td>0/1/true/false/yes/no/on/off</td></tr>
                <tr><td><code>HUGPY_AUTO_DOWNLOAD</code></td><td>true</td><td>Kill-switch for staple downloads at <code>resolve()</code> time; a fresh install auto-pulls the curated model fleet on first use unless disabled (set false on air-gapped boxes/workers).</td><td>0/1/true/false/yes/no/on/off</td></tr>
              </tbody>
            </table>
            </EnvGroup>

            <EnvGroup id="env-discovery" label="Discovery, Hugging Face & Civitai">
            <table className="docs-table docs-table--env">
              <thead><tr><th>Variable</th><th>Default</th><th>Purpose</th><th>Values</th></tr></thead>
              <tbody>
                <tr><td><code>HF_TOKEN</code></td><td>none (anonymous)</td><td>HF Hub auth token for the shared <code>HfApi</code> client — needed only for gated/private repos or higher rate limits.</td><td>HF token string</td></tr>
                <tr><td><code>CIVITAI_API_TOKEN</code></td><td>none</td><td>Civitai API bearer token for authenticated checkpoint downloads.</td><td>token string</td></tr>
                <tr><td><code>GITHUB_TOKEN</code></td><td>none</td><td>GitHub API token to raise/avoid rate limits when fetching llama.cpp release assets. <span className="docs-note">(GitHub PAT convention, not hugpy-specific.)</span></td><td>token string</td></tr>
              </tbody>
            </table>
            </EnvGroup>

            <EnvGroup id="env-model-defaults" label="Per-task model defaults & overrides">
            <table className="docs-table docs-table--env">
              <thead><tr><th>Variable</th><th>Default</th><th>Purpose</th><th>Values</th></tr></thead>
              <tbody>
                <tr><td><code>DEFAULT_CHAT_MODEL</code></td><td>"Qwen2.5-3B-Instruct-GGUF"</td><td>Stock default chat model (the model a local slot serves when a request arrives without naming one).</td><td>model-key string</td></tr>
                <tr><td><code>DEFAULT_VISION_MODEL</code></td><td>"Qwen2.5-VL-3B-Instruct-GGUF"</td><td>Stock default vision (image-text-to-text) model.</td><td>model-key string</td></tr>
                <tr><td><code>DEFAULT_WHISPER_MODEL</code></td><td>"whisper-large-v3-turbo"</td><td>Stock default speech-to-text model.</td><td>model-key string</td></tr>
                <tr><td><code>DEFAULT_SUMMARIZE_MODEL</code></td><td>"flan-t5-large"</td><td>Stock default summarization model.</td><td>model-key string</td></tr>
                <tr><td><code>DEFAULT_EMBED_MODEL</code></td><td>"all-minilm-l6-v2"</td><td>Stock default embedding model.</td><td>model-key string</td></tr>
                <tr><td><code>DEFAULT_IMAGEGEN_MODEL</code></td><td>"sd-turbo"</td><td>Stock default image-generation model.</td><td>model-key string</td></tr>
                <tr><td><code>DEFAULT_KEYWORDS_MODEL</code></td><td>"all-minilm-l6-v2"</td><td>Stock default keyword-extraction model.</td><td>model-key string</td></tr>
                <tr><td><code>DEFAULT_DEPTH_MODEL</code></td><td>"depth-anything-v2-small"</td><td>Stock default depth-estimation model.</td><td>model-key string</td></tr>
                <tr><td><code>DEFAULT_DETECT_MODEL</code></td><td>"detr-resnet-50"</td><td>Stock default object-detection model.</td><td>model-key string</td></tr>
                <tr><td><code>DEFAULT_IMG_CLASSIFY_MODEL</code></td><td>"vit-base-patch16-224"</td><td>Stock default image-classification model.</td><td>model-key string</td></tr>
                <tr><td><code>DEFAULT_SEGMENT_MODEL</code></td><td>"segformer-b0-ade"</td><td>Stock default segmentation model.</td><td>model-key string</td></tr>
                <tr><td><code>DEFAULT_MODEL_KEY</code></td><td>none</td><td>Discord bot's default model key when a user hasn't chosen one.</td><td>model-key string</td></tr>
                <tr><td><code>MODEL_&lt;KEY&gt;</code></td><td>none</td><td>Dynamic per-model-key override (key upper-cased) forcing a model's resolved path to an explicit location.</td><td>filesystem path</td></tr>
                <tr><td><code>&lt;NAME&gt;_PORT</code> / <code>&lt;NAME&gt;_HOST</code></td><td>none</td><td>Dynamic per-discovered-model port/host override, keyed by the model's shortname (e.g. <code>MYMODEL_PORT</code>), merged into the discovery record.</td><td>int port / host string</td></tr>
              </tbody>
            </table>
            </EnvGroup>

            <EnvGroup id="env-dispatch" label="Chat/dispatch continuation & streaming">
            <table className="docs-table docs-table--env">
              <thead><tr><th>Variable</th><th>Default</th><th>Purpose</th><th>Values</th></tr></thead>
              <tbody>
                <tr><td><code>HUGPY_MAX_CONTINUATIONS</code> <span className="docs-note">(alias <code>WORKER_MAX_CONTINUATIONS</code>)</span></td><td>20</td><td>Max auto-continuation passes past the per-pass token cap before giving up (runaway guard).</td><td>int, unbounded (no code-side ceiling; 0 stops after the first pass)</td></tr>
                <tr><td><code>HUGPY_SEAM_WINDOW</code> <span className="docs-note">(alias <code>WORKER_SEAM_WINDOW</code>)</span></td><td>400</td><td>Max chars of overlap dropped at a continuation "seam" (model re-emitting the tail of the previous part).</td><td>int chars, unbounded (no code-side ceiling)</td></tr>
                <tr><td><code>HUGPY_PER_PASS_MAX_TOKENS</code></td><td>16000</td><td>Per-pass output token ceiling forwarded to a runner in the continuation loop (also clamps a version-skewed older worker that raises instead of clamping on over-cap).</td><td>int tokens, unbounded (no code-side ceiling; lower it to match a smaller worker cap)</td></tr>
                <tr><td><code>HUGPY_MAX_CHUNKS</code></td><td>256</td><td>Ceiling on continuation chunks for the "unbounded" chat/generate streaming paths (<code>base_runner.py</code>, <code>generate_runner.py</code>) — the hard backstop behind the loop guard below.</td><td>int, unbounded (no code-side ceiling on the value itself; this IS the loop's own ceiling)</td></tr>
                <tr><td><code>HUGPY_LOOP_GUARD_MAX_REPEAT</code></td><td>2</td><td>Consecutive near-identical continue-passes required before the anti-repetition guard finalizes the stream.</td><td>int, floored at 1 (<code>max(1, value)</code> in code — a value &lt;1 is silently raised to 1)</td></tr>
                <tr><td><code>HUGPY_LOOP_GUARD_SIMILARITY</code></td><td>0.95</td><td>Similarity ratio (<code>SequenceMatcher.ratio()</code>) above which two consecutive chunks count as a repeat for the loop guard.</td><td>float, unclamped by code but only 0.0–1.0 is meaningful (ratio() never exceeds 1.0)</td></tr>
                <tr><td><code>HUGPY_SSE_HEARTBEAT_SECS</code></td><td>15</td><td>Interval between SSE keepalive comments during long generation gaps, so proxy read-timeouts don't cut the stream.</td><td>float seconds</td></tr>
                <tr><td><code>DEFAULT_TIMEOUT</code></td><td>3600</td><td>Default request/operation timeout used across the imports/config layer.</td><td>float seconds</td></tr>
                <tr><td><code>DEFAULT_MAX_TOKENS</code></td><td>32768</td><td>Default max generation length fallback (chat/model schema defaults).</td><td>int; downstream consumers commonly clamp it, e.g. the vision request schema enforces <code>gt=0, le=32768</code></td></tr>
                <tr><td><code>DEFAULT_TEMPERATURE</code></td><td>0.1</td><td>Default LLM sampling temperature.</td><td>float, unclamped by code (0.0–2.0 is the conventional sane range most backends accept)</td></tr>
                <tr><td><code>DEFAULT_TOP_P</code></td><td>1</td><td>Default nucleus-sampling top_p.</td><td>float, unclamped by code (0.0–1.0 is the only meaningful range)</td></tr>
                <tr><td><code>MIN_INPUT_WORDS_DEFAULT</code></td><td>10</td><td>Minimum input word count below which <code>summarize()</code> (<code>InputPolicy.STRICT</code>) rejects a request as too short for a meaningful summary (<code>SummaryRequest.check_input()</code>). <span className="docs-note">CONFIRMED consumer, but the env read is dead in practice: <code>imports/src/constants/constants.py</code> reads it via <code>get_env_value(...)</code>, but <code>imports/src/schemas/summarizer_schemas.py</code> immediately re-defines the same name as a hardcoded literal (<code>= 10</code>) after its wildcard import, shadowing the env-derived value before it ever reaches <code>SummaryRequest.min_input_words</code>. Setting this env var today has no runtime effect; per-request <code>min_input_words</code> is the only live override.</span></td><td>int, ≥0 (per-request override is pydantic <code>Field(ge=0)</code>); env-level default is currently unreachable</td></tr>
              </tbody>
            </table>
            </EnvGroup>

            <EnvGroup id="env-uploads" label="Uploads, CORS & downloads">
            <table className="docs-table docs-table--env">
              <thead><tr><th>Variable</th><th>Default</th><th>Purpose</th><th>Values</th></tr></thead>
              <tbody>
                <tr><td><code>HUGPY_UI_DIST</code></td><td>auto-discovered (<code>console_dist</code> in the wheel, else dev <code>ui/dist</code>)</td><td>Override path to the built console UI (SPA) dist directory to serve.</td><td>filesystem path</td></tr>
                <tr><td><code>HUGPY_ALLOWED_ORIGINS</code></td><td>unset (reflect-any)</td><td>CORS allowlist tightening cross-origin browser callers on a public deployment.</td><td>comma-separated origin URLs</td></tr>
                <tr><td><code>HUGPY_MAX_UPLOAD_MB</code></td><td>100</td><td>Flask <code>MAX_CONTENT_LENGTH</code> cap (bounds <code>/uploads</code> and any POST body) to prevent unbounded memory/disk DoS.</td><td>float/int MB</td></tr>
                <tr><td><code>HUGPY_DOWNLOAD_STALL_SECONDS</code></td><td>180</td><td>Seconds with no new bytes written before a model download is considered stalled (killed + resumed).</td><td>int seconds</td></tr>
                <tr><td><code>HUGPY_DOWNLOAD_MAX_ATTEMPTS</code></td><td>4</td><td>Max resume attempts for a stalled/failed download.</td><td>int count</td></tr>
                <tr><td><code>HUGPY_PULL_CONCURRENCY</code></td><td>8</td><td>Max simultaneous byte-range connections for a single model-file transfer.</td><td>int, ≥1</td></tr>
                <tr><td><code>HUGPY_CHUNK_BYTES</code></td><td>32 MiB</td><td>Per-chunk size for the content-verified chunked transfer/resume mechanism.</td><td>int bytes, clamped 4–256 MiB</td></tr>
              </tbody>
            </table>
            </EnvGroup>

            <EnvGroup id="env-vision" label="Vision, image-gen & ComfyUI">
            <table className="docs-table docs-table--env">
              <thead><tr><th>Variable</th><th>Default</th><th>Purpose</th><th>Values</th></tr></thead>
              <tbody>
                <tr><td><code>HUGPY_VISION_INPROCESS</code></td><td>off on central; forced on for workers (they have no separate vision server)</td><td>Forces the in-process vision backend instead of the HTTP vision-server backend.</td><td>1/true/yes/on</td></tr>
                <tr><td><code>VISION_HOST</code></td><td><code>http://127.0.0.1</code></td><td>Default host baked into <code>VisionBackendConfig</code> for the HTTP vision-server backend (only consulted when <code>HUGPY_VISION_INPROCESS</code> is off).</td><td>URL</td></tr>
                <tr><td><code>HUGPY_IMG2IMG_QUANTIZE</code></td><td>"auto"</td><td>Whether an oversized img2img model is bnb 4-bit quantized + CPU-offloaded on load to fit VRAM.</td><td>"auto" | "always" | "never"</td></tr>
                <tr><td><code>COMFY_URL</code></td><td><code>http://127.0.0.1:8188</code></td><td>Base URL of the operator-adopted ComfyUI instance this box drives; a console <code>comfy_url</code> setting is projected onto this env var on a worker.</td><td>URL</td></tr>
                <tr><td><code>COMFY_TIMEOUT_S</code></td><td>600</td><td>HTTP timeout for calls to the ComfyUI backend.</td><td>float seconds</td></tr>
                <tr><td><code>COMFY_IPADAPTER_WEIGHT</code> / <code>COMFY_CLIPVISION_MODEL</code></td><td>per-family default filename</td><td>Override filenames for ComfyUI's IP-Adapter / CLIP-Vision weight files.</td><td>filename string</td></tr>
                <tr><td><code>COMFY_CHECKPOINTS_DIR</code></td><td><code>~/ComfyUI/models/checkpoints</code></td><td>Where a worker's adopted ComfyUI loads checkpoints from, for symlink-provisioning comfy models.</td><td>filesystem path</td></tr>
              </tbody>
            </table>
            </EnvGroup>

            <EnvGroup id="env-pools" label="ML task routing & worker pools">
            <table className="docs-table docs-table--env">
              <thead><tr><th>Variable</th><th>Default</th><th>Purpose</th><th>Values</th></tr></thead>
              <tbody>
                <tr><td><code>HUGPY_ML_POOL</code></td><td>"" (disabled)</td><td>Reserved worker-pool tag <code>/ml/*</code> endpoints route to.</td><td>pool name/tag string</td></tr>
                <tr><td><code>HUGPY_ML_GENERAL_ROUTE_TASKS</code></td><td>"image-text-to-text"</td><td>Tasks that use the general established worker route instead of the reserved ML pool.</td><td>comma-separated task-name list</td></tr>
                <tr><td><code>WORKER_POOL</code></td><td>"" (general pool)</td><td>Dedicated pool label this worker registers/heartbeats under. A pooled worker serves only requests tagged for its pool; general traffic never lands on it.</td><td>free-text pool label</td></tr>
                <tr><td><code>WORKER_PRELOAD</code></td><td>—</td><td>Retired: the worker no longer warms a model at assignment time. A model loads into VRAM/RAM only when a request for it arrives. This variable is ignored.</td><td>(ignored)</td></tr>
              </tbody>
            </table>
            </EnvGroup>

            <EnvGroup id="env-worker-identity" label="Worker — identity & heartbeat">
            <table className="docs-table docs-table--env">
              <thead><tr><th>Variable</th><th>Default</th><th>Purpose</th><th>Values</th></tr></thead>
              <tbody>
                <tr><td><code>WORKER_NAME</code></td><td>hostname</td><td>Display/registration name for this worker in the console.</td><td>free-text string</td></tr>
                <tr><td><code>WORKER_HOST</code></td><td><code>0.0.0.0</code></td><td>Bind address for the worker's inference/HTTP server.</td><td>IP/hostname</td></tr>
                <tr><td><code>WORKER_PORT</code></td><td>9100</td><td>Listen port for the worker HTTP server.</td><td>int port, 1–65535 (not clamped in code)</td></tr>
                <tr><td><code>WORKER_URL</code></td><td><code>http://&lt;host&gt;:&lt;port&gt;</code></td><td>Callback URL central should use when source-IP autodetect is wrong (e.g. NAT).</td><td>URL</td></tr>
                <tr><td><code>WORKER_MODELS</code></td><td>"" / none</td><td>Comma-separated model_keys to self-assign on registration.</td><td>comma-separated list</td></tr>
                <tr><td><code>WORKER_ID_FILE</code></td><td><code>~/.abstract_hugpy_worker.json</code> (gguf_worker: <code>~/.gguf_worker.json</code>)</td><td>Path persisting this worker's central-assigned worker_id across restarts.</td><td>filesystem path</td></tr>
                <tr><td><code>WORKER_HEARTBEAT</code></td><td>15</td><td>Heartbeat interval (seconds) sent to central.</td><td>float seconds</td></tr>
                <tr><td><code>WORKER_ROLE</code></td><td>"worker"</td><td>This box's fleet role: whole-model server vs. GPU lender to a shard pool.</td><td>"worker" | "rpc"</td></tr>
                <tr><td><code>WORKER_RPC_HOST</code> / <code>WORKER_RPC_PORT</code></td><td><code>0.0.0.0</code> / 50052</td><td>Bind address/port for llama.cpp's rpc-server (role=rpc).</td><td>IP/hostname / int port</td></tr>
                <tr><td><code>WORKER_RPC_BIN</code> <span className="docs-note">(alias <code>LLAMA_RPC_BIN</code>)</span></td><td>resolved via the engine resolver, else "rpc-server" on PATH</td><td>Path to the llama.cpp rpc-server binary.</td><td>filesystem path or bare command name</td></tr>
                <tr><td><code>LLAMA_SERVER_BIN</code></td><td>resolved via engine dir/PATH</td><td>Explicit override for the <code>llama-server</code> binary path, checked before the fetched-engine dir and PATH.</td><td>filesystem path</td></tr>
                <tr><td><code>LLAMA_CLI_BIN</code></td><td>resolved via engine dir/PATH</td><td>Explicit override for the <code>llama-cli</code> binary path.</td><td>filesystem path</td></tr>
                <tr><td><code>USER</code> / <code>LOGNAME</code></td><td>"" </td><td>Which OS user to <code>loginctl enable-linger</code> for, so the per-user worker systemd unit survives logout/reboot. <span className="docs-note">(Standard POSIX vars, not hugpy-specific.)</span></td><td>OS username</td></tr>
              </tbody>
            </table>
            </EnvGroup>

            <EnvGroup id="env-worker-update" label="Worker — self-update & enrollment">
            <table className="docs-table docs-table--env">
              <thead><tr><th>Variable</th><th>Default</th><th>Purpose</th><th>Values</th></tr></thead>
              <tbody>
                <tr><td><code>WORKER_PKG_NAME</code> <span className="docs-note">(mirrors <code>HUGPY_PKG_NAME</code>)</span></td><td>"abstract_hugpy_dev"</td><td>Pip distribution name this worker tracks/self-updates; must match what central advertises.</td><td>pip distribution name</td></tr>
                <tr><td><code>WORKER_PKG_INDEX</code></td><td>PyPI</td><td>Override <code>pip --index-url</code> for self-update; used for restricted-egress workers pointed at central's own simple index.</td><td>URL (PEP-503 simple index)</td></tr>
                <tr><td><code>WORKER_SELF_UPDATE</code></td><td>on</td><td>Enables/disables auto-update from central (pip install -U + re-exec).</td><td>0/false/no/off disables</td></tr>
                <tr><td><code>WORKER_ENV_TIER</code></td><td>"stable"</td><td>Names which venv/library tier this worker unit runs; central routes tier-mapped models only to matching workers.</td><td>"stable" | "edge" (free text, lower-cased)</td></tr>
                <tr><td><code>HUGPY_PKG_NAME</code></td><td>"abstract_hugpy_dev"</td><td>Distribution name central tracks/reports the version of.</td><td>package name</td></tr>
                <tr><td><code>HUGPY_REQUIRED_PKG_VERSION</code></td><td>none</td><td>Operator pin of the version workers should converge to (advertised in every heartbeat).</td><td>version string (PEP 440)</td></tr>
                <tr><td><code>HUGPY_REQUIRED_PKG_VERSION_FILE</code></td><td><code>&lt;manifest_dir&gt;/required_pkg_version</code></td><td>Path to a file holding the pinned version, used if the env var itself is unset.</td><td>filesystem path</td></tr>
                <tr><td><code>HUGPY_PKG_INDEX_DIR</code></td><td><code>&lt;manifest_dir&gt;/pip_index</code></td><td>Directory of built wheels central serves as a PEP-503 pip index for worker self-update.</td><td>filesystem path</td></tr>
                <tr><td><code>HUGPY_MODEL_ENV_TIERS</code></td><td>"" </td><td>Operator map of model_key to required runtime-environment tier, gating which workers may serve a model.</td><td>"key:tier,…" e.g. "Qwen3.6-27B-AEON:edge"</td></tr>
              </tbody>
            </table>
            </EnvGroup>

            <EnvGroup id="env-worker-caps" label="Worker — resource caps, gen-gate & lifecycle">
            <table className="docs-table docs-table--env">
              <thead><tr><th>Variable</th><th>Default</th><th>Purpose</th><th>Values</th></tr></thead>
              <tbody>
                <tr><td><code>HUGPY_INPROCESS_MAX_CONCURRENCY</code></td><td>1</td><td>Safe concurrent entrants into one in-process (llama-cpp-python / transformers) model runner (llama.cpp contexts aren't concurrency-safe).</td><td>int, ≥1</td></tr>
                <tr><td><code>HUGPY_WORKER_GEN_GATE_TIMEOUT_S</code></td><td>120</td><td>Bounded wait for the per-model gate to free before returning a structured "busy" error.</td><td>float seconds, ≥0</td></tr>
                <tr><td><code>HUGPY_WORKER_GEN_GATE</code></td><td>on</td><td>Emergency escape hatch to disable the in-process generation gate entirely.</td><td>off/0/false/no disables</td></tr>
                <tr><td><code>HUGPY_STUDIO_QUEUE_DEPTH</code></td><td>4</td><td>Bounded FIFO depth for Studio-render jobs waiting behind the one in flight; beyond this the worker returns 409.</td><td>int, &gt;0 (a non-positive or unparseable value silently falls back to the default of 4)</td></tr>
                <tr><td><code>HUGPY_WORKER_RESTART_DRAIN_S</code></td><td>30</td><td>Bound on how long a restart waits for in-flight generations to drain before exiting anyway.</td><td>float seconds, ≥0</td></tr>
                <tr><td><code>HUGPY_WORKER_SYSTEMD</code></td><td>auto-detect</td><td>Explicit override forcing the systemd-vs-standalone restart-path decision (exit+respawn vs re-exec in place).</td><td>1/true/yes/on = true; only consulted if non-empty</td></tr>
                <tr><td><code>WORKER_SERVICE</code></td><td>"auto"</td><td>Distinct from <code>HUGPY_WORKER_SYSTEMD</code> above: an <em>install-time</em> flag to <code>hugpy worker</code>'s installer choosing which OS auto-start mechanism to wire up, not a runtime restart-path override.</td><td>"auto" | "systemd" | "launchd" | "schtasks" | "foreground" | "none"</td></tr>
                <tr><td><code>INVOCATION_ID</code> / <code>NOTIFY_SOCKET</code></td><td>— </td><td>systemd-standard presence signals used (alongside parent-process confirmation) to detect running-under-systemd. <span className="docs-note">(systemd-standard, not hugpy-specific.)</span></td><td>presence/absence only</td></tr>
                <tr><td><code>HUGPY_FOREIGN_CALL_TTL_S</code></td><td>1800</td><td>TTL an unclosed foreign-call PID attribution (e.g. a ComfyUI generation whose end was missed) stays trusted before going stale.</td><td>float seconds</td></tr>
              </tbody>
            </table>
            </EnvGroup>

            <EnvGroup id="env-engine" label="Native engine (llama.cpp) build & fetch">
            <table className="docs-table docs-table--env">
              <thead><tr><th>Variable</th><th>Default</th><th>Purpose</th><th>Values</th></tr></thead>
              <tbody>
                <tr><td><code>HUGPY_ENGINE_REPO</code></td><td>"ggml-org/llama.cpp"</td><td>GitHub repo to fetch prebuilt llama.cpp release assets from.</td><td>"owner/repo"</td></tr>
                <tr><td><code>HUGPY_ENGINE_TAG</code></td><td>"latest" (fetch) / none (source build = default branch)</td><td>Which GitHub release tag / git ref to fetch or clone.</td><td>tag string or "latest"</td></tr>
                <tr><td><code>HUGPY_ENGINE_GIT_URL</code></td><td>"https://github.com/ggml-org/llama.cpp.git"</td><td>Git remote to clone from source when no prebuilt asset matches.</td><td>git URL</td></tr>
              </tbody>
            </table>
            </EnvGroup>

            <EnvGroup id="env-platform" label="Cross-platform paths & gguf_worker">
            <table className="docs-table docs-table--env">
              <thead><tr><th>Variable</th><th>Default</th><th>Purpose</th><th>Values</th></tr></thead>
              <tbody>
                <tr><td><code>LOCALAPPDATA</code></td><td><code>~/AppData/Local</code></td><td>Windows per-user local app data root, used only if the <code>platformdirs</code> dependency is unavailable. <span className="docs-note">(Windows-standard, not hugpy-specific.)</span></td><td>filesystem path</td></tr>
                <tr><td><code>XDG_DATA_HOME</code> / <code>XDG_CONFIG_HOME</code> / <code>XDG_CACHE_HOME</code></td><td><code>~/.local/share</code> / <code>~/.config</code> / <code>~/.cache</code></td><td>Linux XDG Base Directory roots, same <code>platformdirs</code>-unavailable fallback. <span className="docs-note">(freedesktop.org XDG standard, not hugpy-specific.)</span></td><td>filesystem path</td></tr>
                <tr><td><code>GGUF_WORKER_MODELS_DIR</code></td><td><code>~/gguf-models</code></td><td>Directory the slim ARM/Termux GGUF worker stores/loads model files from.</td><td>filesystem path</td></tr>
                <tr><td><code>GGUF_WORKER_N_CTX</code></td><td>4096</td><td>llama.cpp context window for the loaded model.</td><td>int tokens</td></tr>
                <tr><td><code>GGUF_WORKER_N_THREADS</code></td><td>0 (auto)</td><td>llama.cpp CPU thread count.</td><td>int (0 = auto)</td></tr>
                <tr><td><code>GGUF_WORKER_N_GPU_LAYERS</code></td><td>0</td><td>Layers offloaded to GPU for this standalone worker.</td><td>int, unbounded (0 = CPU-only, -1 = all layers, per llama.cpp convention; not clamped here)</td></tr>
              </tbody>
            </table>
            </EnvGroup>

            <EnvGroup id="env-phonebrick" label="Phone-brick fleet">
            <table className="docs-table docs-table--env">
              <thead><tr><th>Variable</th><th>Default</th><th>Purpose</th><th>Values</th></tr></thead>
              <tbody>
                <tr><td><code>MODEL_PATH</code></td><td>bundled default</td><td>Path to the ONNX YOLO detector weights this PPE-vision worker loads.</td><td>filesystem path</td></tr>
                <tr><td><code>YOLO_IMGSZ</code></td><td>640</td><td>Inference image size for the ONNX YOLO detector.</td><td>int pixels, unbounded (YOLO convention favors multiples of 32; not clamped here)</td></tr>
                <tr><td><code>YOLO_CONF</code></td><td>0.25</td><td>Confidence threshold for reported YOLO detections.</td><td>float 0.0–1.0</td></tr>
                <tr><td><code>HOST</code> / <code>PORT</code></td><td><code>0.0.0.0</code> / 5002</td><td>Bind host/port for the phone-brick worker's Flask app.</td><td>host / int port</td></tr>
                <tr><td><code>PHONE_BRICK_ENABLE_SHELL</code></td><td>off</td><td>Enables a dangerous <code>sh &lt;command&gt;</code> verb (arbitrary shell execution) on the worker's job queue.</td><td>1/true/yes/on</td></tr>
                <tr><td><code>PHONE_BRICK_CENTRAL</code> <span className="docs-note">(alias <code>PHONE_BRICK_CENTRAL_URL</code>)</span></td><td>unset = registration is a no-op</td><td>Base URL of the console/central this phone announces to and heartbeats.</td><td>URL</td></tr>
                <tr><td><code>PHONE_BRICK_NAME</code></td><td>hostname</td><td>Display name for this phone in the console's phone-brick pool.</td><td>free-text string</td></tr>
                <tr><td><code>PHONE_BRICK_COLOR</code></td><td>"#58a6ff"</td><td>Box colour used to draw this phone's detections in the console UI.</td><td>hex color</td></tr>
                <tr><td><code>PHONE_BRICK_HEARTBEAT</code></td><td>20</td><td>Seconds between heartbeat POSTs reporting live status.</td><td>float seconds</td></tr>
                <tr><td><code>PHONE_BRICK_FILE_SERVER</code></td><td>derived from central + <code>/api/phone-brick/files/</code></td><td>Base URL phones fetch a seeded image from.</td><td>full URL ending in "/"</td></tr>
                <tr><td><code>PHONE_BRICK_OUTPUT_DIR</code></td><td><code>&lt;registry_dir&gt;/phone_brick_runs</code></td><td>Directory the orchestrator seeds images into and phones write annotated results back to.</td><td>filesystem path</td></tr>
                <tr><td><code>PHONE_BRICK_RPC</code></td><td>off</td><td>Master switch: turns a phone into a llama.cpp rpc-server shard backend, advertising its RAM as pseudo-VRAM to the GPU worker pool.</td><td>1/true/yes/on</td></tr>
                <tr><td><code>PHONE_BRICK_RPC_RAM_GIB</code></td><td>auto-probed free RAM, else 2 GiB</td><td>RAM budget this phone advertises as poolable pseudo-VRAM.</td><td>float GiB</td></tr>
                <tr><td><code>PHONE_BRICK_RPC_HOST</code></td><td>auto (local IP toward central, else hostname)</td><td>Host advertised in the <code>rpc_endpoint</code> the shard-plan lead connects to.</td><td>hostname/IP</td></tr>
                <tr><td><code>PHONE_BRICK_RPC_PORT</code></td><td>50052</td><td>Port rpc-server binds/advertises.</td><td>int port, 1–65535 (not clamped in code)</td></tr>
                <tr><td><code>PHONE_BRICK_RPC_BIN</code></td><td>resolved via engine resolver, else "rpc-server" on PATH</td><td>Path to the llama.cpp rpc-server binary.</td><td>filesystem path or bare name</td></tr>
              </tbody>
            </table>
            </EnvGroup>

            <EnvGroup id="env-studio" label="Identity-render & Studio video">
            <table className="docs-table docs-table--env">
              <thead><tr><th>Variable</th><th>Default</th><th>Purpose</th><th>Values</th></tr></thead>
              <tbody>
                <tr><td><code>IDENTITY_RENDER_URL</code></td><td>"" (not configured)</td><td>Base URL of the remote identity-render GPU service (mesh + 360° turntable build) — central itself has no GPU.</td><td>URL</td></tr>
                <tr><td><code>IDENTITY_RENDER_TOKEN</code></td><td>"" </td><td>Bearer token sent as <code>X-Identity-Render-Token</code> to the remote render service. <span className="docs-note">Read on BOTH ends: central sends it, and the standalone <code>identity_render_service</code> itself reads the SAME name server-side (<code>service.py</code>) to validate the incoming header — refuses to boot if unset there.</span></td><td>opaque token</td></tr>
                <tr><td><code>IDENTITY_RENDER_POLL_INTERVAL_S</code></td><td>5</td><td>Seconds between polls of the remote render job.</td><td>float seconds</td></tr>
                <tr><td><code>IDENTITY_RENDER_DEADLINE_S</code></td><td>14100</td><td>Whole-job poll deadline (just under the bus's 14400s registered timeout) before the runner gives up with a retryable timeout.</td><td>float seconds</td></tr>
                <tr><td><code>IDENTITY_FRONT_VISION_CONNECT_TIMEOUT_S</code> / <code>IDENTITY_FRONT_VISION_READ_TIMEOUT_S</code></td><td>10 / 60</td><td>Connect/read timeout budgets for per-candidate <code>/ml/vision</code> calls during fleet-VLM front-view auto-selection.</td><td>float seconds</td></tr>
                <tr><td><code>IDENTITY_FRONT_AUTOSELECT</code></td><td>on</td><td>Kill-switch for fleet-VLM auto-selection of the best "front view" reference image for mesh building.</td><td>off/0/false disables</td></tr>
                <tr><td><code>HUGPY_CENTRAL_URL</code></td><td><code>http://127.0.0.1:7002</code></td><td>Central's own base URL, used to reach the local <code>/ml/vision</code> amenity during front-view auto-select (distinct from <code>IDENTITY_RENDER_URL</code>, the remote render service).</td><td>URL</td></tr>
                <tr><td><code>STUDIO_OUTPUT_ROOT</code></td><td>required — boot fails if missing</td><td>Root dir where Studio renders write output artifacts.</td><td>filesystem path</td></tr>
                <tr><td><code>STUDIO_WEIGHTS_ROOT</code></td><td>required</td><td>Root dir of pinned Studio model weights (Wan i2v/t2v/vace etc).</td><td>filesystem path</td></tr>
                <tr><td><code>STUDIO_WEIGHTS_HOT_ROOT</code></td><td>none (falls back to <code>STUDIO_WEIGHTS_ROOT</code>)</td><td>Per-box NVMe "hot copy" weights root, checked first; live-process-env only (never in the manifest hash, so it can't affect a clip's content hash).</td><td>filesystem path</td></tr>
                <tr><td><code>STUDIO_MANIFEST_ROOT</code></td><td>required</td><td>Root dir for RenderManifest JSON files.</td><td>filesystem path</td></tr>
                <tr><td><code>STUDIO_MASTER_COLORSPACE</code></td><td>required</td><td>Mastering colorspace Studio renders target (fails at boot, not at frame 900 with a green tint).</td><td>free-text string, no enum check in code — must match what the render/mastering pipeline actually expects (e.g. "rec709")</td></tr>
                <tr><td><code>STUDIO_MASTER_FPS</code></td><td>required</td><td>Master frame rate Studio renders target.</td><td>int, unbounded (no range check; loader only requires it be castable to <code>int</code>)</td></tr>
                <tr><td><code>STUDIO_MAX_VRAM_GB</code></td><td>required</td><td>GPU VRAM budget (GB) for the whole-GPU-vs-offload placement decision.</td><td>float GB, unbounded (no range check; loader only requires it be castable to <code>float</code>)</td></tr>
                <tr><td><code>STUDIO_LOUDNESS_LUFS</code></td><td>required</td><td>Target audio loudness (LUFS) for Studio's audio mastering stage.</td><td>float LUFS, unbounded in code (broadcast convention is roughly -30 to -14)</td></tr>
                <tr><td><code>STUDIO_ALLOW_UNPINNED</code></td><td>off</td><td>Allows runners to use unpinned (non-hash-verified) model weights; seed/dev may run unpinned, production must pin or set this deliberately.</td><td>"1" allows</td></tr>
                <tr><td><code>HUGPY_STUDIO_WORKER</code></td><td>"" (in-process)</td><td>Base URL of the one global Studio GPU worker a clip render delegates to.</td><td>URL</td></tr>
                <tr><td><code>HUGPY_STUDIO_FORCE_REMOTE</code></td><td>off</td><td>Test-only override forcing delegation to the remote worker even for a synthetic (non-real) model binding.</td><td>"1" forces</td></tr>
                <tr><td><code>HUGPY_STUDIO_POLL_INTERVAL_S</code></td><td>2.0</td><td>Cadence of status polls against a delegated Studio worker while a render runs.</td><td>float seconds</td></tr>
                <tr><td><code>HUGPY_STUDIO_DELEGATE_TIMEOUT_S</code></td><td>1800 (30 min)</td><td>Render budget once a delegated job is actually running on the worker (excludes queue-wait).</td><td>float seconds</td></tr>
                <tr><td><code>HUGPY_STUDIO_OVERALL_CAP_S</code></td><td>7200 (2 h)</td><td>Overall wall-clock ceiling for a whole delegation (queue wait + kickoff retries + render), so a wedged worker can never hang a job forever.</td><td>float seconds</td></tr>
                <tr><td><code>HUGPY_STUDIO_KICKOFF_RETRY_WINDOW_S</code> / <code>_INTERVAL_S</code></td><td>30 / 5</td><td>How long, and how often, to retry a render kickoff that failed with a connection reset (covers the post-worker-restart socket-converge window).</td><td>float seconds / float seconds</td></tr>
                <tr><td><code>HUGPY_VIDEOGEN_LOCAL</code></td><td>off</td><td>Permits in-process image/scene generation on central even when a worker fleet is registered but no live worker serves the requested model (else central refuses, to avoid melting itself loading a diffusion model in gunicorn).</td><td>always/1/true/yes/on</td></tr>
                <tr><td><code>HUGPY_SCENE_CONCURRENCY</code></td><td>self-tuned (online worker count for the model, floor 1)</td><td>Pins how many frames of a scene generate concurrently, overriding the self-tuned value.</td><td>int, clamped [1, n_frames]</td></tr>
                <tr><td colSpan="4"><span className="docs-note">The following are read by the standalone <code>identity_render_service</code> process itself (a separate repo/box — e.g. ae, <code>:9750</code>) that <code>IDENTITY_RENDER_URL</code> above points at, not by central.</span></td></tr>
                <tr><td><code>IDENTITY_RENDER_PORT</code></td><td>9750</td><td>Listen port for the identity-render HTTP service.</td><td>int port</td></tr>
                <tr><td><code>IDENTITY_RENDER_HOME</code></td><td><code>~/identity-render/jobs</code></td><td>Working dir the service reads/writes render jobs under.</td><td>filesystem path</td></tr>
                <tr><td><code>IDENTITY_RENDER_LOG_LEVEL</code></td><td>"INFO"</td><td>Python logging level for the service process.</td><td>"DEBUG" | "INFO" | "WARNING" | "ERROR" | "CRITICAL"</td></tr>
                <tr><td><code>BLENDER_BIN</code></td><td>"blender" (resolved on PATH)</td><td>Path/command for the Blender binary the 360° turntable render shells out to.</td><td>filesystem path or bare command name</td></tr>
                <tr><td><code>FFMPEG_BIN</code></td><td>"ffmpeg" (resolved on PATH)</td><td>Path/command for ffmpeg, used to assemble the turntable frames into an mp4.</td><td>filesystem path or bare command name</td></tr>
                <tr><td><code>HUNYUAN3D_MV_MODEL</code></td><td>"tencent/Hunyuan3D-2mv"</td><td>HF repo id for the multi-view mesh-build model.</td><td>HF repo id string</td></tr>
                <tr><td><code>HUNYUAN3D_MV_SUBFOLDER</code></td><td>"hunyuan3d-dit-v2-mv"</td><td>Subfolder within the HF repo holding the mesh-build weights.</td><td>path-segment string</td></tr>
                <tr><td><code>HUNYUAN3D_DEVICE</code></td><td>"cuda"</td><td>Torch device the Hunyuan3D-2mv mesh builder loads onto.</td><td>torch device string (e.g. "cuda", "cuda:0", "cpu")</td></tr>
                <tr><td><code>HUNYUAN3D_TEXTURE_MODEL</code></td><td>"tencent/Hunyuan3D-2"</td><td>HF repo id for the (currently unused-by-default, <code>texture=False</code>) texture-synthesis model.</td><td>HF repo id string</td></tr>
                <tr><td><code>IDENTITY_MESH_KEEP_LOADED</code></td><td>off</td><td>Keeps the mesh-build model resident across jobs instead of releasing it after each one; off is safest when the GPU also hosts image/video models.</td><td>1/true/yes/on keeps loaded; anything else releases after each job</td></tr>
              </tbody>
            </table>
            </EnvGroup>

            <EnvGroup id="env-misc" label="Discord bot, keeper REPL & peers">
            <table className="docs-table docs-table--env">
              <thead><tr><th>Variable</th><th>Default</th><th>Purpose</th><th>Values</th></tr></thead>
              <tbody>
                <tr><td><code>HUGPY_BOT_ENV</code></td><td>unset = python-dotenv searches upward from CWD</td><td>Explicit path to a <code>.env</code> file to load for the bot's config.</td><td>filesystem path</td></tr>
                <tr><td><code>HUGPY_BOT_DATA</code></td><td><code>~/.hugpy/bot</code></td><td>Writable data dir for per-user bot prefs — deliberately not inside the installed package.</td><td>filesystem path</td></tr>
                <tr><td><code>HUGPY_BOT_LEGACY_ENV</code></td><td><code>~/darnell/.env</code></td><td>Optional legacy "darnell" <code>.env</code>, kept only as a token fallback for old setups.</td><td>filesystem path</td></tr>
                <tr><td><code>GUILD_ID</code></td><td>none</td><td>The Discord guild (server) id the bot is scoped/synced to.</td><td>int (Discord snowflake)</td></tr>
                <tr><td><code>OPERATOR_DISCORD_ID</code></td><td>none</td><td>The operator's Discord user id; escalation button-clicks are fail-closed to this user.</td><td>int (Discord snowflake)</td></tr>
                <tr><td><code>HUGPY_BOT_MEMBERS_INTENT</code></td><td>off</td><td>Enables the privileged "Server Members" Discord gateway intent (must also be toggled in the Developer Portal).</td><td>1/true/yes/on</td></tr>
                <tr><td><code>DISCORD_TOKEN</code></td><td>none (raises if unset, unless a legacy darnell token file exists)</td><td>Discord bot token used to log in to the gateway.</td><td>opaque secret string</td></tr>
                <tr><td><code>KEEPER_OBS_LIMIT</code></td><td>6000</td><td>Max size of tool-observation output fed back into an agent turn (interactive REPL and headless Discord-bridge loop).</td><td>int chars, unbounded (no code-side ceiling)</td></tr>
                <tr><td><code>STATION_CONSOLE_UID</code> / <code>_GID</code> / <code>_HOME</code></td><td>1000 / 1000 / <code>/home/ubuntu</code></td><td>Default uid/gid/home the keeper's executor (e.g. an LXD exec context) runs actions as.</td><td>int / int / filesystem path</td></tr>
                <tr><td><code>KEEPER_MAX_STEPS</code></td><td>8</td><td>Default max agent-turn/tool-call steps per keeper invocation.</td><td>int, unbounded (no code-side ceiling)</td></tr>
                <tr><td><code>KEEPER_BRIDGE</code></td><td>unset = interactive REPL only</td><td>Discord-bridge id to attach to for headless relay (poll inbound, reply through central).</td><td>bridge id string</td></tr>
                <tr><td><code>KEEPER_BRIDGE_POLL</code></td><td>3</td><td>Seconds between polls for new inbound bridge messages.</td><td>float seconds</td></tr>
                <tr><td><code>KEEPER_BRIDGE_MODE</code></td><td>"user-strict"</td><td>Informational bridge reply-approval mode; central's own <code>defer_mode</code> is authoritative.</td><td>"user-strict" | "keeper-choice"</td></tr>
                <tr><td><code>KEEPER_MIRROR</code></td><td>unset</td><td>Also push interactive-REPL replies to this bridge id.</td><td>bridge id string</td></tr>
                <tr><td><code>LLM_PEER_ROLE</code></td><td>"central"</td><td>This node's role as reported in the peer registry entry.</td><td>free-form role string</td></tr>
                <tr><td><code>LLM_PEER_NAME</code></td><td>hostname</td><td>This node's display name in the peer registry.</td><td>free-text string</td></tr>
                <tr><td><code>HUGPY_CONSOLE_URL</code></td><td>none</td><td>URL of the delegated console UI, surfaced in the <code>/readiness</code> payload (the Landing page's console pointable).</td><td>URL</td></tr>
                <tr><td><code>HUGPY_DEEPCODER_URL</code></td><td>"https://hugpy.abstractendeavors.com"</td><td>Legacy hosted codegen endpoint — explicitly <em>not</em> hugpy central.</td><td>URL</td></tr>
              </tbody>
            </table>
            </EnvGroup>

            <EnvGroup id="env-internal" label="Internal schema constants (rarely need touching)">
            <p className="docs-note">Low-level literal-value registries for internal pydantic schemas; overriding them changes what the API layer will accept, not typical deployment config.</p>
            <table className="docs-table docs-table--env">
              <thead><tr><th>Variable</th><th>Default</th><th>Purpose</th><th>Values</th></tr></thead>
              <tbody>
                <tr><td><code>DISK_AUTHORITATIVE</code></td><td>"name,folder,framework,filename"</td><td>Fields for which on-disk discovery is authoritative over the registry when merging model config.</td><td>comma-separated list</td></tr>
                <tr><td><code>OVERLAY_ALLOWED</code></td><td>"port, host, timeout_s, include"</td><td>Fields a registry "overlay" is allowed to set on a discovered model.</td><td>comma-separated list</td></tr>
                <tr><td><code>SOURCEKIND</code></td><td>"text,url,file,image"</td><td>Allowed source-kind literal values for schemas.</td><td>comma-separated list</td></tr>
                <tr><td><code>JOBSTATUS</code></td><td>"queued,running,completed,failed,cancelled"</td><td>Allowed job-status literal values for schemas.</td><td>comma-separated list</td></tr>
                <tr><td><code>FINISH_REASONS</code></td><td>"stop,max_tokens,cancelled,error"</td><td>Allowed finish-reason literal values.</td><td>comma-separated list</td></tr>
                <tr><td><code>ROLES</code></td><td>"system,user,assistant"</td><td>Allowed chat-role literal values.</td><td>comma-separated list</td></tr>
                <tr><td><code>TOKENIZER_SENTINEL_THRESHOLD</code></td><td>10<sup>9</sup></td><td>Threshold above which a tokenizer's declared max length is treated as a "no real limit" sentinel rather than a genuine window.</td><td>float, unbounded (used only as an upper comparison bound: <code>0 &lt; raw &lt; threshold</code>)</td></tr>
              </tbody>
            </table>
            </EnvGroup>

            <EnvGroup id="env-frontend" label="React console apps (build-time)">
            <p className="docs-note">
              These are <strong>Vite build-time</strong> vars for the two standalone arm UIs (<code>video_intelligence_ui</code>,
              <code>media_intelligence_ui</code>) — baked into the bundle at <code>npm run build</code>, not readable at
              runtime. The main console (<code>ui/src</code>, <code>@hugpy/ui</code>) has none of these: it resolves its
              backend at runtime via <code>baseUrl</code>/<code>HugpyProvider</code>, same-origin <code>/api</code>, or
              <code>GET /api/auth/config</code> — never a <code>VITE_*</code> var.
            </p>
            <table className="docs-table docs-table--env">
              <thead><tr><th>Variable</th><th>Default</th><th>Purpose</th><th>Values</th></tr></thead>
              <tbody>
                <tr><td><code>VITE_HUGPY_API_BASE</code></td><td>a fixed dev-host default</td><td>Base API URL the standalone arm UI calls.</td><td>URL</td></tr>
                <tr><td><code>VITE_HUGPY_SITE_URL</code></td><td>"https://hugpy.ai"</td><td>Public brochure site link (CTA target).</td><td>URL</td></tr>
                <tr><td><code>VITE_HUGPY_DEMO_MEDIA_BASE</code> <span className="docs-note">(video_intelligence_ui only)</span></td><td>"https://hugpy.ai/demo-media"</td><td>Base URL the canned demo (<code>?demo=1</code>) loads its sample media from (media is hosted, not bundled). Runtime overrides also exist: <code>?mediaBase=</code> query param, then <code>window.__HUGPY_MEDIA_BASE__</code>, then this var.</td><td>URL or absolute path</td></tr>
                <tr><td><code>VITE_HUGPY_UPLOAD_URL</code></td><td><code>&lt;apiBase&gt;/uploads</code></td><td>Upload endpoint override.</td><td>URL</td></tr>
                <tr><td><code>VITE_HUGPY_POOL</code></td><td>"" (un-pooled/general)</td><td>Worker pool tag these requests are bound to (e.g. "ml").</td><td>pool name string</td></tr>
                <tr><td><code>VITE_HUGPY_WITH_CREDENTIALS</code></td><td>false</td><td>Send credentials (cookies) with API requests, once hugpy's CORS is credential-ready.</td><td>true/false</td></tr>
                <tr><td><code>VITE_HUGPY_SEND_REQUEST_ID</code></td><td>false</td><td>Attach a client-generated request-id header, once hugpy allows it.</td><td>true/false</td></tr>
                <tr><td><code>VITE_HUGPY_COMFY_URL</code></td><td>"" </td><td>ComfyUI endpoint override (a plain-HTTP localhost ComfyUI is blocked as mixed content on an HTTPS page, so this points at a reachable one).</td><td>URL</td></tr>
                <tr><td><code>VITE_HUGPY_REGISTRY_URL</code> <span className="docs-note">(media_intelligence_ui only)</span></td><td><code>&lt;apiBase&gt;/prompt/tasks</code></td><td>Task-registry endpoint override.</td><td>URL</td></tr>
                <tr><td><code>VITE_HUGPY_MEDIA_ANALYZE_URL</code> <span className="docs-note">(media_intelligence_ui only)</span></td><td><code>&lt;apiBase&gt;/media/analyze</code></td><td>Media-analysis endpoint override (e.g. a separate bridge host).</td><td>URL</td></tr>
                <tr><td colSpan="4"><span className="docs-note">The following 19 are <code>video_intelligence_ui</code>-only — each overrides exactly one API endpoint, defaulting to <code>&lt;apiBase&gt;/&lt;matching route&gt;</code>. See <code>video_intelligence_ui/src/config.ts</code> for the source of truth.</span></td></tr>
                <tr><td><code>VITE_HUGPY_VIDEO_INGEST_URL</code></td><td><code>&lt;apiBase&gt;/video/ingest</code></td><td>Resolve an uploaded server path into a typed MediaRef (native dims, mime, kind) — the first hop after upload for every media station.</td><td>URL</td></tr>
                <tr><td><code>VITE_HUGPY_CROP_ENQUEUE_URL</code></td><td><code>&lt;apiBase&gt;/video/jobs/crop</code></td><td>Enqueue a crop job (spatial and/or temporal axis) on the job bus.</td><td>URL</td></tr>
                <tr><td><code>VITE_HUGPY_FRAME_EXTRACT_URL</code></td><td><code>&lt;apiBase&gt;/video/jobs/frame_extract</code></td><td>Enqueue a frame-extract job (Frames &amp; Models station) — one job → many frames.</td><td>URL</td></tr>
                <tr><td><code>VITE_HUGPY_AUDIO_EXTRACT_URL</code></td><td><code>&lt;apiBase&gt;/video/jobs/audio_extract</code></td><td>Enqueue an audio-extract job (Audio Crop station) — pulls a video's audio track into one standalone audio MediaRef.</td><td>URL</td></tr>
                <tr><td><code>VITE_HUGPY_GENERATE_URL</code></td><td><code>&lt;apiBase&gt;/video/jobs/generate_image</code></td><td>Enqueue a text/image → image generate job (Generate station).</td><td>URL</td></tr>
                <tr><td><code>VITE_HUGPY_GENERATE_SCENE_URL</code></td><td><code>&lt;apiBase&gt;/video/jobs/generate_scene</code></td><td>Enqueue a text/image → scene generate job (Generate station, Scene mode) — one ordered multimodal prompt → N consecutive frames + an assembled mp4.</td><td>URL</td></tr>
                <tr><td><code>VITE_HUGPY_GENERATE_MOVIE_URL</code></td><td><code>&lt;apiBase&gt;/video/jobs/generate_movie</code></td><td>Enqueue a goal-timeline → movie generate job (Generate station, Movie mode) — an ordered, contiguous goal list tiling [0,total) → an N-segment movie.</td><td>URL</td></tr>
                <tr><td><code>VITE_HUGPY_PRESETS_URL</code></td><td><code>&lt;apiBase&gt;/video/presets</code></td><td>Curated default settings for the Generate station — knobs + model per preset; picking one selects that model and its knobs. It does not load the model — that happens when you generate.</td><td>URL</td></tr>
                <tr><td><code>VITE_HUGPY_MOVIE_PRESETS_URL</code></td><td><code>&lt;apiBase&gt;/movie/presets</code></td><td>Curated MOVIE templates for the Generate-station Movie tab — each preset is a bare array item carrying a whole shot list (goal timeline + settings).</td><td>URL</td></tr>
                <tr><td><code>VITE_HUGPY_STUDIO_I2V_URL</code></td><td><code>&lt;apiBase&gt;/video/studio/i2v</code></td><td>Enqueue a Studio image-to-video clip (Studio Clips viewer) — runs through the cinema-studio spine to a content-addressed clip.</td><td>URL</td></tr>
                <tr><td><code>VITE_HUGPY_STUDIO_MOVIE_URL</code></td><td><code>&lt;apiBase&gt;/video/studio/movie</code></td><td>Enqueue a Studio MOVIE (Studio Movie composer) — an ordered goal timeline rendered into one NLE row (segment clips + an assembled movie.mp4 last).</td><td>URL</td></tr>
                <tr><td><code>VITE_HUGPY_STUDIO_CLIPS_URL</code></td><td><code>&lt;apiBase&gt;/video/studio/clips</code></td><td>Durable recent Studio clips for the Studio Clips list, sourced from the media catalog (not the ~600s-TTL comms jobs view), so old clips still list.</td><td>URL</td></tr>
                <tr><td><code>VITE_HUGPY_STUDIO_PRESETS_URL</code></td><td><code>&lt;apiBase&gt;/video/studio/presets</code></td><td>Curated Studio clip presets — pin a capability + geometry + VRAM budget and let the Studio router pick the model.</td><td>URL</td></tr>
                <tr><td><code>VITE_HUGPY_PROJECTS_URL</code></td><td><code>&lt;apiBase&gt;/video/projects</code></td><td>Distinct known project names (read-only, derived from the media-bus job store) for the Settings "Project" combobox.</td><td>URL</td></tr>
                <tr><td><code>VITE_HUGPY_IDENTITY_PROFILES_URL</code></td><td><code>&lt;apiBase&gt;/video/identity-profiles</code></td><td>Identity profiles (Studio stage) — the durable, named reference set that IS an identity; list/create/associate across clips, movies and stills.</td><td>URL</td></tr>
                <tr><td><code>VITE_HUGPY_PROMPT_ASSIST_URL</code></td><td><code>&lt;apiBase&gt;/video/prompt/assist</code></td><td>LLM prompt-assist for the Generate-station composer — "detail" enriches a draft, "generate" writes one from scratch.</td><td>URL</td></tr>
                <tr><td><code>VITE_HUGPY_MODELS_URL</code></td><td><code>&lt;apiBase&gt;/v1/models</code></td><td>Model registry, filtered client-side to the generation image tasks (text-to-image + image-to-image) for the Frames/Generate model dropdown.</td><td>URL</td></tr>
                <tr><td><code>VITE_HUGPY_PROMPT_TASKS_URL</code></td><td><code>&lt;apiBase&gt;/prompt/tasks</code></td><td>Per-task default-model map + task list.</td><td>URL</td></tr>
                <tr><td><code>VITE_HUGPY_LLM_JOBS_URL</code></td><td><code>&lt;apiBase&gt;/llm/jobs</code></td><td>Authoritative unified feed of every in-flight worker call across all transports (web/v1/discord/cli/media) — feeds the Active Processes panel, fleet-wide (not just this session's own jobs).</td><td>URL</td></tr>
                <tr><td colSpan="4"><span className="docs-note">The following two are <strong>server-side</strong> (env or <code>.env</code>, read by <code>hugpy serve</code> — not <code>VITE_*</code> build vars): they let a self-hosted server point the packaged video arm's canned demo at its own media without rebuilding the UI.</span></td></tr>
                <tr><td><code>HUGPY_DEMO_MEDIA_BASE</code></td><td>"https://hugpy.ai/demo-media"</td><td>Demo-media base URL injected into the served video arm's <code>index.html</code> (<code>window.__HUGPY_MEDIA_BASE__</code>).</td><td>URL or absolute path</td></tr>
                <tr><td><code>HUGPY_DEMO_MEDIA_DIR</code></td><td>unset</td><td>Local directory to serve at <code>/demo-media/</code>; when set it wins over <code>HUGPY_DEMO_MEDIA_BASE</code> and the served demo points at <code>/demo-media</code>. Unset → no route.</td><td>directory path</td></tr>
              </tbody>
            </table>
            </EnvGroup>
            </EnvReference>
          </section>

          <div className="docs-pager">
            <a className="docs-pager-card" href="#cli"><span className="d">Next</span><span className="t">CLI →</span></a>
          </div>
    </>
  )
}

/* =====================================================================
   PAGE · CLI
   ===================================================================== */
export function CliPage() {
  return (
    <>
      <div className="docs-crumbs">Reference <span>/</span> CLI</div>
      <h1>Command-line interface</h1>
      <p className="docs-lede">
        One <code>hugpy</code> entry point drives every role — central server, worker, Discord bot,
        engine build, and the keeper REPL.
      </p>
      <p className="docs-note">
        This page is the flag-by-flag reference for operators scripting deployments. For a guided
        setup, use <a className="docs-inline-link" href="#installation">Installation</a>.
      </p>

      <section id="serve" className="docs-sec">
        <h2><code>hugpy serve</code></h2>
        <p>Run the central server (API + bundled console) in one process.</p>
        <pre className="docs-block"><code>{`hugpy serve [--host 0.0.0.0] [--port 7002] [--threads 8]
            [--auth open|external] [--origins …] [--debug]`}</code></pre>
        <ul className="docs-routes">
          <li><code>--host</code> defaults to <code>0.0.0.0</code>; <code>--port</code> to <code>7002</code>.</li>
          <li><code>--auth</code> sets <code>HUGPY_AUTH_MODE</code> (default <code>open</code> here).</li>
          <li>Server selection is gunicorn → waitress → Flask dev server.</li>
        </ul>
      </section>

      <section id="worker" className="docs-sec">
        <h2><code>hugpy worker</code></h2>
        <p>Join a box to the fleet (full GPU worker or RPC shard backend).</p>
        <pre className="docs-block"><code>{`hugpy worker --central http://your-hugpy:7002
             [--role worker|rpc] [--token …] [--name …]
             [--rpc-port 50052] [--pkg-index <central>/api/llm/pip/simple]
             [--heartbeat 15]  (+ spill flags)`}</code></pre>
        <p>
          <code>--central</code> sets <code>HUGPY_BASE_URL</code>; <code>--token</code> is the enrollment
          token. The dedicated <strong>pool</strong> is set with the <code>WORKER_POOL</code> env var, not a
          flag. <code>--role rpc</code> runs only <code>rpc-server</code> for cross-machine sharding.
        </p>
      </section>

      <section id="bot" className="docs-sec">
        <h2><code>hugpy bot</code></h2>
        <p>Run the Discord arm (needs the <code>[bot]</code> extra).</p>
        <pre className="docs-block"><code>{`hugpy bot --central http://your-hugpy:7002 [--env .env] [--guild …]`}</code></pre>
        <p><code>--central</code> sets <code>HUGPY_BASE_URL</code>; <code>--env</code> loads the bot token/settings.</p>
      </section>

      <section id="install-engine" className="docs-sec">
        <h2><code>hugpy install-engine</code></h2>
        <p>Resolve / fetch / build the native llama.cpp binaries.</p>
        <pre className="docs-block"><code>{`hugpy install-engine [--cuda] [--build-from-source] [--tag …] [--jobs N] [--force]`}</code></pre>
        <p>The source build always sets <code>-DGGML_RPC=ON</code> so <code>rpc-server</code> exists for the shard fleet.</p>
      </section>

      <section id="keeper" className="docs-sec">
        <h2><code>hugpy keeper</code></h2>
        <p>
          A stdlib-only terminal REPL where an LLM “keeps” a machine / LXD guest via a shell action loop
          (Agent–Executor pattern) — independent of the central product.
        </p>
      </section>

      <div className="docs-pager">
        <a className="docs-pager-card" href="#architecture"><span className="d">Related</span><span className="t">Architecture tour →</span></a>
      </div>
    </>
  )
}
