// Fix it — general troubleshooting. The symptom-indexed field guide for a
// self-hosted hugpy deployment: config/env, routing/API, models/memory,
// workers/publishing, Discord, concurrency, and the GPU engine build. Every
// entry is a real pitfall hit on a live deployment — symptom first (what you
// see), then why, then a copy-paste fix.
//
// Source: UI-DOCS-PITFALLS-DOSSIER.md (operator-authored). Sibling page
// TroubleshootingPage.jsx is the *worker-box* field guide; this page is the
// central/deployment one. Section ids here are LOAD-BEARING — FixDoc links
// #troubleshooting/engine-cpu-only. Keep the ids stable.

export const TROUBLE_GENERAL_TOC = [
  ['config-env', 'Config & environment', [
    ['drop-in-ignored', 'Env var ignored by the service'],
    ['env-not-reaching-api', '.env doesn’t reach the API'],
    ['auth-external-surprise', 'Auth acts external in open mode'],
  ]],
  ['routing-api', 'Routing & API', [
    ['html-body', '200 OK but the body is HTML'],
    ['edit-no-effect', 'Edit didn’t change the service'],
    ['api-prefix', '/api/… vs /… prefix'],
  ]],
  ['models-memory', 'Models & memory', [
    ['ram-oom', 'Eats RAM / OOMs at startup'],
    ['free-cache', 'free says full, nothing runs'],
    ['vm-memory-cut', 'Live memory cut won’t apply'],
  ]],
  ['workers-publishing', 'Workers & publishing', [
    ['version-ok-false', 'Workers stuck version_ok: false'],
    ['pypi-empty-name', 'Publish fails: empty package name'],
    ['pypi-lag', 'Upload OK but PyPI shows old version'],
  ]],
  ['discord', 'Discord', [
    ['discord-intents', 'Bot won’t connect with intents on'],
    ['slash-commands-slow', 'Slash commands slow to appear'],
    ['discord-delivery', 'API messages never arrive / late'],
  ]],
  ['concurrency', 'Concurrency', [
    ['console-wedges', 'Console wedges while streaming'],
  ]],
  ['gpu-engine', 'GPU / engine build', [
    ['engine-cpu-only', 'CPU-only engine on a GPU box'],
  ]],
]

export default function TroubleshootingGeneralPage() {
  return (
    <>
      <div className="docs-crumbs">Fix it <span>/</span> Troubleshooting</div>
      <h1>Troubleshooting</h1>
      <p className="docs-lede">
        A symptom index for a self-hosted hugpy deployment — each entry is what you actually see,
        why it happens, and the fix. Problems specific to a <em>worker box</em> (unreachable agents,
        broken engine wheels, ComfyUI) live in{' '}
        <a className="docs-inline-link" href="#worker-troubleshooting">Worker troubleshooting</a>.
      </p>

      {/* ============================ CONFIG / ENV ============================ */}
      <section id="config-env" className="docs-sec">
        <h2>Configuration &amp; environment</h2>
        <p className="docs-note">
          The single truth about config: <strong>the process environment is what runs</strong>, and a
          systemd drop-in beats the unit. Most “my setting is ignored” reports resolve here.
        </p>

        <h3 id="drop-in-ignored" className="docs-h3">Env var set in the unit, but the service ignores it</h3>
        <p><strong>What you see:</strong> you added <code>Environment=HUGPY_…</code> to the service unit, reloaded, restarted — and the app still behaves as if it isn&apos;t set.</p>
        <p><strong>Why:</strong> drop-ins override the unit — an <code>Environment=</code> in a later <code>.d/*.conf</code> wins — and hardening layers can mask unit env entirely. The process environment is the only truth.</p>
        <p><strong>Fix:</strong> read every layer, then verify what the running process actually has:</p>
        <pre className="docs-block"><code>{`systemctl cat <svc>          # every drop-in in order — the LAST Environment= wins
tr '\\0' '\\n' < /proc/$(systemctl show -p MainPID --value <svc>)/environ | grep HUGPY`}</code></pre>
        <p>Put the value in the layer the process actually reads, then <code>systemctl daemon-reload</code> and restart.</p>

        <h3 id="env-not-reaching-api" className="docs-h3">Your <code>.env</code> values don&apos;t reach the API</h3>
        <p><strong>What you see:</strong> settings in the tree&apos;s <code>.env</code> take effect for the Discord bot but never for the central API.</p>
        <p><strong>Why:</strong> the Flask central API does <strong>not</strong> read the tree&apos;s <code>.env</code> — its environment comes from the unit + drop-ins only. The Discord bot <em>does</em> read <code>.env</code> (and falls back to a legacy token file for <code>DISCORD_TOKEN</code>).</p>
        <p><strong>Fix:</strong> API config → a systemd drop-in; bot config → <code>.env</code>. Verify with the reality-check above (<a className="docs-inline-link" href="#troubleshooting/drop-in-ignored">Env var ignored</a>).</p>

        <h3 id="auth-external-surprise" className="docs-h3">Auth behaves like <code>external</code> even though you set <code>open</code></h3>
        <p><strong>What you see:</strong> <code>HUGPY_AUTH_MODE=open</code> is set, but the console still bounces to a login and the API 401s.</p>
        <p><strong>Why:</strong> same root cause as above — a hardening drop-in pins <code>external</code>. Note the gate <strong>fails closed</strong>: if the upstream auth service is unreachable, 401s during that outage are by design.</p>
        <p><strong>Fix:</strong> find and reconcile the drop-in that sets the mode. For CLI/automation against an external-mode deployment, set <code>HUGPY_OPERATOR_TOKEN</code> in the service env and send it as a header:</p>
        <pre className="docs-block"><code>{`curl -H "X-Operator-Token: $HUGPY_OPERATOR_TOKEN" http://localhost:7002/api/llm/workers
# (Authorization: Bearer <token> works too). Never flip a public deployment to open.`}</code></pre>
      </section>

      {/* ============================ ROUTING / API ============================ */}
      <section id="routing-api" className="docs-sec">
        <h2>Routing &amp; API debugging</h2>

        <h3 id="html-body" className="docs-h3">Your API call returns 200 — but the body is HTML</h3>
        <p><strong>What you see:</strong> a request “succeeds” with 200, but parsing the body dies with <code>Expecting value: line 1 column 1 (char 0)</code>.</p>
        <p><strong>Why:</strong> the SPA catch-all serves <code>index.html</code> for any unknown route, so a typo&apos;d or unregistered route looks like success. That JSON error almost always means “HTML where JSON was expected”, not a JSON bug.</p>
        <p><strong>Fix:</strong> treat an HTML body as a routing miss. Confirm the route exists (did the service restart after you added it? right source tree — see <a className="docs-inline-link" href="#troubleshooting/edit-no-effect">the next entry</a>?), and sanity-check against a known-good JSON route first:</p>
        <pre className="docs-block"><code>{`curl -s http://localhost:7002/api/version   # JSON back = the API is routing fine`}</code></pre>

        <h3 id="edit-no-effect" className="docs-h3">You edited the code but the running service didn&apos;t change</h3>
        <p><strong>What you see:</strong> a code change has no effect no matter how many times you restart.</p>
        <p><strong>Why:</strong> deployments often have multiple checkouts (working repo, release mirror, the tree the service actually runs). Editing the wrong one silently does nothing.</p>
        <p><strong>Fix:</strong> find the tree the unit runs, edit <em>that</em> one, restart:</p>
        <pre className="docs-block"><code>{`systemctl cat <svc> | grep -Ei 'chdir|pythonpath|WorkingDirectory'
# when in doubt, md5sum the file in both trees to prove which one is live`}</code></pre>

        <h3 id="api-prefix" className="docs-h3">Do I call <code>/api/discord/…</code> or <code>/discord/…</code>?</h3>
        <p><strong>What you see:</strong> two working prefixes and uncertainty about which is canonical.</p>
        <p><strong>Why:</strong> both are mounted — the <code>/api</code> prefix is stripped by nginx in production and by middleware on bare gunicorn.</p>
        <p><strong>Fix:</strong> pick one and use it consistently. Remember the bare gunicorn port bypasses nginx: no TLS and no vhost-level protections there.</p>
      </section>

      {/* ============================ MODELS / MEMORY ============================ */}
      <section id="models-memory" className="docs-sec">
        <h2>Models &amp; memory</h2>

        <h3 id="ram-oom" className="docs-h3">The server eats way more RAM than expected / OOMs at startup</h3>
        <p><strong>What you see:</strong> memory use spikes at boot, sometimes an OOM before a single request.</p>
        <p><strong>Why:</strong> local llama-server slots preload <code>DEFAULT_CHAT_MODEL</code> at boot — <code>SLOT_COUNT</code> of them — and each loaded GGUF costs roughly its file size.</p>
        <p><strong>Fix:</strong> size RAM ≈ base services (~2–4 GiB) + the largest model(s) you&apos;ll hold concurrently, or lower <code>SLOT_COUNT</code> (<code>SLOT_COUNT=0</code> disables the preloading pool). Offloading inference to workers? Unload local slots first and confirm none remain before shrinking the box:</p>
        <pre className="docs-block"><code>{`pgrep -af llama-server   # expect nothing before you cut the box's RAM`}</code></pre>

        <h3 id="free-cache" className="docs-h3"><code>free</code> says memory is nearly full but nothing is running</h3>
        <p><strong>What you see:</strong> <code>total − free</code> looks alarming even with no models loaded.</p>
        <p><strong>Why:</strong> mmap&apos;d GGUF files show up as buff/cache — reclaimable, not leaked.</p>
        <p><strong>Fix:</strong> read the <code>used</code> and <code>available</code> columns, not <code>total − free</code>; judge the true working set by process RSS:</p>
        <pre className="docs-block"><code>{`free -h                              # look at used + available
ps -eo rss,comm --sort=-rss | head   # real working set, biggest first`}</code></pre>

        <h3 id="vm-memory-cut" className="docs-h3">A VM/host memory cut refused to apply live</h3>
        <p><strong>What you see:</strong> a large downward memory change on a running VM times out or won&apos;t take.</p>
        <p><strong>Why:</strong> big downward changes on a running guest can time out (ballooning).</p>
        <p><strong>Fix:</strong> stop → set the limit → start. Plan a short maintenance blink for it rather than changing it live.</p>
      </section>

      {/* ============================ WORKERS / PUBLISHING ============================ */}
      <section id="workers-publishing" className="docs-sec">
        <h2>Workers &amp; publishing</h2>
        <p className="docs-note">
          For a worker that&apos;s unreachable, missing files, or crash-looping on a bad engine wheel, see
          the box-level <a className="docs-inline-link" href="#worker-troubleshooting">Worker troubleshooting</a>{' '}
          guide. The entries here are about <em>version convergence</em> and <em>publishing</em>.
        </p>

        <h3 id="version-ok-false" className="docs-h3">Workers show <code>version_ok: false</code> and never converge</h3>
        <p><strong>What you see:</strong> workers sit at the wrong package version and never catch up.</p>
        <p><strong>Why:</strong> workers self-update toward <code>HUGPY_REQUIRED_PKG_VERSION</code> from the pip index central serves. Skew means the target was never bumped/published, or the index isn&apos;t reachable from the worker.</p>
        <p><strong>Fix:</strong> compare the reported versions, then publish/bump and confirm reachability:</p>
        <pre className="docs-block"><code>{`curl -s http://localhost:7002/api/llm/workers   # pkg_version vs required_pkg_version per worker
# then: publish/bump the target, confirm the worker can reach the index, wait one heartbeat`}</code></pre>

        <h3 id="pypi-empty-name" className="docs-h3">PyPI publish fails instantly with an empty package name</h3>
        <p><strong>What you see:</strong> the upload dies immediately with an empty/blank package name and a cascade of confusing downstream errors.</p>
        <p><strong>Why:</strong> name resolution executes <code>setup.py</code> — <em>any</em> parse/exec error (a <code>version=&apos;None&apos;</code>, an unterminated string in <code>install_requires</code>) yields an empty name.</p>
        <p><strong>Fix:</strong> validate setup before publishing, and prefer a declarative name:</p>
        <pre className="docs-block"><code>{`python3 -m py_compile setup.py   # any error here is the real failure
python3 setup.py --name          # must print the real name, not blank
# better: declare [project].name in pyproject.toml (read without executing setup.py);
# keep version literals PEP 440-valid`}</code></pre>

        <h3 id="pypi-lag" className="docs-h3">Upload said OK but PyPI still shows the old version</h3>
        <p><strong>What you see:</strong> twine reports success but the JSON API still lists the previous version.</p>
        <p><strong>Why:</strong> PyPI&apos;s JSON API lags the upload by ~30–60 s.</p>
        <p><strong>Fix:</strong> re-poll before concluding failure, and make republish idempotent so a retry can&apos;t 400:</p>
        <pre className="docs-block"><code>{`twine upload --skip-existing dist/*   # retry-safe
# a GitHub push failing AFTER the upload is non-fatal — the release is already public`}</code></pre>
      </section>

      {/* ============================ DISCORD ============================ */}
      <section id="discord" className="docs-sec">
        <h2>Discord arm</h2>

        <h3 id="discord-intents" className="docs-h3">The bot won&apos;t connect when you enable the members intent</h3>
        <p><strong>What you see:</strong> enabling a privileged intent makes the bot fail to connect.</p>
        <p><strong>Why:</strong> privileged intents need <strong>both</strong> the env flag <em>and</em> the Developer-Portal toggle (Bot → Privileged Gateway Intents). One without the other fails to connect.</p>
        <p><strong>Fix:</strong> flip both — the code flag and the portal toggle — then restart the bot.</p>

        <h3 id="slash-commands-slow" className="docs-h3">Slash commands take forever to appear</h3>
        <p><strong>What you see:</strong> newly added slash commands don&apos;t show up for a long time.</p>
        <p><strong>Why:</strong> with <code>GUILD_ID</code> unset, commands sync <em>globally</em> — slow, up to an hour.</p>
        <p><strong>Fix:</strong> set <code>GUILD_ID</code> for instant guild-local sync while developing.</p>

        <h3 id="discord-delivery" className="docs-h3">Messages sent through the API never arrive / arrive late</h3>
        <p><strong>What you see:</strong> a queued outbound message shows up seconds later, or a long one fails.</p>
        <p><strong>Why:</strong> the bot drains the outbox on a ~8 s poll — delivery is <em>eventual</em>, not instant. And Discord hard-caps messages at 2000 chars; the session endpoints reject anything over 1900 up front with a 413 for exactly this reason.</p>
        <p><strong>Fix:</strong> wait ~10 s before diagnosing a “missing” message; split content to stay under the char cap.</p>
      </section>

      {/* ============================ CONCURRENCY ============================ */}
      <section id="concurrency" className="docs-sec">
        <h2>Concurrency</h2>

        <h3 id="console-wedges" className="docs-h3">The console UI wedges when a chat is streaming</h3>
        <p><strong>What you see:</strong> everything in the console stalls while one chat is streaming tokens.</p>
        <p><strong>Why:</strong> a single GIL-bound gunicorn worker can&apos;t serve console concurrency — threads pile up in a write and the whole process stalls.</p>
        <p><strong>Fix:</strong> run central under gunicorn with more than one worker process. This is safe here because the stores are file-locked / multi-process by design:</p>
        <pre className="docs-block"><code>{`gunicorn -w 4 ...   # multiple worker processes; safe because state is file-locked`}</code></pre>
        <p className="docs-note">Don&apos;t add worker processes to a deployment whose state <em>isn&apos;t</em> multi-process-safe.</p>
      </section>

      {/* ============================ GPU / ENGINE ============================ */}
      <section id="gpu-engine" className="docs-sec">
        <h2>GPU / engine build</h2>

        <h3 id="engine-cpu-only" className="docs-h3">⚠ CPU-only engine on a box that has a GPU — inference is slow, VRAM sits idle</h3>
        <p><strong>What you see:</strong> the workers panel shows <code>⚠ CPU-only engine</code> (or a worker reports <code>supports_gpu_offload: false</code>); models run, but slowly, and the GPU stays empty.</p>
        <p><strong>Why:</strong> <code>llama-cpp-python</code>&apos;s default install is a <strong>CPU-only build</strong>. <code>n_gpu_layers</code> is then silently ignored — nothing errors, it&apos;s just slow.</p>
        <p><strong>Diagnostic</strong> (run in the env the worker <em>service</em> uses):</p>
        <pre className="docs-block"><code>{`python -c "from llama_cpp import llama_supports_gpu_offload as f; print(f())"   # False = CPU-only`}</code></pre>

        <p className="docs-eyebrow">The two traps first</p>
        <ul className="docs-routes">
          <li><strong>Source build without the CUDA toolkit fails.</strong> <code>CMAKE_ARGS=&quot;-DGGML_CUDA=on&quot; pip install …</code> dies with <code>Could not find nvcc … CUDA Toolkit not found</code> unless you actually have <code>nvcc</code>.</li>
          <li><strong>The prebuilt cu124 wheel installs but won&apos;t import.</strong> <code>libcudart.so.12: cannot open shared object file</code> — the wheel needs the CUDA <em>runtime</em> libraries, which it does not bundle. The driver alone (nvidia-smi working) is not enough.</li>
        </ul>

        <p className="docs-eyebrow">The working recipe (no system CUDA toolkit — everything inside the env)</p>
        <pre className="docs-block"><code>{`# 1) runtime libs (skip if torch is installed — it bundles them under site-packages/nvidia/)
pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12

# 2) the prebuilt CUDA wheel — --no-deps stops it dragging numpy past your pins
pip install llama-cpp-python --force-reinstall --no-cache-dir --no-deps \\
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124`}</code></pre>
        <p><strong>3) Put the cu12 lib dirs on the loader path.</strong> For a systemd worker, a drop-in:</p>
        <pre className="docs-block"><code>{`Environment=LD_LIBRARY_PATH=<env>/site-packages/nvidia/cuda_runtime/lib:<env>/site-packages/nvidia/cublas/lib`}</code></pre>
        <p>For a conda env, symlinking <code>libcudart.so.12</code>, <code>libcublas.so.12</code>, <code>libcublasLt.so.12</code> into <code>&lt;conda&gt;/lib</code> works with no env var at all.</p>
        <p><strong>4) Restart the worker and verify:</strong> <code>GET /llm/workers</code> shows <code>supports_gpu_offload: true</code> and model loads log <code>offloaded N/N layers to GPU</code>.</p>

        <p className="docs-eyebrow">Trap three — <code>Illegal instruction</code> (SIGILL, exit 132) right after the CUDA device list</p>
        <p>
          This looks like a GPU problem; it is a CPU one. The prebuilt cu124 wheels bundle a single
          <code> libggml-cpu.so</code> compiled <em>with</em> AVX512 and no runtime CPU dispatch, so they
          SIGILL on CPUs without AVX512 — notably Intel 12th–14th gen consumer parts. Watch for a worker
          crash-looping with <code>status=4/ILL</code> in its journal. This is covered in depth (with the
          rebuild) in the worker guide:{' '}
          <a className="docs-inline-link" href="#worker-troubleshooting/sigill">Agent core-dumps (SIGILL)</a>.
          Source-build locally with the CUDA toolkit installed (<code>-march=native</code> gets the right ISA):
        </p>
        <pre className="docs-block"><code>{`CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12" \\
  CMAKE_BUILD_PARALLEL_LEVEL=18 \\
  pip install llama-cpp-python==<ver> --force-reinstall --no-deps --no-cache-dir`}</code></pre>
        <p className="docs-note">
          Pin <code>-DCMAKE_CUDA_ARCHITECTURES</code> to your card (89 = Ada; see NVIDIA&apos;s compute-capability
          list) or it compiles kernels for everything. A native build links the system CUDA libs — no
          <code> LD_LIBRARY_PATH</code> needed (that is only for the prebuilt-wheel + pip-runtime-libs route).
          The pip <code>nvidia-cuda-nvcc-cu12</code> package is NOT a substitute for the toolkit — it ships
          ptxas/nvvm but no <code>nvcc</code> binary; on Ubuntu 24.04, <code>sudo apt install nvidia-cuda-toolkit</code>{' '}
          pairs gcc-12 automatically.
        </p>

        <p className="docs-eyebrow">AMD, and boxes that are correctly CPU-only</p>
        <p>
          AMD: source-build with <code>CMAKE_ARGS=&quot;-DGGML_HIP=on&quot;</code> (ROCm — recent cards only) or
          <code> -DGGML_VULKAN=on</code> (the realistic path for older Polaris RX 470–590 cards ROCm dropped);
          there are no prebuilt wheels there. A box with <strong>no discrete-GPU driver loaded</strong>
          (<code>nvidia-smi</code> fails, <code>/proc/driver/nvidia/version</code> absent) is correctly CPU-only —
          the badge is truth there, not a defect.
        </p>

        <p className="docs-eyebrow">Fleet tip &amp; fallout notes (both bit us)</p>
        <ul className="docs-routes">
          <li><strong>Don&apos;t pull ~1.8 GB per box.</strong> CDN speed varies wildly per connection. Download the wheels once on the fastest box (<code>pip download -d wheelhouse …</code>), then <code>scp</code> them over the LAN and <code>pip install /path/to/*.whl</code> on each worker. A stuck download is a bad CDN session — kill and retry.</li>
          <li><strong>Fix the env the service actually uses</strong> (<code>ps -eo args | grep worker</code> / <code>systemctl cat</code>), not whatever shell you happen to be in.</li>
          <li><strong>Re-pin numpy after.</strong> Without <code>--no-deps</code> the wheel bumps numpy (2.5.x) past common pins (hugpy wants <code>&lt;2.4</code>, numba <code>&lt;2.5</code> and numba hard-fails at import). Re-pin: <code>pip install &apos;numpy==2.3.5&apos;</code>.</li>
          <li><strong>pip&apos;s trailing “dependency resolver” ERROR block is not an install failure</strong> — if the last line says <code>Successfully installed</code>, it installed. Conflicts for packages you don&apos;t import here are noise; version-check enforcers (numba) are the real ones.</li>
        </ul>
      </section>

      <div className="docs-pager">
        <a className="docs-pager-card" href="#worker-troubleshooting"><span className="d">Next</span><span className="t">Worker troubleshooting →</span></a>
      </div>
    </>
  )
}
