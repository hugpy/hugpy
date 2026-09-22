// Fix it — worker troubleshooting. Moved verbatim from Docs.jsx in the
// user-first docs restructure (2026-07-05); content deliberately untouched —
// this page is the tone model for the rest of the docs (symptom-titled
// sections, concrete commands). Section ids are load-bearing: FixDoc links
// #worker-troubleshooting/{admission,unreachable,missing-files,over-budget,
// sigill,registry-error} and the console-models page links
// #worker-troubleshooting/mmproj.

import { CopyBlock } from './docParts'

export const TROUBLE_TOC = [
  ['reading-errors', 'Reading worker errors', []],
  ['admission', 'Stuck pending / blocked', []],
  ['unreachable', '“✗ unreachable” badge', []],
  ['service', 'Worker systemd unit', []],
  ['missing-files', '“✗ missing” model pill', []],
  ['over-budget', '“⚠ over budget” storage', []],
  ['no-torch', "No module named 'torch'", []],
  ['torch-circular', 'torch circular import', []],
  ['sigill', 'SIGILL crash-loop', []],
  ['mmproj', 'Vision GGUF / mmproj', []],
  ['install-engine-rpc', 'install-engine rpc-server', []],
  ['comfy-refused', 'comfy-* connection refused', []],
  ['comfy-service', 'Keep ComfyUI running', []],
  ['registry-error', '“registry error” chip', []],
]

export default function TroubleshootingPage() {
  return (
    <>
      <div className="docs-crumbs">Fix it <span>/</span> Worker Troubleshooting</div>
      <h1>Worker Troubleshooting</h1>
      <p className="docs-lede">
        Field guide for a misbehaving worker box — each entry is a symptom you see in the console
        or the worker journal, why it happens, and the fix. Box setup itself (units, engine
        flavors, ComfyUI adoption) lives in the repo&apos;s <code>WORKER-SETUP.md</code>.
      </p>

      <section id="reading-errors" className="docs-sec">
        <h2>Reading worker errors</h2>
        <p>
          Errors that originate on a worker are stamped <code>on worker &lt;name&gt; (&lt;id&gt;)</code>
          (0.1.133+) — the named box is where to look, not central. On that box:
        </p>
        <pre className="docs-block"><code>{`journalctl --user -u abstract-hugpy-worker -e   # full trace of the failure`}</code></pre>
        <p className="docs-note">
          The console surfaces only the summary line; the journal holds the stack trace, the exact
          import error, or the crash signal the entries below key off.
        </p>
      </section>

      <section id="admission" className="docs-sec">
        <h2>Worker stuck “pending”, or shows “blocked”</h2>
        <p>
          A worker never serves on join — it stays <strong>pending</strong> until an operator admits it.
          This is the admission gate, not a fault: states (pending / approved / blocked) persist across
          heartbeats.
        </p>
        <ul className="docs-routes">
          <li><strong>Admit it.</strong> In the console&apos;s <strong>Compute</strong> tab, click <strong>✓ admit</strong> on the worker&apos;s card (API: <code>POST /llm/workers/&lt;id&gt;/admit</code>). Only approved workers receive traffic.</li>
          <li><strong>It won&apos;t even register?</strong> When <code>HUGPY_WORKER_ENROLL_REQUIRED</code> is on, the box must present a one-time enrollment token (issued from the panel, passed as <code>WORKER_ENROLL_TOKEN</code>) or registration is refused.</li>
          <li><strong>Blocked means evicted.</strong> A blocked worker gets a 403 and its agent exits cleanly with no respawn — the row stays until you click <strong>unblock</strong> (or Remove). Use <strong>Block</strong> to evict, not Remove (a live agent you merely remove re-appears as pending).</li>
          <li><strong>Admitted but still not picked?</strong> Central skips a worker only when it reports <code>engine.installed == false</code> — install the engine on that box (<a className="docs-inline-link" href="#installation/engine">Build the engine</a>).</li>
        </ul>
      </section>

      <section id="unreachable" className="docs-sec">
        <h2>Worker shows “✗ unreachable”</h2>
        <p>
          Central pinged the worker&apos;s <code>/health</code> and could not connect. Work down the
          list on the worker box:
        </p>
        <ul className="docs-routes">
          <li><strong>Is the agent running?</strong> <code>systemctl --user status abstract-hugpy-worker</code>. If it is crash-looping, see the SIGILL entry below.</li>
          <li><strong>Is the port reachable from central?</strong> The agent listens on <code>9100</code> by default (<code>WORKER_PORT</code>). A firewall or NAT between central and the box breaks the relay even while the worker&apos;s own heartbeats still arrive.</li>
          <li><strong>One agent, one port.</strong> The <em>user</em> unit is canonical — a leftover system unit fighting it over 9100 crash-loops or steals the port. <code>systemctl status abstract-hugpy-worker</code> (no <code>--user</code>) should find nothing.</li>
          <li><strong>Did it register a URL central can dial?</strong> Check the worker row&apos;s URL — a loopback or wrong-interface address means central can never reach it.</li>
        </ul>
      </section>

      <section id="service" className="docs-sec">
        <h2>Worker systemd unit (canonical)</h2>
        <p>
          The canonical worker runs as a systemd <em>user</em> unit at
          <code> ~/.config/systemd/user/hugpy-worker.service</code>, driving a Python venv at
          <code> ~/hugpy-worker/venv</code>. This is what <code>hugpy worker</code>&apos;s installer
          writes; reproduce it by hand when a box needs a unit rebuilt, or to see exactly what the
          agent is launched with. Replace <code>&lt;WORKER_NAME&gt;</code> and
          <code> &lt;central-host&gt;</code>; the <code>%h</code> specifiers expand to the unit
          user&apos;s home.
        </p>
        <CopyBlock title="~/.config/systemd/user/hugpy-worker.service" code={`# ~/.config/systemd/user/hugpy-worker.service
[Unit]
Description=hugpy worker (<WORKER_NAME>)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Environment="WORKER_CENTRAL_URL=http://<central-host>:7002"
Environment="WORKER_NAME=<WORKER_NAME>"
Environment="WORKER_PORT=9100"
Environment="HUGPY_ENGINE_DIR=%h/hugpy-worker/engine"
Environment="DEFAULT_SERVE_MODE=off"
Environment="DEFAULT_ROOT=/mnt/storage/hugpy-worker/storage"
ExecStart=%h/hugpy-worker/venv/bin/python -m abstract_hugpy_dev.worker_agent --central http://<central-host>:7002 --name <WORKER_NAME> --port 9100
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target`} />

        <p>Enable it durably (survives logout and reboot):</p>
        <CopyBlock title="enable the worker unit" code={`systemctl --user daemon-reload
systemctl --user enable --now hugpy-worker
loginctl enable-linger $USER   # so it runs without an active login session`} />

        <h3 className="docs-h3-plain">Override without reinstalling (drop-in)</h3>
        <p>
          The installer <strong>hardcodes the central URL into <code>ExecStart</code></strong>, and
          re-running it overwrites your edits to the unit file. To repoint a worker (e.g. onto a LAN
          IP) durably, add a drop-in instead of editing the unit — a drop-in survives a reinstall.
          The empty <code>ExecStart=</code> line first is required: it clears the inherited command
          before the replacement (systemd otherwise rejects a second <code>ExecStart</code> for a
          <code> Type=simple</code> service).
        </p>
        <CopyBlock title="~/.config/systemd/user/hugpy-worker.service.d/central-lan.conf" code={`# ~/.config/systemd/user/hugpy-worker.service.d/central-lan.conf
[Service]
ExecStart=
ExecStart=%h/hugpy-worker/venv/bin/python -m abstract_hugpy_dev.worker_agent --central http://<CENTRAL_HOST>:7002 --name <WORKER_NAME> --port 9100
Environment=WORKER_CENTRAL_URL=http://<CENTRAL_HOST>:7002`} />
        <p className="docs-note">
          <code>--central</code> / <code>WORKER_CENTRAL_URL</code> must be reachable <strong>from the
          worker box</strong> — run <code>curl &lt;url&gt;/health</code> on the box first. A public
          hostname can fail to resolve or route from <em>inside</em> the same LAN while the LAN IP
          works; that mismatch is the usual cause of a worker that heartbeats but never gets picked.
        </p>
        <p className="docs-note">
          Point <code>DEFAULT_ROOT</code> at the <strong>largest writable volume</strong> — never a
          small root disk. The store fills as models pull, and once the volume is full the worker
          refuses loads with “won&apos;t fit / disk full” (and the assign preflight stalls re-pulls;
          see <a className="docs-inline-link" href="#worker-troubleshooting/over-budget">“⚠ over
          budget”</a>).
        </p>
        <p className="docs-note">
          Note the unit and journal names differ from this file&apos;s other examples: the installer
          also ships an <code>abstract-hugpy-worker</code> unit on some boxes — use whichever name
          <code> systemctl --user list-unit-files | grep hugpy</code> reports on the box in hand.
        </p>
      </section>

      <section id="missing-files" className="docs-sec">
        <h2>Model pill shows “✗ missing”</h2>
        <p>
          The model is assigned to the worker but its files are absent on that box. This is usually
          self-healing: the agent&apos;s reconcile loop notices the drift and re-pulls, so the pill
          should move to <code>⏳ pulling</code> on its own. If it stays missing:
        </p>
        <ul className="docs-routes">
          <li><strong>Disk:</strong> check the worker&apos;s 💾 disk chip — the assign preflight refuses models that don&apos;t fit the model-root volume, and a full disk stalls the re-pull.</li>
          <li><strong>Vision GGUFs:</strong> the pair may be blocked on the projector sidecar — see <a className="docs-inline-link" href="#worker-troubleshooting/mmproj">Vision GGUFs “files incomplete”</a>.</li>
          <li><strong>Trace it:</strong> the worker journal logs every provisioning attempt — see <a className="docs-inline-link" href="#worker-troubleshooting/reading-errors">Reading worker errors</a>.</li>
        </ul>
      </section>

      <section id="over-budget" className="docs-sec">
        <h2>Worker storage shows “⚠ over budget”</h2>
        <p>
          The worker&apos;s local model cache has grown past its disk-cache ceiling. The badge reports how
          far over, and the panel proposes a fix below it: evict the <em>coldest unprotected</em> models to
          get back under. Nothing is deleted until you approve it.
        </p>
        <ul className="docs-routes">
          <li><strong>Approve the proposal</strong> to reclaim space — it targets cold models. Loaded (🔥), heating (🔶), pulling (⏳), assigned (📎), static (🔒) and protected (🛡) models are never auto-evicted. <strong>Pinned (📌) files are eligible</strong>: a pin keeps the model&apos;s <em>allocation</em> (routing to this worker survives restarts), not its files — an evicted pinned model re-downloads on its next call while the pin stays put.</li>
          <li><strong>Or raise the ceiling</strong> from the worker&apos;s limits editor (the <em>disk cache GiB</em> field) — central limits are clamped to the box&apos;s own <code>caps.disk_cache_gib</code>, so the box&apos;s cap wins.</li>
          <li><strong>Or free disk on the box.</strong> The assign preflight refuses models that won&apos;t fit the model-root volume, so a genuinely full disk also stalls re-pulls (see <a className="docs-inline-link" href="#worker-troubleshooting/missing-files">“✗ missing”</a>).</li>
        </ul>
        <p className="docs-note">
          Every guard is re-checked on the worker at deletion time — protected / loaded / static / assigned
          files are verified again per model before anything is removed, so an approval can never take a
          model that became busy in the meantime. (Pin is not a file guard — see the pin note above.)
        </p>
      </section>

      <section id="no-torch" className="docs-sec">
        <h2>“No module named &apos;torch&apos;” — transformers models fail on a GGUF-only worker</h2>
        <p>
          The package lazy-<em>imports</em> its heavy dependencies but never auto-installs them
          (worker self-update runs <code>--no-deps</code> by design). A worker installed for GGUF
          serving registers fine, then fails at load time for every transformers-family model.
        </p>
        <p>
          Install the extras into the worker&apos;s venv over the operator-gated pip relay — the pip
          control in the console&apos;s workers panel, or directly:
        </p>
        <pre className="docs-block"><code>{`POST /llm/workers/<id>/pip       {"package": "torch"}
POST /llm/workers/<id>/pip       {"package": "transformers"}
POST /llm/workers/<id>/restart   # the agent must restart before runners pick it up`}</code></pre>
        <p>
          The full <code>[transformers]</code> extras group is torch, transformers, accelerate,
          sentencepiece, timm — install what the model family actually needs.
        </p>
        <p className="docs-note">
          Long installs outlive the HTTP relay (it times out around two minutes; the worker keeps
          pip running on its own). Poll the worker&apos;s registry <code>env</code> block for the new
          versions rather than trusting the relay reply.
        </p>
      </section>

      <section id="torch-circular" className="docs-sec">
        <h2>torch “circular import” — only inside the agent</h2>
        <p>
          Symptom: <code>partially initialized module &apos;torch&apos; … circular import</code> on model
          load, while a bare <code>venv/bin/python -c &quot;import torch&quot;</code> on the same box works.
          That split means unit-environment poisoning: an <code>Environment=LD_LIBRARY_PATH=…</code>
          line in the worker&apos;s user unit (typically added for a llama-cpp CUDA build) forces
          system CUDA libraries over the nvidia wheels torch bundles for itself.
        </p>
        <p>Confirm by importing torch with and without the unit&apos;s value:</p>
        <pre className="docs-block"><code>{`venv/bin/python -c "import torch"        # clean
LD_LIBRARY_PATH=<unit's value> \\
  venv/bin/python -c "import torch"      # reproduces the error`}</code></pre>
        <p>
          Fix: when llama-cpp is a CPU wheel the line is vestigial — remove it from the unit, then:
        </p>
        <pre className="docs-block"><code>{`systemctl --user daemon-reload
systemctl --user restart abstract-hugpy-worker`}</code></pre>
      </section>

      <section id="sigill" className="docs-sec">
        <h2>Agent core-dumps (SIGILL) at startup or first model load</h2>
        <p>
          The unit dies with <code>status=4/ILL</code>; the journal shows <code>ggml_cuda_init</code>
          and then nothing. A prebuilt <code>llama-cpp-python</code> wheel was compiled with CPU
          instructions this box doesn&apos;t have. Check:
        </p>
        <pre className="docs-block"><code>{`grep -o 'avx512[a-z]*' /proc/cpuinfo | sort -u   # empty output → no AVX512`}</code></pre>
        <p>Rebuild the wheel from source on the box (needs cmake + gcc; expect 10–20 minutes):</p>
        <pre className="docs-block"><code>{`venv/bin/pip install --force-reinstall --no-binary llama-cpp-python \\
    llama-cpp-python==<fleet version>`}</code></pre>
        <p className="docs-note">
          The eager slot-filler (0.1.133+) loads a model at startup, so a broken wheel crash-loops the unit
          immediately instead of failing at the first request. A tight start → SIGILL → restart
          loop in the journal <em>is</em> this failure — stop the unit and rebuild; don&apos;t wait
          it out.
        </p>
      </section>

      <section id="mmproj" className="docs-sec">
        <h2>Vision GGUFs: “files incomplete” / missing mmproj</h2>
        <p>
          A vision GGUF is a <strong>pair</strong>: the quant plus an <code>mmproj-*.gguf</code>
          projector sidecar in the same directory. Central&apos;s worker transfer carries the whole
          model directory, but an HF-side download may fetch only the pinned quant (a known gap) —
          the model then shows “files incomplete” and central refuses to hand it to workers.
        </p>
        <p>
          Fix: fetch the <code>mmproj-*.gguf</code> from the same HF repo into the same model
          directory as the quant.
        </p>
        <p className="docs-note">
          Serving vision GGUFs also needs the native <code>llama-server</code> binary on the worker
          (<code>hugpy install-engine</code>) — the in-process engine cannot load mmproj projectors.
        </p>
      </section>

      <section id="install-engine-rpc" className="docs-sec">
        <h2>install-engine fails at the “rpc-server” target</h2>
        <p>
          The build unconditionally targets <code>rpc-server</code>, which needs
          <code>-DGGML_RPC=ON</code> — a build path that reaches the target without the flag exits
          non-zero <em>after</em> <code>llama-server</code> is already built. If
          <code>build/bin/llama-server</code> exists and runs, the engine is usable — the resolver
          searches <code>build/bin</code> under <code>HUGPY_ENGINE_DIR</code> — and the failure exit
          is cosmetic.
        </p>
        <pre className="docs-block"><code>{`$HUGPY_ENGINE_DIR/build/bin/llama-server --version   # runs → the engine is fine`}</code></pre>
        <p className="docs-note">
          Only chase this failure if the box must lend its GPU to the cross-machine shard pool —
          that is the one consumer of <code>rpc-server</code>.
        </p>
      </section>

      <section id="comfy-refused" className="docs-sec">
        <h2>Every comfy-* model fails with “Connection refused”</h2>
        <p>
          The worker&apos;s ComfyUI process is down. The agent never runs ComfyUI itself — it only
          proxies to one on the box (default <code>http://127.0.0.1:8188</code>, override
          <code>COMFY_URL</code>). If ComfyUI is already installed on the box, just start it
          again — there is no enrollment step: once <code>/system_stats</code> answers, the
          agent adopts it and the registry row&apos;s <code>comfy.available</code> flips true on
          the next heartbeat (the 🧩 chip returns in the workers panel).
        </p>
        <pre className="docs-block"><code>{`# already installed? find and start its service:
systemctl --user list-unit-files | grep -i comfy   # then: systemctl --user start <unit>
# or launch directly from its venv (foreground; wrap in a user unit to keep it):
~/ComfyUI/venv/bin/python ~/ComfyUI/main.py --listen 127.0.0.1

# fresh install (its OWN venv, not the worker's):
git clone https://github.com/comfyanonymous/ComfyUI ~/ComfyUI
python3 -m venv ~/ComfyUI/venv && ~/ComfyUI/venv/bin/pip install -r ~/ComfyUI/requirements.txt
# user unit ExecStart=%h/ComfyUI/venv/bin/python3 %h/ComfyUI/main.py --listen 127.0.0.1

# verify adoption (on the box; then watch the workers panel):
curl -s http://127.0.0.1:8188/system_stats | head -c 120`}</code></pre>
        <p className="docs-note">
          Point ComfyUI&apos;s checkpoints dir (or a symlink) at the shared volume&apos;s
          <code>/checkpoints</code> so the Civitai download flow and worker checkpoint
          provisioning land in one place. ComfyUI has no auth — keep it bound to localhost/LAN.
        </p>
        <p>
          Starting it by hand only lasts until the next crash or reboot. To keep the 🧩 comfy chip
          lit unattended, run ComfyUI as a systemd service — see{' '}
          <a className="docs-inline-link" href="#worker-troubleshooting/comfy-service">Keep ComfyUI
          running</a>.
        </p>
      </section>

      <section id="comfy-service" className="docs-sec">
        <h2>Keep ComfyUI running (persistent service)</h2>
        <p>
          Image and video generation works only while ComfyUI is up on the worker box. ComfyUI is a
          separate peer process the agent <em>adopts</em> — it never launches or owns it — so a crash
          or a reboot silently drops that box&apos;s image capability until someone starts it again. A
          systemd <em>user</em> service closes that gap: it relaunches ComfyUI on crash and brings it
          back after a reboot, so the <strong>🧩 comfy</strong> chip stays lit unattended.
        </p>
        <p className="docs-note">
          You do <strong>not</strong> restart the worker after (re)starting ComfyUI. The agent
          re-probes <code>http://127.0.0.1:8188/system_stats</code> every ~60 seconds; once ComfyUI
          answers, the worker&apos;s <code>comfy.available</code> flips true on the next heartbeat and
          central resumes routing <code>comfy-…</code> work to the box.
        </p>

        <h3 className="docs-h3-plain">1 · Find ComfyUI&apos;s own interpreter</h3>
        <p>
          ComfyUI runs from its <strong>own</strong> virtualenv, <em>not</em> the worker&apos;s venv —
          they hold different dependencies and must stay separate. Get the absolute path of the python
          that launches ComfyUI; that path is your <code>&lt;comfy-venv-python&gt;</code>:
        </p>
        <pre className="docs-block"><code>{`ls <COMFY_DIR>/venv/bin/python        # ComfyUI's interpreter usually lives here
# or, from a shell where you already start ComfyUI by hand:
which python                          # copy this absolute path — NOT the worker's venv`}</code></pre>

        <h3 className="docs-h3-plain">2 · Write the unit</h3>
        <p>
          Create <code>~/.config/systemd/user/comfyui.service</code> — a <em>user</em> unit, the same
          kind as the worker agent, so no root is needed. Replace the placeholders with your paths:
        </p>
        <pre className="docs-block"><code>{`[Unit]
Description=ComfyUI (image/video backend adopted by the hugpy worker)
After=network-online.target
Wants=network-online.target
# Bounded restart: retry fast, but back off after repeated crashes so a broken
# install can't tight-loop forever (the crash-loop landmine).
StartLimitIntervalSec=60
StartLimitBurst=5

[Service]
Type=simple
WorkingDirectory=<COMFY_DIR>
ExecStart=<comfy-venv-python> main.py --listen 127.0.0.1 --port 8188
Restart=always
RestartSec=5
TimeoutStartSec=300   # cold start imports torch + all custom nodes; give it room

[Install]
WantedBy=default.target`}</code></pre>
        <p>
          <code>--listen 127.0.0.1</code> is the secure default: ComfyUI never appears on the network,
          and the same-box worker still reaches <code>127.0.0.1:8188</code>, so the capability is
          unaffected. Keep it this way when the gating proxy runs on the <em>same</em> box (or reaches
          ComfyUI over an SSH tunnel presented as <code>127.0.0.1:8188</code>). Only use
          <code> --listen 0.0.0.0</code> when the proxy lives on <em>another</em> host that must reach
          ae over the LAN — and then firewall <code>:8188</code> to the proxy&apos;s IP, because on
          <code>0.0.0.0</code> ComfyUI is an open, unauthenticated port to the whole LAN.
        </p>

        <h3 className="docs-h3-plain">3 · Enable it durably, then verify</h3>
        <pre className="docs-block"><code>{`loginctl enable-linger <user>                 # user services survive logout & reboot (as the worker unit does)
systemctl --user daemon-reload                # load the new unit file
systemctl --user enable --now comfyui.service # start it now and on every boot
systemctl --user status comfyui.service       # expect: active (running)
curl -s http://127.0.0.1:8188/system_stats    # JSON back = ComfyUI is up`}</code></pre>
        <p>
          No worker restart is needed: within ~60 seconds the worker&apos;s <code>comfy.available</code>
          flips true and the <strong>🧩 comfy</strong> chip returns on its card in the
          <strong> Compute</strong> tab.
        </p>

        <h3 className="docs-h3-plain">4 · Gate it like the console</h3>
        <p className="docs-note">
          ComfyUI ships <strong>no authentication</strong>. A publicly resolvable
          <code> comfy.hugpy.ai</code> with no gate is an open GPU anyone can drive. Put the
          <em> same</em> session gate in front of it that the console uses: in external auth mode the
          console validates the first-party session cookie against the auth service&apos;s
          <code> /me</code> (<code>operator_auth._validate_session_external</code>), so have nginx do the
          identical <code>auth_request</code> before it proxies to ComfyUI.
        </p>
        <pre className="docs-block"><code>{`# /etc/nginx/sites-enabled/comfy.hugpy.ai   (host front door; TLS via certbot)
server {
  server_name comfy.hugpy.ai;
  # ... listen 443 ssl + your certbot cert lines ...

  # Let the console / video UI embed it; never DENY-frame it.
  add_header Content-Security-Policy "frame-ancestors https://<your-console-origin>" always;

  # Ask the auth service "is this cookie a valid session?" — the same /me the console uses.
  location = /_authcheck {
    internal;
    proxy_pass               https://AUTH_HOST/me;   # = HUGPY_AUTH_BASE
    proxy_pass_request_body  off;
    proxy_set_header         Content-Length "";
    proxy_set_header         Cookie $http_cookie;     # forward the first-party session
  }

  location / {
    auth_request /_authcheck;                          # 200 = allow, 401/403 = bounce
    error_page 401 403 = @login;

    proxy_pass         http://COMFY_UPSTREAM;          # 127.0.0.1:8188 if nginx is on ae,
    proxy_http_version 1.1;                            # else ae's LAN IP:8188 (--listen 0.0.0.0)
    proxy_set_header   Host $host;
    proxy_set_header   Upgrade $http_upgrade;          # ComfyUI uses websockets
    proxy_set_header   Connection "upgrade";
    proxy_read_timeout 3600s;                          # long generations
  }

  location @login { return 302 https://LOGIN_HOST/?next=https://comfy.hugpy.ai$request_uri; }
}`}</code></pre>
        <p className="docs-note">
          The same-box worker still reaches <code>127.0.0.1:8188</code> directly, so gating the public
          door never costs the capability. <strong>For the embedded viewer</strong> (the video UI&apos;s
          ComfyUI station) to pass this gate from inside its iframe, the session cookie has to survive a
          cross-subdomain frame — scope it <code>Domain=.hugpy.ai</code> with
          <code> SameSite=None; Secure</code> in the auth service, or the framed request arrives
          cookieless and bounces to login. Bind ComfyUI to <code>127.0.0.1</code> when nginx runs on the
          same box; use <code>--listen 0.0.0.0</code> only when the proxy is on another host.
        </p>
        <p>
          <strong>When it&apos;s already down:</strong> a <code>comfy-…</code> request failing with
          <em> Connection refused</em> on <code>:8188</code> means ComfyUI isn&apos;t running —{' '}
          <a className="docs-inline-link" href="#worker-troubleshooting/comfy-refused">the
          comfy-refused entry</a> gets it started; this section is how you keep it from recurring.
        </p>
      </section>

      <section id="registry-error" className="docs-sec">
        <h2>Console shows “registry error”</h2>
        <p>
          A panel&apos;s periodic poll of central failed — the chip sits next to that panel&apos;s
          data source (workers registry, Discord bindings, comms sessions…) and hovering it shows
          the underlying error. It clears by itself on the next successful poll. If it persists:
        </p>
        <ul className="docs-routes">
          <li><strong>Central down or restarting</strong> — check that <code>/api/version</code> answers from the browser&apos;s origin.</li>
          <li><strong>Auth expired</strong> — in external auth mode a lapsed session turns API replies into login redirects; reload and re-login.</li>
          <li><strong>Proxy in the path</strong> — an nginx/tunnel error page instead of JSON parses as a fetch error; the hover text usually gives it away.</li>
        </ul>
        <p className="docs-note">
          This chip is about the console ↔ central hop, not the workers — a worker can be serving
          fine while a panel shows registry error, and vice versa.
        </p>
      </section>

      <div className="docs-pager">
        <a className="docs-pager-card" href="#architecture/worker-fleet"><span className="d">Related</span><span className="t">Worker fleet &amp; pools →</span></a>
      </div>
    </>
  )
}
