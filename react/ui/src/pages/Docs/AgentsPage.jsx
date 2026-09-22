// Operator guide for the portable hugpy-agent runtime. This page documents
// the stable happy path first; short symptom-based fixes live at the end.
//
// Section ids are addressable as #agents/<id>; keep them stable if other pages
// start linking here.
import { CopyBlock } from './docParts'

export const AGENTS_TOC = [
  ['concepts', 'What the agent system is', []],
  ['prereqs', 'Before you start', []],
  // Linked from the /fleet screen's per-block "Docs ↗" buttons
  // (react/agents_ui/src/FleetOverview.tsx → DOCS_HREFS). Keep both ids.
  ['install-link', 'Your install link', []],
  ['fleet-console', 'The fleet-console package', []],
  ['quickstart', 'Quickstart: your first run', []],
  ['agent-commands', 'CLI command reference', []],
  ['agent-tools', 'Agent tool reference', [
    ['tool-policy', 'Tools and policy'],
    ['tool-files', 'Files, shell, and network'],
    ['tool-ml', 'Fleet ML and generation'],
    ['tool-meta', 'Memory, operator, and completion'],
  ]],
  ['enroll', 'Enroll a box as a service', []],
  ['escalation', 'Approvals: the ask_operator flow', []],
  ['node-mode', 'Node mode: join the fleet', []],
  ['evals', 'Pick a brain: the eval harness', []],
  ['agent-trouble', 'Troubleshooting', []],
]

export default function AgentsPage() {
  return (
    <>
      <div className="docs-crumbs">Agents <span>/</span> Working the agent system</div>
      <h1>Agents</h1>
      <p className="docs-lede">
        <code>hugpy-agent</code> is a portable agent runtime that uses your hugpy fleet as its
        inference brain: an assess→act→observe loop with a workspace-jailed toolset, memory, a
        crash-safe journal, and a policy gate. Install it on any machine that can reach your fleet;
        the agent box does not need a GPU or a local model engine.
      </p>

      <div className="docs-callout">
        <div className="docs-callout-t">The shortest successful path</div>
        <p>
          Install the package in its own venv, export the fleet URL, API key, model ID, and workspace,
          then run one task with <code>--policy auto</code>. Once that succeeds, choose whether the
          long-running service should consume a local queue, a Discord inbox, or central node work.
          Discord and systemd are optional extensions—not prerequisites for a first run.
        </p>
      </div>

      {/* ============================ CONCEPTS ============================ */}
      <section id="concepts" className="docs-sec">
        <h2>What the agent system is</h2>
        <p>
          The pieces below are the whole mental model. Nothing here runs a model locally — every
          thought is a call to your fleet.
        </p>
        <ul className="docs-routes">
          <li><strong>An agent</strong> is one <em>run</em>: a task, an assess→act→observe loop that
            calls tools until it emits <code>final_answer</code>, and a structured report
            (<code>{'{outcome, steps, tool_calls, est_tokens, answer}'}</code>). Every message and tool
            call is journaled to a per-workspace SQLite ledger, so an interrupted run can be resumed.</li>
          <li><strong>An agent node</strong> is that same runtime left running as the <code>serve</code>
            daemon (the P2.7 unit): it polls a task source, and runs each inbound task as a normal
            journaled run — policy, audit, and escalation all apply exactly as in an ad-hoc run.</li>
          <li><strong>Brains ride the fleet.</strong> The agent has no model of its own. All inference
            goes to your hugpy fleet over the OpenAI-compatible <code>/v1</code> API; the default brain
            is <code>Qwen~Qwen3-Coder-Next-GGUF</code> — see
            the think-suffix note in <a className="docs-inline-link" href="#agents/quickstart">Quickstart</a>).
            Pick another with <code>--model</code> / <code>HUGPY_MODEL</code>.</li>
          <li><strong>Tools</strong> come in four groups: <em>local</em> (<code>fs_read</code>/
            <code>fs_write</code>/<code>fs_glob</code>, <code>shell</code>, <code>http_fetch</code> — all
            workspace-jailed and symlink-safe; <code>shell</code> is risk-classed, fetch is GET-only with
            a 64&nbsp;KB cap), <em>fleet text ML</em> (<code>summarize</code>, <code>keywords</code>,
            <code>embed</code>, <code>similarity</code>), <em>fleet file ML</em> (<code>transcribe</code>,
            <code>classify</code>, <code>detect</code>, <code>segment</code>, <code>depth</code>,
            <code>vision</code>), <em>fleet generation</em> (<code>generate_image</code>,
            <code>generate_scene</code> — async, capped per run), and <em>meta</em>
            (<code>models_list</code>, <code>remember</code>, <code>final_answer</code>). A fleet capacity
            gap or job failure comes back to the model verbatim as structured data — never a masked
            exception.</li>
          <li><strong>Policy modes</strong> gate every tool call:</li>
        </ul>
        <table className="docs-table">
          <thead><tr><th>Mode</th><th>What it allows</th></tr></thead>
          <tbody>
            <tr><td><code>readonly</code></td><td>Read-only tools run; anything that writes, shells, fetches, or spends GPU is denied as data. The model is told honestly and reports it — the run still completes.</td></tr>
            <tr><td><code>ask</code> <span className="docs-note">(default)</span></td><td>Non-readonly tools pause for an operator decision <em>via the escalation channel</em>. <strong>With no channel configured this fails closed to deny</strong> — identical to <code>readonly</code> in practice. This is the trap in the callout above.</td></tr>
            <tr><td><code>auto</code></td><td>All tools run without asking. The right mode for a trusted ad-hoc run on a box you own; audit still records everything.</td></tr>
          </tbody>
        </table>
        <p>
          <strong>Audit trail.</strong> Every tool call writes one JSONL audit line with the decision
          (<code>allow</code>/<code>deny</code>), the timing, and the tool <em>arguments hashed as
          sha256</em> — a secret passed in a tool arg appears only as its hash, never in the clear.
        </p>
      </section>

      {/* ============================ PREREQS ============================ */}
      <section id="prereqs" className="docs-sec">
        <h2>Before you start</h2>
        <p>A first run needs only an API key and one model that the fleet can serve.</p>

        <h3 className="docs-h3-plain">1 · An API key</h3>
        <p>
          Mint a key in the console under <strong>API access</strong> (keys look like <code>hp_…</code>,
          shown once, stored hashed), then export it as <code>HUGPY_API_KEY</code>. Environment variables
          or the service's mode-0600 env file are the normal places for credentials.
        </p>

        <h3 className="docs-h3-plain">2 · A ready model ID</h3>
        <p>
          Run <code>hugpy-agent models</code> and copy the exact registry ID. The examples use
          <code> Qwen~Qwen3-Coder-Next-GGUF</code>. Model loading, worker admission, and GPU placement
          belong to the Hugpy fleet; the agent machine only calls the API.
        </p>

        <h3 className="docs-h3-plain">Optional · Discord</h3>
        <p>
          Discord is needed only for <code>ask</code>-mode approvals or the
          <code> discord-inbox</code> daemon source. Mint a scoped comms session in the console and set
          its full <code>…/api/discord/session/&lt;token&gt;</code> URL as
          <code> HUGPY_DISCORD_SESSION</code>. A channel ID by itself is not a session token.
        </p>
      </section>

      {/* =========================== INSTALL LINK =========================== */}
      <section id="install-link" className="docs-sec">
        <h2>Your install link</h2>
        <p>
          A one-time install link is the short way to do everything on this page: it carries a freshly
          minted API key scoped to your account, so the box you are enrolling never needs you to paste a
          key by hand. Sign in and open <a className="docs-inline-link" href="/fleet">the Fleet screen</a>
          {' '}— <strong>Your agent install link</strong> — give it a label, and press
          <strong> Generate my agent install link</strong>.
        </p>
        <p>
          The response is shown <strong>once</strong>. It contains the link URL and a ready-to-paste
          command per platform; run the one for the box you are enrolling.
        </p>
        <CopyBlock title="linux / macOS" code={'curl -fsSL https://your-hugpy/api/agent/install/<link_id>.sh | bash'} />
        <CopyBlock title="windows (PowerShell)" code={'irm https://your-hugpy/api/agent/install/<link_id>.ps1 | iex'} />
        <p className="docs-note">
          macOS testers who would rather double-click than open a terminal can use the
          <code> .zip</code> (and, where the deployment can build one, the <code>.pkg</code>) listed under
          <strong> downloads</strong> in the same response.
        </p>

        <h3 className="docs-h3-plain">What the link actually is</h3>
        <ul className="docs-routes">
          <li><strong>One-time by default.</strong> Only the payload fetch spends a use — the
            <code> .sh</code>/<code>.ps1</code>/<code>.zip</code>/<code>.pkg</code> wrappers are free, so the
            canonical one-liner costs exactly one use. It also expires (24h by default).</li>
          <li><strong>Scoped, not privileged.</strong> A member's link mints a key limited to the product
            scopes (<code>v1</code>, <code>ml</code>). It can never be <code>full</code> or the
            fleet-enrolling <code>agent-register</code>, and an API key of any scope is never an operator
            credential.</li>
          <li><strong>Yours.</strong> The creating username is recorded on the link. The list on the Fleet
            screen shows only your own links, and only you (or an operator) can revoke one — revoking the
            link revokes the key it minted.</li>
          <li><strong>The raw key is never shown to you.</strong> It exists only inside the templated
            download, which is why the link is one-time.</li>
        </ul>
        <p className="docs-note">
          Prefer to do it by hand? Everything below still works — mint an ordinary key under
          <strong> API access</strong> and follow the quickstart.
        </p>
      </section>

      {/* =========================== FLEET CONSOLE ========================== */}
      <section id="fleet-console" className="docs-sec">
        <h2>The fleet-console package</h2>
        <p>
          <code>fleet-console</code> is the desktop console for your fleet, shipped as a Debian package.
          Signed-in members download it from <a className="docs-inline-link" href="/fleet">the Fleet
          screen</a> under <strong>Download fleet-console</strong>; it is not a public download.
        </p>
        <CopyBlock title="install the .deb" code={`# download from /fleet, then:
sha256sum ~/Downloads/fleet-console_*.deb     # compare with the SHA-256 shown on /fleet
sudo apt install ./fleet-console_*.deb`} />
        <p>
          Always compare the SHA-256 printed next to the download with the one you compute locally before
          installing. The page shows the exact digest for the artifact it is serving.
        </p>

        <h3 className="docs-h3-plain">The paired agent wheel</h3>
        <p>
          Where the deployment stages one, the same block offers a <strong>paired agent wheel</strong> —
          the matching <code>hugpy_agent</code> build for that console version. Install it into the agent
          venv when you want the two pinned together:
        </p>
        <CopyBlock title="install the paired wheel" code={'"$HOME/hugpy-agent/venv/bin/pip" install --upgrade ./hugpy_agent-*.whl'} />
        <p className="docs-note">
          <strong>Nothing staged?</strong> The block says so plainly. The artifacts are built and staged per
          deployment — a fleet with no package staged simply has no console download yet, and the agent
          runtime below works without it.
        </p>
      </section>

      {/* ============================ QUICKSTART ============================ */}
      <section id="quickstart" className="docs-sec">
        <h2>Quickstart: your first run</h2>
        <p>
          From the package directory, create a dedicated venv, install the runtime, and export four
          values. Keeping the venv at <code>~/hugpy-agent/venv</code> also matches the service installer.
        </p>
        <CopyBlock title="install and configure" code={`cd path/to/hugpy_agent
python3 -m venv "$HOME/hugpy-agent/venv"
"$HOME/hugpy-agent/venv/bin/pip" install --upgrade .

export HUGPY_BASE=https://your-hugpy/api
export HUGPY_API_KEY=hp_REPLACE_ME
export HUGPY_MODEL=Qwen~Qwen3-Coder-Next-GGUF
export HUGPY_WORKSPACE="$HOME/hugpy-agent/workspace"

mkdir -p "$HUGPY_WORKSPACE"
"$HOME/hugpy-agent/venv/bin/hugpy-agent" models`} />
        <p>
          Use the exact model ID printed by <code>models</code>. Then prove the full path with one small
          write task:
        </p>
        <CopyBlock title="first successful run" code={`"$HOME/hugpy-agent/venv/bin/hugpy-agent" run \\
  --policy auto \\
  --max-steps 6 \\
  "Use fs_write to create hello.txt containing exactly: Hello from hugpy-agent"

cat "$HUGPY_WORKSPACE/hello.txt"`} />
        <p className="docs-note">
          <code>auto</code> is intentional for this trusted smoke test. Use <code>readonly</code> for
          inspection, or <code>ask</code> after an operator session is configured.
        </p>
        <p>
          Other everyday commands: <code>hugpy-agent chat</code> (interactive REPL, same tools),
          <code> hugpy-agent runs</code> (journaled runs in this workspace), and
          <code> hugpy-agent resume &lt;run_id&gt;</code> (continue an interrupted run — completed side
          effects are replayed, not re-executed).
        </p>

        <h3 className="docs-h3-plain">Memory: remember &amp; recall</h3>
        <p>
          The agent can persist a fact across runs by calling the <code>remember</code> tool; facts land
          as markdown under the workspace’s <code>memory/</code> with a <code>MEMORY.md</code> index, and
          are indexed for semantic recall so a later run retrieves the relevant ones on its own. This is
          per-workspace — point runs at the same <code>--workspace</code> to share memory.
        </p>

      </section>

      {/* ============================ CLI COMMANDS ============================ */}
      <section id="agent-commands" className="docs-sec">
        <h2>CLI command reference</h2>
        <p>
          The installed <code>hugpy-agent</code> CLI has seven commands. Configuration can come from
          flags, environment variables, the workspace <code>.env</code>, or
          <code> agent.toml</code>; explicit CLI flags win. Run <code>hugpy-agent &lt;command&gt;
          --help</code> for the exact options supported by the installed version.
        </p>
        <table className="docs-table">
          <thead><tr><th>Command</th><th>Purpose</th><th>Typical invocation</th></tr></thead>
          <tbody>
            <tr><td><code>run</code></td><td>Run one task to completion.</td><td><code>hugpy-agent run --policy auto "Create hello.txt"</code></td></tr>
            <tr><td><code>chat</code></td><td>Start an interactive REPL with the same tools and policy gate.</td><td><code>hugpy-agent chat --policy readonly</code></td></tr>
            <tr><td><code>resume</code></td><td>Continue an interrupted journaled run without replaying completed effects.</td><td><code>hugpy-agent resume &lt;run_id&gt; --policy auto</code></td></tr>
            <tr><td><code>models</code></td><td>List models available through the configured fleet.</td><td><code>hugpy-agent models</code></td></tr>
            <tr><td><code>runs</code></td><td>List runs recorded in the current workspace journal.</td><td><code>hugpy-agent runs</code></td></tr>
            <tr><td><code>eval</code></td><td>Score one or more brains against the built-in task suite.</td><td><code>hugpy-agent eval --model Qwen~Qwen3-Coder-Next-GGUF --policy auto</code></td></tr>
            <tr><td><code>serve</code></td><td>Poll a queue, Discord inbox, or central node source as a long-running daemon.</td><td><code>hugpy-agent serve --task-source queue --policy auto</code></td></tr>
          </tbody>
        </table>
        <CopyBlock title="common configuration" code={`export HUGPY_BASE=https://your-hugpy/api
export HUGPY_API_KEY=hp_REPLACE_ME
export HUGPY_WORKSPACE="$HOME/hugpy-agent/workspace"
export HUGPY_MODEL=Qwen~Qwen3-Coder-Next-GGUF

mkdir -p "$HUGPY_WORKSPACE"
hugpy-agent models`} />
        <p>
          Common flags are <code>--base</code>, <code>--model</code>,
          <code> --workspace</code>, <code>--max-steps</code>,
          <code> --tools-mode</code> (<code>prompted</code>, <code>constrained</code>,
          <code> native</code>, or <code>auto</code>), <code>--policy</code>,
          <code> --audit-verbose</code>, <code>--think</code>, and <code>--quiet</code>.
          The default prompted tool mode is the dependable choice for seams that cannot round-trip a
          native <code>role="tool"</code> message.
        </p>
        <CopyBlock title="local queue daemon" code={`hugpy-agent serve --task-source queue --policy auto --poll-interval 5

# Dispatch from another terminal:
printf '%s\\n' 'Create queue-test.txt containing: queue works' \\
  >> "$HUGPY_WORKSPACE/.hugpy_agent/tasks.queue"`} />
      </section>

      {/* ============================ TOOL REFERENCE ============================ */}
      <section id="agent-tools" className="docs-sec">
        <h2>Agent tool reference</h2>
        <p>
          Tools are functions the model selects during <code>run</code>, <code>chat</code>, or
          <code> serve</code>; they are not separate shell subcommands. Name a tool in the task when
          deterministic selection matters—for example, “use <code>fs_write</code>, not
          <code> shell</code>.” The table below reflects the current 21-tool registry.
        </p>

        <h3 id="tool-policy" className="docs-h3">Tools and policy</h3>
        <table className="docs-table">
          <thead><tr><th>Risk class</th><th><code>readonly</code></th><th><code>ask</code></th><th><code>auto</code></th></tr></thead>
          <tbody>
            <tr><td><code>readonly</code></td><td>Allow</td><td>Allow</td><td>Allow</td></tr>
            <tr><td><code>write</code></td><td>Deny</td><td>Ask operator</td><td>Allow</td></tr>
            <tr><td><code>network</code></td><td>Deny</td><td>Ask operator</td><td>Allow</td></tr>
            <tr><td><code>remote_compute</code></td><td>Deny</td><td>Ask operator</td><td>Allow</td></tr>
            <tr><td><code>destructive</code></td><td>Deny</td><td>Ask operator</td><td>Allow</td></tr>
          </tbody>
        </table>
        <p className="docs-note">
          An <code>ask</code> decision fails closed when the comms session is unavailable. A Discord
          channel ID is not a session token: use an operator-minted endpoint shaped like
          <code> /api/discord/session/&lt;token&gt;</code>. A bad <code>/send</code> endpoint commonly
          surfaces as HTTP 405 and the requested tool remains denied.
        </p>

        <h3 id="tool-files" className="docs-h3">Files, shell, and network</h3>
        <table className="docs-table">
          <thead><tr><th>Tool</th><th>Risk</th><th>What it does</th></tr></thead>
          <tbody>
            <tr><td><code>fs_glob</code></td><td><code>readonly</code></td><td>List workspace paths matching a glob such as <code>**/*.py</code>.</td></tr>
            <tr><td><code>fs_read</code></td><td><code>readonly</code></td><td>Read up to 64 KB of a text file from a byte offset.</td></tr>
            <tr><td><code>fs_write</code></td><td><code>write</code></td><td>Create or overwrite a workspace text file; optionally append and create parent directories.</td></tr>
            <tr><td><code>shell</code></td><td><code>destructive</code></td><td>Run a command with the workspace as cwd; returns capped stdout/stderr. Timeout defaults to 60s and is capped at 300s.</td></tr>
            <tr><td><code>http_fetch</code></td><td><code>network</code></td><td>HTTP GET a URL and return status, content type, and up to 64 KB of text.</td></tr>
          </tbody>
        </table>
        <CopyBlock title="inspect, change, and test a project" code={`hugpy-agent run --policy auto --max-steps 15 \\
  "Use fs_glob to inspect the project, fs_read to understand the relevant files, \\
fs_write to implement the change, shell to run the tests, and final_answer to report it"`} />

        <h3 id="tool-ml" className="docs-h3">Fleet ML and generation</h3>
        <table className="docs-table">
          <thead><tr><th>Tool</th><th>Risk</th><th>What it does</th></tr></thead>
          <tbody>
            <tr><td><code>models_list</code></td><td><code>readonly</code></td><td>List models currently available on the fleet.</td></tr>
            <tr><td><code>summarize</code></td><td><code>remote_compute</code></td><td>Summarize long text with a fleet model.</td></tr>
            <tr><td><code>keywords</code></td><td><code>remote_compute</code></td><td>Extract keywords from text.</td></tr>
            <tr><td><code>embed</code></td><td><code>remote_compute</code></td><td>Create a text embedding; returns dimensions, norm, and optionally the full vector.</td></tr>
            <tr><td><code>similarity</code></td><td><code>remote_compute</code></td><td>Return a semantic similarity score from 0 to 1 for two texts.</td></tr>
            <tr><td><code>transcribe</code></td><td><code>remote_compute</code></td><td>Transcribe an audio or video file inside the workspace.</td></tr>
            <tr><td><code>vision</code></td><td><code>remote_compute</code></td><td>Ask a natural-language question about a workspace image.</td></tr>
            <tr><td><code>classify</code></td><td><code>remote_compute</code></td><td>Classify the contents of a workspace image.</td></tr>
            <tr><td><code>detect</code></td><td><code>remote_compute</code></td><td>Detect image objects and return labels and boxes.</td></tr>
            <tr><td><code>segment</code></td><td><code>remote_compute</code></td><td>Segment an image and save the mask under <code>artifacts/</code>.</td></tr>
            <tr><td><code>depth</code></td><td><code>remote_compute</code></td><td>Create a depth map under <code>artifacts/</code>.</td></tr>
            <tr><td><code>generate_image</code></td><td><code>remote_compute</code></td><td>Generate an image from a prompt and save it under <code>artifacts/</code>; slow and capped per run.</td></tr>
            <tr><td><code>generate_scene</code></td><td><code>remote_compute</code></td><td>Generate a short MP4 scene from a prompt; very slow and capped per run.</td></tr>
          </tbody>
        </table>
        <CopyBlock title="image understanding and generation" code={`cp ~/Pictures/example.jpg "$HUGPY_WORKSPACE/example.jpg"

hugpy-agent run --policy auto \\
  "Use vision on example.jpg and describe it; then use detect to list its objects"

hugpy-agent run --policy auto --max-steps 6 \\
  "Use generate_image to create a 512x512 image of a red barn at dusk"`} />

        <h3 id="tool-meta" className="docs-h3">Memory, operator, and completion</h3>
        <table className="docs-table">
          <thead><tr><th>Tool</th><th>Risk</th><th>What it does</th></tr></thead>
          <tbody>
            <tr><td><code>remember</code></td><td><code>write</code></td><td>Save a durable workspace fact as markdown and index it in <code>memory/MEMORY.md</code>.</td></tr>
            <tr><td><code>ask_operator</code></td><td><code>readonly</code></td><td>Ask a human a clarifying question with one to five response labels and wait for the selected label.</td></tr>
            <tr><td><code>final_answer</code></td><td><code>readonly</code></td><td>Finish exactly once with the complete result. It does not write files; use <code>fs_write</code> first when an artifact is required.</td></tr>
          </tbody>
        </table>
        <CopyBlock title="enumerate the installed registry" code={`"$HOME/hugpy-agent/venv/bin/python" - <<'PY'
from hugpy_agent.config import load_config
from hugpy_agent.gateway import Gateway
from hugpy_agent.memory import Memory
from hugpy_agent.tools import build_registry

cfg = load_config()
registry = build_registry(cfg.workspace, Gateway.from_config(cfg), Memory(cfg.workspace))
specs = getattr(registry, "specs", None)
specs = specs() if callable(specs) else specs
specs = specs if specs is not None else getattr(registry, "_specs", {})
items = specs.values() if isinstance(specs, dict) else specs
for spec in sorted(items, key=lambda item: item.name):
    print(f"{spec.name:24} {spec.risk_class:18} {spec.description}")
PY`} />
      </section>

      {/* ============================ ENROLLMENT ============================ */}
      <section id="enroll" className="docs-sec">
        <h2>Keep it running as a service</h2>
        <p>
          Do this after the first ad-hoc run succeeds. The package installer writes a systemd
          <strong> user</strong> unit plus a mode-0600 environment file. Start with the local queue;
          it has no external integration to configure.
        </p>
        <CopyBlock title="install the queue service" code={`export HUGPY_BASE=https://your-hugpy/api
export HUGPY_API_KEY=hp_REPLACE_ME
export HUGPY_MODEL=Qwen~Qwen3-Coder-Next-GGUF
export HUGPY_WORKSPACE="$HOME/hugpy-agent/workspace"

"$HOME/hugpy-agent/venv/bin/python" -m hugpy_agent.install \\
  --central "$HUGPY_BASE" \\
  --workspace "$HUGPY_WORKSPACE" \\
  --task-source queue \\
  --model "$HUGPY_MODEL" \\
  --policy readonly \\
  --venv "$HOME/hugpy-agent/venv"

systemctl --user status hugpy-agent.service --no-pager`} />

        <h3 className="docs-h3-plain">Task sources</h3>
        <table className="docs-table">
          <thead><tr><th><code>--task-source</code></th><th>How work arrives</th></tr></thead>
          <tbody>
            <tr><td><code>queue</code></td><td>Default simple path. Append one task per line to <code>&lt;workspace&gt;/.hugpy_agent/tasks.queue</code>.</td></tr>
            <tr><td><code>discord-inbox</code></td><td>Optional. Poll a minted comms session for messages beginning with <code>task:</code>, then reply with the outcome.</td></tr>
            <tr><td><em>(unset)</em></td><td><strong>Fail-closed idle</strong> — the daemon heartbeats and does nothing.</td></tr>
          </tbody>
        </table>
        <p className="docs-note">
          To add Discord later, mint a comms session, set <code>HUGPY_DISCORD_SESSION</code>, and rerun
          the installer with <code>--task-source discord-inbox --session "$HUGPY_DISCORD_SESSION"</code>.
        </p>

        <CopyBlock title="dispatch one queue task" code={`printf '%s\\n' 'Summarize the workspace' \\
  >> "$HUGPY_WORKSPACE/.hugpy_agent/tasks.queue"

journalctl --user -u hugpy-agent.service -f`} />
      </section>

      {/* ============================ ESCALATION ============================ */}
      <section id="escalation" className="docs-sec">
        <h2>Approvals: the ask_operator flow</h2>
        <p>
          This is the piece that makes <code>ask</code> mode useful instead of a wall. With an escalation
          channel configured (the operator comms session), a non-readonly tool call in <code>ask</code>
          mode pauses the run and posts a decision request into Discord. You see the tool, its arguments
          (summarised), and a set of labelled options.
        </p>
        <ul className="docs-routes">
          <li><strong>Deny blocks.</strong> The tool call is refused and returned to the model as
            structured data — the run continues and reports the denial honestly; nothing is executed.</li>
          <li><strong>Approve proceeds.</strong> The tool runs, its result flows back into the loop, and
            the run carries on.</li>
          <li><strong>Labels return.</strong> A custom option’s label is handed back to the run verbatim,
            so an escalation can offer more than a yes/no when the model asked for a choice.</li>
        </ul>
        <p className="docs-note">
          Without a valid operator session, <code>ask</code> safely denies non-readonly calls. Use
          <code> readonly</code> when no changes are expected, or <code>auto</code> for trusted work that
          should proceed without approval.
        </p>
      </section>

      {/* ============================ NODE MODE ============================ */}
      <section id="node-mode" className="docs-sec">
        <h2>Node mode: join the fleet</h2>
        <p>
          Node mode is the centrally dispatched alternative to a local queue. The runtime registers,
          stores its node credential, heartbeats, and pulls tasks from central. It uses the same model,
          workspace, tools, and policy as every other service mode.
        </p>
        <CopyBlock title="run as a central node" code={`export HUGPY_API_KEY=hp_REPLACE_ME

"$HOME/hugpy-agent/venv/bin/hugpy-agent" serve \\
  --node \\
  --central "$HUGPY_BASE" \\
  --policy readonly`} />
        <p className="docs-note">
          Registration requires a valid console API key. The node credential returned by central is
          then used for heartbeat and task polling; the runtime manages that lifecycle.
        </p>
      </section>

      {/* ============================ EVALS ============================ */}
      <section id="evals" className="docs-sec">
        <h2>Pick a brain: the eval harness</h2>
        <p>
          Use <code>eval</code> when comparing models—not as part of installation. It runs the same task
          suite through each requested model and reports readiness, pass rate, latency, and tool accuracy.
        </p>
        <pre className="docs-block"><code>{`hugpy-agent eval \\
  --model Qwen~Qwen3-Coder-Next-GGUF \\
  --model ANOTHER-EXACT-MODEL-ID \\
  --policy auto`}</code></pre>
        <p>The scorecard columns:</p>
        <table className="docs-table">
          <thead><tr><th>Column</th><th>Meaning</th></tr></thead>
          <tbody>
            <tr><td><code>ready</code></td><td>The model completed a small chat readiness request.</td></tr>
            <tr><td><code>passed</code></td><td>Tasks passed out of the suite (e.g. <code>3/4</code>). A pass needs <code>outcome==done</code>, under the step cap, <em>and</em> the deterministic checker holding.</td></tr>
            <tr><td><code>steps_avg</code> / <code>tokens_avg</code></td><td>Mean loop steps and estimated tokens per task.</td></tr>
            <tr><td><code>wall_avg</code></td><td>Mean wall-clock per task — the latency signal.</td></tr>
            <tr><td><code>tool_acc</code></td><td>Tool accuracy: the fraction of executed tool calls that returned a non-error result (no bad paths / invalid args).</td></tr>
          </tbody>
        </table>
      </section>

      {/* ============================ TROUBLESHOOTING ============================ */}
      <section id="agent-trouble" className="docs-sec">
        <h2>Troubleshooting</h2>
        <p>
          Most setup problems fit one of these short checks. Fleet-side model problems do not require
          reinstalling the agent client.
        </p>
        <table className="docs-table">
          <thead><tr><th>Symptom</th><th>Why</th><th>Fix</th></tr></thead>
          <tbody>
            <tr>
              <td><strong><code>hugpy-agent</code> is not found</strong></td>
              <td>The wrong venv is active, or the package was installed from the wrong directory.</td>
              <td>Run <code>"$HOME/hugpy-agent/venv/bin/pip" install --upgrade .</code> from the directory containing <code>pyproject.toml</code>.</td>
            </tr>
            <tr>
              <td><strong>The model is unknown, pending, or cannot fit</strong></td>
              <td>The requested registry ID or assigned fleet worker is not ready.</td>
              <td>Copy an exact ID from <code>hugpy-agent models</code>. Admit/repair the assigned worker or choose a model it can serve. Do not reinstall the agent.</td>
            </tr>
            <tr>
              <td><strong><code>401 Unauthorized</code></strong></td>
              <td>The API key is absent, expired, or not accepted by this central.</td>
              <td>Export a valid console-minted <code>HUGPY_API_KEY</code> and retry <code>hugpy-agent models</code>.</td>
            </tr>
            <tr>
              <td><strong>Writes or shell calls are denied</strong></td>
              <td><code>readonly</code>, or <code>ask</code> without a working operator session, blocks non-readonly tools.</td>
              <td>Use <code>--policy auto</code> for trusted work, or configure the Discord approval session.</td>
            </tr>
            <tr>
              <td><strong>Discord returns HTTP 405</strong></td>
              <td>A channel ID or incorrect URL was used where a minted comms-session URL was expected.</td>
              <td>Mint a session and set the complete <code>…/api/discord/session/&lt;token&gt;</code> URL.</td>
            </tr>
            <tr>
              <td><strong>The service is inactive</strong></td>
              <td>The user unit or its saved environment failed to start.</td>
              <td>Run <code>systemctl --user status hugpy-agent.service</code> and <code>journalctl --user -u hugpy-agent.service -n 100</code>.</td>
            </tr>
          </tbody>
        </table>
      </section>

      <div className="docs-pager">
        <a className="docs-pager-card" href="#troubleshooting"><span className="d">Related</span><span className="t">Troubleshooting →</span></a>
      </div>
    </>
  )
}
