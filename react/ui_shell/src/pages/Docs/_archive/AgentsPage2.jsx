// Agents — end-to-end operator walkthrough for the hugpy agent system
// (the portable `hugpy-agent` runtime that uses the fleet as its brain).
// Written 2026-07-15 for an operator who has never driven the system
// successfully: concepts → prereqs → quickstart → enrollment → approvals →
// subagents → node mode → evals → troubleshooting, in operator order.
//
// Ground truth: hugpy_agent/README.md + bootstrap.sh, STATUS.md (live-verified
// behaviour + known quirks), the P3.1 /agent/* blueprint handoff, and the
// 2026-07-15 eval scorecard. Behaviours marked "pending deploy" are the
// /agent/* node routes, which are built but not yet live on central.
//
// Section ids are addressable as #agents/<id>; keep them stable if other pages
// start linking here.
import { CopyBlock } from './docParts'

export const AGENTS_TOC = [
  ['concepts', 'What the agent system is', []],
  ['prereqs', 'Before you start', []],
  ['quickstart', 'Quickstart: your first run', []],
  ['enroll', 'Enroll a box as a service', []],
  ['escalation', 'Approvals: the ask_operator flow', []],
  ['subagents', 'Subagents (delegation)', []],
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
        crash-safe journal, and a policy gate. This page walks an operator all the way through — from
        a single ad-hoc run to a box enrolled as a background service — and calls out the traps that
        make the system look broken when it is actually doing exactly what it was told.
      </p>

      <div className="docs-callout">
        <div className="docs-callout-t">Read this first — the #1 “using it wrong” trap</div>
        <p>
          The default policy is <code>ask</code>, and until an escalation channel is configured
          <code> ask</code> <strong>fails closed to deny</strong> every non-readonly tool. A run that
          tries to write a file, run a shell command, or fetch a URL is refused — cleanly, as data the
          model reports back — and looks like the agent “can’t do anything”. It can; it is waiting for
          an approval path that isn’t wired yet. For a trusted ad-hoc run, pass{' '}
          <code>--policy auto</code> (or <code>HUGPY_POLICY=auto</code>). To keep <code>ask</code> and
          actually get approvals, wire the Discord escalation channel (
          <a className="docs-inline-link" href="#agents/escalation">Approvals</a>).
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
            is <code>ponpoke/flux2-klein-9b-uncensored-text-encoder</code> (a Qwen3-family model — see
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
        <p>Three things have to be true before an agent can do useful work.</p>

        <h3 className="docs-h3-plain">1 · An API key</h3>
        <p>
          Mint a key in the console under <strong>API access</strong> (keys look like <code>hp_…</code>,
          shown once, stored hashed). The <code>/v1</code> family is <strong>designed to enforce keys</strong>{' '}
          for chat, so get one and set it as <code>HUGPY_API_KEY</code>. Prefer the environment (or a
          <code> 0600 .env</code>) over a flag so it never lands in <code>ps</code> or the shell history.
        </p>
        <p className="docs-note">
          The <code>/v1</code> key requirement is a <strong>sitewide posture toggle</strong>: a deployment
          can be configured keyless by design, so whether an anonymous <code>/v1</code> call is refused
          depends on that deployment's own configuration. Never build a workflow that assumes anonymous
          access — always mint and send a key, and check <a className="docs-inline-link"
          href="#agents/agent-trouble">the 401 entry</a> below if you are being rejected. Note that the
          agent enrollment gate is a <em>separate</em> category and is <strong>not</strong> covered by that
          toggle — see <a className="docs-inline-link" href="#agents/node-mode">Node mode</a>.
        </p>

        <h3 className="docs-h3-plain">2 · A Discord comms session (for daemon tasks &amp; approvals)</h3>
        <p>
          Ad-hoc runs need no Discord. But the daemon’s <code>discord-inbox</code> source and the
          <code> ask</code>-mode escalation flow both speak through an operator-minted <em>comms
          session</em> — a scoped bearer token that can read and post to exactly one channel. Mint it in
          the console (<a className="docs-inline-link" href="#console-discord/sessions">Mint a comms
          session</a>) and pass its endpoint URL as <code>HUGPY_DISCORD_SESSION</code>.
        </p>
        <div className="docs-callout">
          <div className="docs-callout-t">Known quirk — option buttons don’t render yet</div>
          <p>
            When the agent asks for a decision it offers labelled options. On this deployment the bot
            posts them as <strong>plain text, not clickable buttons</strong>. The fallback works and is
            what acceptance rode on: <strong>type a reply whose text matches the offered label</strong>
            (e.g. <code>Approve</code>, <code>Deny</code>) and it is received as that choice.
          </p>
        </div>

        <h3 className="docs-h3-plain">3 · A brain that is actually ready</h3>
        <p>
          Brains serve from your LLM workers. A model that is down or still loading does <strong>not</strong>
          fail with an HTTP error — <code>/v1/chat/completions</code> returns <strong>HTTP 200 whose body
          is an error string</strong> like <code>[error: … 404 NOT FOUND …]</code>. So “200 came back”
          proves nothing. Confirm readiness with a real token echo, not a status code — the eval harness
          does exactly this (<a className="docs-inline-link" href="#agents/evals">Pick a brain</a>), and
          the same false-200 trap is the first row in{' '}
          <a className="docs-inline-link" href="#agents/agent-trouble">Troubleshooting</a>.
        </p>
      </section>

      {/* ============================ QUICKSTART ============================ */}
      <section id="quickstart" className="docs-sec">
        <h2>Quickstart: your first run</h2>
        <p>
          Install the runtime (stdlib-only, Python ≥ 3.10 — zero dependencies), point it at your fleet,
          and run one task. The <code>models</code> command is the smoke test that the base URL and key
          resolve.
        </p>
        <pre className="docs-block"><code>{`python3 -m venv .venv && . .venv/bin/activate
pip install -e path/to/hugpy_agent       # the hugpy_agent/ package dir, NOT the repo root
                                         # (a bare "." only works if you cd'd INTO hugpy_agent/)

export HUGPY_BASE=https://your-hugpy/api        # may end in /api, /v1, /api/v1, or be a bare origin
export HUGPY_API_KEY=hp_REPLACE_ME              # from the console's API access tab
hugpy-agent models                              # lists fleet models — proves base + key resolve`}</code></pre>
        <p className="docs-note">
          The command is <code>hugpy-agent</code> (singular, hyphenated). If it's "not found" after
          install, you almost certainly ran <code>pip install -e .</code> from the repo root instead of
          the <code>hugpy_agent/</code> package directory — that's the package whose{' '}
          <code>pyproject.toml</code> declares the <code>hugpy-agent</code> entry point.
        </p>
        <p>
          Config precedence is <strong>env &gt; <code>.env</code> in the workspace &gt;
          <code> agent.toml</code> in the workspace</strong>; CLI flags (<code>--base</code>,
          <code> --model</code>, <code>--workspace</code>, …) beat everything. Secrets belong in env or a
          gitignored <code>.env</code> — never in <code>agent.toml</code> or the source.
        </p>
        <p>
          Now a real run. Because the default policy denies writes until escalation is configured, pass{' '}
          <code>--policy auto</code> for this trusted first run:
        </p>
        <pre className="docs-block"><code>{`hugpy-agent run "Read the files in this workspace, describe what the project \\
does, and write your findings to report.md" --policy auto
# progress streams to stderr; the final JSON report {outcome, steps, ...} prints to stdout`}</code></pre>
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

        <div className="docs-callout">
          <div className="docs-callout-t">The think-suffix note (why a run sometimes “does nothing”)</div>
          <p>
            Qwen3-family thinking brains (the default <code>flux2-klein</code> is one) will otherwise
            spend the entire token budget inside a <code>&lt;think&gt;</code> block and never emit a tool
            call. The harness defends against this by appending <code>&nbsp;/no_think</code> to the wire
            copy of the latest user turn (never to stored history); it is on by default
            (<code>HUGPY_NO_THINK=true</code>). Leave it on for Qwen3-family brains. Coder variants (e.g.
            <code> Qwen3-Coder-Next</code>) have no thinking mode and echo the suffix as inert text —
            harmless, no special handling needed. Disable only with <code>--think</code> /
            <code> no_think = false</code> if you know your brain wants it.
          </p>
        </div>
      </section>

      {/* ============================ ENROLLMENT ============================ */}
      <section id="enroll" className="docs-sec">
        <h2>Enroll a box as a service</h2>
        <p>
          <code>bootstrap.sh</code> takes a bare box to a running systemd <strong>user</strong> service
          in one command (idempotent — re-run to upgrade). Export secrets first so they ride the
          environment into the installer, not the process argv:
        </p>
        <CopyBlock title="enroll a box (one command)" code={`export HUGPY_API_KEY=hp_REPLACE_ME          # prefer env over --key for secrets
bash bootstrap.sh --central https://your-hugpy/api \\
    --session https://your-hugpy/api/discord/session/SESS_TOKEN_REPLACE_ME \\
    --task-source discord-inbox \\
    --workspace ~/hugpy-agent/workspace

journalctl --user -u hugpy-agent -f        # watch it heartbeat / run tasks`} />
        <p>What it writes:</p>
        <ul className="docs-routes">
          <li>A venv at <code>~/hugpy-agent/venv</code> and the package installed into it.</li>
          <li><code>~/.config/systemd/user/hugpy-agent.service</code> (<code>Restart=on-failure</code>,
            <code> %h</code>-portable, <code>ExecStart=… serve</code>).</li>
          <li><code>~/.config/hugpy-agent/agent.env</code> at <strong>mode 0600</strong> holding all
            config <em>including the key</em> — never the unit file, never argv.</li>
          <li>Linger enabled, then the unit enabled and started.</li>
        </ul>
        <div className="docs-callout">
          <div className="docs-callout-t">Linger precondition on headless boxes</div>
          <p>
            A box with no active login session (no <code>XDG_RUNTIME_DIR</code>) cannot host systemd
            <em> user</em> units until <strong>linger</strong> is enabled for the user —
            <code> loginctl enable-linger &lt;user&gt;</code>. <code>bootstrap.sh</code> enables it for
            you; the note matters when you are diagnosing a unit that “won’t start” on a server you only
            ever reach over SSH. This is the same precondition the GPU-worker unit has.
          </p>
        </div>

        <h3 className="docs-h3-plain">Task sources</h3>
        <table className="docs-table">
          <thead><tr><th><code>--task-source</code></th><th>How work arrives</th></tr></thead>
          <tbody>
            <tr><td><code>discord-inbox</code></td><td>Polls the operator comms session for inbound messages starting with <strong><code>task:</code></strong> — the rest of the message is the task. The outcome is replied into the channel (<code>task finished (run &lt;id&gt;): outcome=… steps=…</code>).</td></tr>
            <tr><td><code>queue</code></td><td>A local file (<code>HUGPY_TASK_QUEUE</code>, default <code>&lt;workspace&gt;/.hugpy_agent/tasks.queue</code>), one task per line. Append a line to dispatch; one task is consumed atomically per poll.</td></tr>
            <tr><td><em>(unset)</em></td><td><strong>Fail-closed idle</strong> — the daemon heartbeats and does nothing.</td></tr>
          </tbody>
        </table>
        <p className="docs-note">
          On startup the first <code>discord-inbox</code> poll only sets the message watermark —
          historical <code>task:</code> messages are never replayed. If a task landed while the daemon
          was down, re-send it. The daemon polls every <code>HUGPY_POLL_INTERVAL</code> (10s);
          <code> systemctl --user stop hugpy-agent</code> finishes the current task, then exits.
        </p>

        <h3 className="docs-h3-plain">Verify the enrollment</h3>
        <pre className="docs-block"><code>{`systemctl --user is-enabled hugpy-agent      # -> enabled
journalctl --user -u hugpy-agent -f          # expect: poll / heartbeat lines, no errors`}</code></pre>
        <p>
          Then post <code>task: what is 2+2? reply with just the number</code> into the bound channel and
          expect the daemon to reply <code>task finished (run &lt;id&gt;): outcome=… steps=…</code>
          within seconds (live acceptance saw a <code>task:</code> → channel answer round-trip in
          ~12&nbsp;s). If nothing comes back, the brain may be cold — see the false-200 and “no tool call”
          rows in <a className="docs-inline-link" href="#agents/agent-trouble">Troubleshooting</a>.
        </p>
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
        <div className="docs-callout">
          <div className="docs-callout-t">Answer by typing the label</div>
          <p>
            Because the option buttons don’t render on this deployment (see{' '}
            <a className="docs-inline-link" href="#agents/prereqs">Before you start</a>), respond by
            <strong> typing a message whose text matches the offered label</strong> —
            <code> Approve</code>, <code>Deny</code>, or a custom label exactly as shown. Until this
            channel exists, <code>ask</code> denies every non-readonly tool by default (fail-closed) — so
            an agent that “refuses to write anything” almost always means escalation isn’t wired, not
            that the run failed.
          </p>
        </div>
      </section>

      {/* ============================ SUBAGENTS ============================ */}
      <section id="subagents" className="docs-sec">
        <h2>Subagents (delegation)</h2>
        <p>
          A run can delegate with the <code>spawn</code> tool: it launches a child run for a scoped
          sub-task and folds the child’s result back into its own reasoning (a parent run has been
          accepted spawning two linked children and synthesising their answers). The rule that matters:
          <strong> delegation never widens authority</strong>. A child inherits the parent’s policy and
          workspace jail — it can do no more than the parent could, so you cannot escape
          <code> readonly</code> or an <code>ask</code> denial by spawning a subagent. Authority only ever
          narrows down the tree, never up.
        </p>
      </section>

      {/* ============================ NODE MODE ============================ */}
      <section id="node-mode" className="docs-sec">
        <h2>Node mode: join the fleet</h2>
        <div className="docs-callout">
          <div className="docs-callout-t">Available on this dev central — confirm elsewhere</div>
          <p>
            The base <code>/agent/*</code> routes below (register, heartbeat, pull, roster, dispatch) are{' '}
            <strong>available on this dev central</strong>. A result-retrieval route is landing in a
            sibling slice around the same time as this page. Everything above (ad-hoc runs, the
            <code> serve</code> daemon with <code>discord-inbox</code>/<code>queue</code> sources) works
            everywhere and is the supported path regardless. Node mode's availability on other
            deployments (including production) is not guaranteed by this page —
            <strong> confirm with your fleet admin</strong> before relying on it there.
          </p>
        </div>
        <p>
          Node mode lets a box enroll with central like a GPU worker: it registers, holds a token,
          heartbeats, and pulls dispatched tasks. The operator dispatches from the console. The lifecycle:
        </p>
        <table className="docs-table">
          <thead><tr><th>Step</th><th>Route</th><th>Auth</th></tr></thead>
          <tbody>
            <tr><td><strong>register</strong></td><td><code>POST /agent/register</code> <code>{'{name, host, capabilities}'}</code> → <code>{'{id, token, …}'}</code></td><td><strong>Console API key, always</strong> — not waivable by the sitewide key toggle or <code>HUGPY_AGENT_OPEN</code>; mints a one-time <code>agt_…</code> token</td></tr>
            <tr><td><strong>heartbeat</strong></td><td><code>POST /agent/&lt;id&gt;/heartbeat</code> <code>{'{status, current_task, version}'}</code></td><td>Node token (<code>Authorization: Bearer agt_…</code>)</td></tr>
            <tr><td><strong>pull</strong></td><td><code>GET /agent/&lt;id&gt;/tasks?since=&lt;seq&gt;</code> → <code>{'{tasks, cursor, …}'}</code></td><td>Node token; monotonic cursor, idempotent re-pull</td></tr>
            <tr><td><strong>roster</strong> <span className="docs-note">(operator)</span></td><td><code>GET /agent/nodes</code> → full roster + live status</td><td>Operator-only</td></tr>
            <tr><td><strong>dispatch</strong> <span className="docs-note">(operator)</span></td><td><code>POST /agent/&lt;id&gt;/dispatch</code> <code>{'{task}'}</code></td><td>Operator-only</td></tr>
          </tbody>
        </table>
        <p>
          The token is returned <strong>once</strong> at register and stored only as a sha256 hash —
          persist it on the node; it can’t be re-read. Node auth fails closed: a missing/bad token is
          <code> 401</code>, an unknown node is <code>410</code> (re-register), a revoked node is
          <code> 403</code>. All routes are <code>/api</code>-dual-mounted like the workers.
        </p>
        <div className="docs-callout">
          <div className="docs-callout-t">Enrolling always needs a console API key</div>
          <p>
            <code>POST /agent/register</code> requires a valid console-minted API key —{' '}
            <strong>permanently, and independent of every other switch</strong>. Agent keys are a separate
            category from the <code>/v1</code> chat key: the <code>/v1</code> gate is a deliberate sitewide
            posture toggle a deployment may turn off, but <em>that toggle does not reach enrollment</em>,
            and neither does <code>HUGPY_AGENT_OPEN</code>. There is no configuration in which
            <code> /agent/register</code> accepts an anonymous caller.
          </p>
          <p>
            This is intended, not an oversight. Register is the fleet <strong>bootstrap</strong> — the one
            route that <em>mints</em> node credentials — and it is publicly reachable, so it is the
            hardest door, not the softest. Every agent node must be handed a key to enroll. Mint one in
            the console under <strong>API access</strong>; keys are individually revocable.
          </p>
        </div>
        <div className="docs-callout">
          <div className="docs-callout-t">Keep <code>HUGPY_AGENT_OPEN</code> unset in production</div>
          <p>
            <code>HUGPY_AGENT_OPEN</code> is a default-closed escape hatch for bring-up. Its scope is
            narrow: it waives <em>only</em> the <strong>operator gate</strong> on <code>nodes</code>,{' '}
            <code>dispatch</code>, and <code>tasks/&lt;seq&gt;</code>. It does <strong>not</strong> waive
            the register API-key gate, and it does <strong>not</strong> waive node-token auth on
            heartbeat/tasks/result — a node's token is its identity, not a human gate.
          </p>
          <p className="docs-note">
            On a deployment reachable from the public internet, setting it exposes the roster and dispatch
            to anyone — <strong>leave it unset in production</strong> unless you deliberately want those
            operator routes open. Unset the variable and restart to restore the operator gate.
          </p>
        </div>
      </section>

      {/* ============================ EVALS ============================ */}
      <section id="evals" className="docs-sec">
        <h2>Pick a brain: the eval harness</h2>
        <p>
          “Which brain” is data, not vibes. <code>hugpy-agent eval</code> runs a small suite of
          deterministic tasks against each model <em>through the real agent loop</em> and emits a
          comparative scorecard. Each task has a deterministic checker (an artifact appeared on disk / the
          answer contains a required fact / the run finished under its step cap) — never an LLM judging an
          LLM.
        </p>
        <pre className="docs-block"><code>{`hugpy-agent eval --model ponpoke/flux2-klein-9b-uncensored-text-encoder \\
                 --model Qwen/Qwen3-Coder-Next-GGUF
# or straight from a checkout, no install:
python evals/runner.py --model A --model B`}</code></pre>
        <p>The scorecard columns:</p>
        <table className="docs-table">
          <thead><tr><th>Column</th><th>Meaning</th></tr></thead>
          <tbody>
            <tr><td><code>ready</code></td><td>Passed the <strong>chat token-echo</strong> readiness gate. This is gated on a real echoed token, <em>never</em> on HTTP 200 or an <code>/api/llm/serving</code> mode flag — a model that never becomes servable gets <code>ready=NO</code> so the blocker is in the data, not a hang.</td></tr>
            <tr><td><code>passed</code></td><td>Tasks passed out of the suite (e.g. <code>3/4</code>). A pass needs <code>outcome==done</code>, under the step cap, <em>and</em> the deterministic checker holding.</td></tr>
            <tr><td><code>steps_avg</code> / <code>tokens_avg</code></td><td>Mean loop steps and (client-side estimated) tokens per task — <code>usage</code> is null at the seam today, so tokens are an estimate.</td></tr>
            <tr><td><code>wall_avg</code></td><td>Mean wall-clock per task — the latency signal.</td></tr>
            <tr><td><code>tool_acc</code></td><td>Tool accuracy: the fraction of executed tool calls that returned a non-error result (no bad paths / invalid args).</td></tr>
          </tbody>
        </table>
        <p>
          The first live run (2026-07-15) is a clean illustration of the tradeoff you’re weighing:
          <code> flux2-klein</code> passed <strong>3/4</strong> at <strong>~5.6&nbsp;s</strong>/task (it
          tripped its own prompted-format terminator on the aggregate task), while
          <code> Qwen3-Coder-Next</code> passed <strong>4/4</strong> but at <strong>~26&nbsp;s</strong>/task.
          Speed vs. reliability — the scorecard hands you the numbers, the call is yours. A larger suite
          tightens the pass-rate signal.
        </p>
      </section>

      {/* ============================ TROUBLESHOOTING ============================ */}
      <section id="agent-trouble" className="docs-sec">
        <h2>Troubleshooting</h2>
        <p>
          The failure modes that make the agent system look broken, and what each one actually is.
          Several are the same false signals the harness was hardened against — if you gate on them by
          hand you’ll be misled the same way.
        </p>
        <table className="docs-table">
          <thead><tr><th>Symptom</th><th>Why</th><th>Fix</th></tr></thead>
          <tbody>
            <tr>
              <td><strong>A run “succeeds” but the answer is garbage / a chat call “worked” but the reply is <code>[error: …]</code></strong></td>
              <td>The false-200 trap: while a worker is down or loading, <code>/v1/chat/completions</code> returns HTTP&nbsp;200 whose <em>body</em> is an error string (<code>[error: … 404 NOT FOUND …]</code>).</td>
              <td>Never gate on “200 + text”. Wait for the model to warm and retry; confirm readiness with a token echo (the eval <code>ready</code> gate), not a status code.</td>
            </tr>
            <tr>
              <td><strong><code>/api/llm/serving/&lt;key&gt;</code> says <code>mode=off</code> but chat is clearly being served</strong></td>
              <td>The serving route reports <em>config</em>, not live state — it can read <code>off</code> (even flapping <code>off→swap→off</code>) while the worker is actively serving.</td>
              <td>Don’t use the serving mode flag as a liveness check. Prove serving with an actual chat round-trip.</td>
            </tr>
            <tr>
              <td><strong>A model works on <code>/v1</code> but the serving route 404s on the same name (or vice-versa)</strong></td>
              <td>Two naming seams: the <code>/v1</code> chat seam wants the full <code>owner/…</code> id (<code>ponpoke/…</code>, <code>Qwen/…</code>); the central serving route / catalog wants the <strong>bare</strong> model name (<code>flux2-klein-…</code>, or <code>~</code>-separated).</td>
              <td>Use the <code>owner/…</code> id for <code>--model</code> / <code>HUGPY_MODEL</code> (the chat path). Use the bare name only when calling the serving route directly.</td>
            </tr>
            <tr>
              <td><strong>The reply has no tool call — the model narrates instead of acting</strong></td>
              <td>Either the brain is a Qwen3-family thinker with <code>/no_think</code> disabled (it burns the budget inside <code>&lt;think&gt;</code>), or the model is cold/churning and returns prose, or a weak brain emitted the bare word <code>final_answer</code> instead of a <code>&lt;tool_call&gt;{'{…}'}&lt;/tool_call&gt;</code> envelope.</td>
              <td>Keep <code>HUGPY_NO_THINK=true</code> for Qwen3-family brains. Retry once the model is warm. If a specific brain does this repeatedly on a task, the eval scorecard will show it as a tool-call-discipline miss — pick a more reliable brain.</td>
            </tr>
            <tr>
              <td><strong><code>401 Unauthorized</code> on chat / <code>models</code> / <code>register</code></strong></td>
              <td>No valid key was sent. On chat/<code>models</code> that means the deployment's sitewide key gate is on. On <code>POST /agent/register</code> it is <em>always</em> the cause — enrollment requires a key unconditionally, so no toggle or <code>HUGPY_AGENT_OPEN</code> will make this 401 go away.</td>
              <td>Mint a key under the console’s API access tab and set <code>HUGPY_API_KEY</code> (env or <code>0600 .env</code>); for enrollment pass it as <code>Authorization: Bearer &lt;key&gt;</code>. Confirm the base URL includes the <code>/api</code> hop your deployment expects.</td>
            </tr>
            <tr>
              <td><strong>Every write/shell/fetch is denied and the run “can’t do anything”</strong></td>
              <td>Default <code>ask</code> mode with no escalation channel fails closed to deny for all non-readonly tools.</td>
              <td>Either wire the Discord escalation channel (<a className="docs-inline-link" href="#agents/escalation">Approvals</a>) and answer by typing the label, or pass <code>--policy auto</code> (<code>HUGPY_POLICY=auto</code>) for a trusted ad-hoc run.</td>
            </tr>
            <tr>
              <td><strong>The enrolled user unit won’t start on a headless box</strong></td>
              <td>No login session / no <code>XDG_RUNTIME_DIR</code> — systemd user units can’t run without linger.</td>
              <td><code>loginctl enable-linger &lt;user&gt;</code> (<code>bootstrap.sh</code> does this; re-run it, or set linger by hand), then <code>systemctl --user enable --now hugpy-agent</code>.</td>
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
