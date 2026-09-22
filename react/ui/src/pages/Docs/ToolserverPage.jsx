// Toolserver, abstract-claude, and the mct frontier seat — the tool + Claude-seat
// layer built 2026-08-31. Static reference: the toolserver HTTP tool API and its
// categories, the MCP bridge that advertises them to Claude Code, the
// abstract-claude package (launch / mct / oauth / reset), how hugpy-station's
// frontier seat now runs on it, fresh-VM seat provisioning, the secure install
// from dev.hugpy.ai, and the central DB-backed todo board. VM control lives on
// its own page — see "VM pool & API".
import { CopyBlock } from './docParts'

export const TOOLSERVER_TOC = [
  ['ts-overview', 'Overview', []],
  ['ts-toolserver', 'The toolserver', [
    ['ts-categories', 'Tool categories'],
    ['ts-delegation', 'Delegation (ui/media/vm)'],
    ['ts-access', 'Access & security'],
  ]],
  ['ts-abstract-claude', 'abstract-claude', [
    ['ts-ac-launch', 'launch — quota fallback'],
    ['ts-ac-oauth', 'Durable OAuth'],
    ['ts-ac-reset', 'Fresh sessions & template'],
  ]],
  ['ts-mcp', 'MCP bridge', []],
  ['ts-mct', 'mct — the frontier seat', []],
  ['ts-station', 'Station integration', [
    ['ts-seats', 'Seat provisioning'],
    ['ts-install', 'Secure install'],
  ]],
  ['ts-dbarm', 'The DB arm', [
    ['ts-todo', 'Central todo board'],
    ['ts-comms', 'comms.ping'],
    ['ts-board', 'Central board & mirror'],
    ['ts-metrics', 'Model metrics'],
  ]],
]

export default function ToolserverPage() {
  return (
    <article className="docs-article">
      <h1>Toolserver &amp; abstract-claude</h1>

      <section id="ts-overview" className="docs-sec">
        <p>
          The <strong>toolserver</strong> exposes the <code>abstract_*</code> ecosystem as an
          API-callable tool layer, and <strong>abstract-claude</strong> is the CLI that runs a
          Claude Code seat against it. Together they give the fleet one authed entry point —{' '}
          <a href="https://toolserver.hugpy.ai" target="_blank" rel="noreferrer">toolserver.hugpy.ai</a>{' '}
          — that a <code>.claude</code> session discovers over MCP and drives as native tools:
          filesystem, web, database, media, computer-use, VM control, and Claude-session
          management, all behind the LAN/WireGuard wall.
        </p>
      </section>

      <section id="ts-toolserver" className="docs-sec">
        <h2 className="docs-h2">The toolserver</h2>
        <p>
          <code>abstract_toolserver</code> is a Flask app that turns each tool function into a
          self-describing HTTP endpoint. It runs under gunicorn as{' '}
          <code>7004_hugpy_toolserver</code> on <strong>ae</strong> (<code>127.0.0.1:7004</code>),
          fronted by nginx at <code>toolserver.hugpy.ai</code>. Discovery is built in:
        </p>
        <table className="docs-table">
          <thead><tr><th>endpoint</th><th>what it gives</th></tr></thead>
          <tbody>
            <tr><td><code>GET /prefixes</code></td><td>the tool categories</td></tr>
            <tr><td><code>GET /endpoints</code></td><td>every tool as <code>{'{endpoint, url, methods}'}</code></td></tr>
            <tr><td><code>GET /&lt;cat&gt;/&lt;tool&gt;?help=true</code></td><td>that tool&rsquo;s signature/help</td></tr>
          </tbody>
        </table>
        <p>Call a tool with JSON; the reply is <code>{'{"result": …}'}</code> (or <code>{'{"error": …}'}</code>).</p>
        <CopyBlock
          title="discover and call"
          code={`curl -s https://toolserver.hugpy.ai/prefixes
curl -s https://toolserver.hugpy.ai/text/count_tokens -d '{"text":"hello world"}'
# {"result": 2}`}
        />

        <div id="ts-categories" className="docs-h3-block">
          <h3 className="docs-h3">Tool categories</h3>
          <table className="docs-table">
            <thead><tr><th>category</th><th>tools</th><th>runs on</th></tr></thead>
            <tbody>
              <tr><td><code>fs</code></td><td>search, read_span, extract, read_file, write_file, read_json, find_keys, find_paths, glob, imports, find_content</td><td>ae</td></tr>
              <tr><td><code>text</code></td><td>count_tokens, chunk, detect_language</td><td>ae</td></tr>
              <tr><td><code>web</code></td><td>text, links, attributes</td><td>ae</td></tr>
              <tr><td><code>db</code></td><td>tables, schema, columns, fetch, query <em>(read-only)</em></td><td>ae · dedicated <code>toolserver</code> DB</td></tr>
              <tr><td><code>todo</code></td><td>add, list, update, done, remove <em>(scoped write)</em></td><td>ae · <code>toolserver</code> DB</td></tr>
              <tr><td><code>claude</code></td><td>state, oauth_status, oauth_probe, oauth_set, set_model, save_template, restore, reset</td><td>ae</td></tr>
              <tr><td><code>ui</code></td><td>capture, ocr, monitors, windows, click_verify</td><td>computron (desktop)</td></tr>
              <tr><td><code>media</code></td><td>ocr_image, pdf_text, summarize, keywords, transcribe</td><td>computron (GPU)</td></tr>
              <tr><td><code>vm</code></td><td>list, state, start, stop, run, console, ip, exec</td><td>computron — see <a href="#/vmpool">VM pool</a></td></tr>
              <tr><td><code>sys</code></td><td>run_cmd <em>(allowlist-gated, off by default)</em></td><td>ae</td></tr>
            </tbody>
          </table>
        </div>

        <div id="ts-delegation" className="docs-h3-block">
          <h3 className="docs-h3">Delegation (ui / media / vm)</h3>
          <p>
            The computer-use, media, and VM tools need a real desktop and a GPU, so those whole
            categories are <strong>delegated</strong> to <strong>computron</strong>
            (<code>192.168.1.128</code>), which runs its own toolserver on <code>:7004</code> under a
            virtual display. ae proxies them transparently via{' '}
            <code>TOOLSERVER_DELEGATE</code> — callers still hit one URL.
          </p>
          <CopyBlock
            title="delegation config (ae unit env)"
            code={`TOOLSERVER_DELEGATE={"ui":"http://192.168.1.128:7004","media":"http://192.168.1.128:7004","vm":"http://192.168.1.128:7004"}`}
          />
        </div>

        <div id="ts-access" className="docs-h3-block">
          <h3 className="docs-h3">Access &amp; security</h3>
          <p>
            <code>toolserver.hugpy.ai</code> is gated{' '}
            <strong>LAN (<code>192.168.1.0/24</code>) + WireGuard (<code>192.168.2.0/24</code>) only</strong>{' '}
            — the gate is the auth. The <code>db</code> tools are read-only (SELECT gate);{' '}
            <code>sys.run_cmd</code> is off unless an allowlist is set; <code>todo</code> is the one
            scoped write path (its own table, validated columns).
          </p>
        </div>
      </section>

      <section id="ts-abstract-claude" className="docs-sec">
        <h2 className="docs-h2">abstract-claude</h2>
        <p>
          A stdlib-only, <code>pip</code>-installable CLI that manages a Claude Code seat: quota
          fallback, durable auth, fresh provisioned sessions, the MCP bridge, and the{' '}
          <code>mct</code> frontier REPL. It is the single source of truth for the frontier seat.
        </p>
        <CopyBlock title="install" code={`pip install -U abstract-claude`} />
        <table className="docs-table">
          <thead><tr><th>command</th><th>what it does</th></tr></thead>
          <tbody>
            <tr><td><code>launch</code></td><td>start Claude with quota fallback + token injection</td></tr>
            <tr><td><code>mct</code></td><td>the pointer-exchange frontier REPL (see below)</td></tr>
            <tr><td><code>mcp</code></td><td>stdio MCP bridge to the toolserver</td></tr>
            <tr><td><code>oauth-set / oauth-mint / oauth-status</code></td><td>durable 1-year token</td></tr>
            <tr><td><code>reset / restore / save-template</code></td><td>fresh-session lifecycle (keeps login)</td></tr>
            <tr><td><code>serve / install-service</code></td><td>the settings web console (:9111)</td></tr>
          </tbody>
        </table>

        <div id="ts-ac-launch" className="docs-h3-block">
          <h3 className="docs-h3">launch — quota fallback</h3>
          <p>
            Claude Code&rsquo;s <code>fallbackModel</code> only fires on overload, never on quota,
            and it has no auto-switch when a model is <em>maxed</em>.{' '}
            <code>abstract-claude launch</code> handles the real case: if the default (e.g. Fable 5)
            is over quota it starts on the fallback (Opus 4.8) instead. Configure in{' '}
            <code>config.json</code> (<code>default_model</code>, <code>quota_fallback_model</code>).
          </p>
          <CopyBlock
            title="use it as your claude"
            code={`alias claude='abstract-claude launch --dangerously-skip-permissions'`}
          />
        </div>

        <div id="ts-ac-oauth" className="docs-h3-block">
          <h3 className="docs-h3">Durable OAuth</h3>
          <p>
            Login rides on a durable <strong>1-year</strong> token (<code>CLAUDE_CODE_OAUTH_TOKEN</code>),
            stored <code>0600</code> in <code>~/.config/claude-auth/</code> and injected by{' '}
            <code>launch</code> — so headless seats, the MCP bridge, and cron never drop to{' '}
            <code>/login</code>. Note: an <em>interactive</em> TUI still needs one real{' '}
            <code>/login</code> per machine (Claude Code reads <code>~/.claude/.credentials.json</code>
            for interactive startup; the env token covers headless).
          </p>
          <CopyBlock
            title="seed the token (any of)"
            code={`abstract-claude oauth-mint            # browser once, captured + stored
abstract-claude oauth-set sk-ant-oat01-...   # paste a token minted elsewhere`}
          />
        </div>

        <div id="ts-ac-reset" className="docs-h3-block">
          <h3 className="docs-h3">Fresh sessions &amp; template</h3>
          <p>
            The template is <strong>config only</strong> — a <code>settings.json</code> plus the
            minimal <code>~/.claude.json</code> flags (onboarding/theme/trust) and the MCP server
            registration. No credentials or transcripts. <code>reset</code> wipes prior state for a
            genuinely clean session but <strong>preserves the login</strong> by default
            (<code>--hard</code> also removes it).
          </p>
          <CopyBlock title="fresh, provisioned, tool-aware session" code={`abstract-claude reset -y && abstract-claude launch`} />
        </div>
      </section>

      <section id="ts-mcp" className="docs-sec">
        <h2 className="docs-h2">MCP bridge</h2>
        <p>
          The toolserver is a plain HTTP API; Claude Code learns tools over{' '}
          <strong>MCP</strong> (Model Context Protocol). <code>abstract-claude mcp</code> is a stdio
          MCP server that <strong>discovers</strong> the toolserver&rsquo;s <code>/endpoints</code>
          (+ <code>?help</code>) at connect time and advertises each as a native tool; a{' '}
          <code>tools/call</code> forwards to the HTTP endpoint. Because it reads live discovery,
          adding a tool to the toolserver makes it appear in the session automatically.
        </p>
        <CopyBlock
          title="register it for your sessions"
          code={`claude mcp add -s user toolserver \\
  -e TOOLSERVER_URL=https://toolserver.hugpy.ai -- abstract-claude mct
# then in a session:  /mcp   (lists the toolserver + its tools)`}
        />
        <p>
          The fresh-session template registers this automatically, so every provisioned seat is
          tool-aware through the one authed URL.
        </p>
      </section>

      <section id="ts-mct" className="docs-sec">
        <h2 className="docs-h2">mct — the frontier seat</h2>
        <p>
          <code>mct</code> (Mediated Context, promoted from &ldquo;mct2&rdquo; and now the only
          name) is a <strong>pointer-exchange REPL</strong> over a Claude session. You (C) type a
          prompt; the deterministic REPL (B) writes it to a file; Claude (A, <code>claude -p
          --resume</code>) receives only the file <em>pointers</em>, reads the prompt, and writes
          its answer to a response file; your terminal renders both as a normal chat — while A&rsquo;s
          context carried nothing but pointers. It adds persistent history (<code>--resume</code>),
          mid-turn arbitration lanes, flight state, and a capture lane.
        </p>
        <table className="docs-table">
          <thead><tr><th>lane</th><th>effect</th></tr></thead>
          <tbody>
            <tr><td><code>!msg</code></td><td>INTERRUPT — kill A mid-turn; next turn opens with a reconcile brief</td></tr>
            <tr><td><code>msg</code></td><td>COALESCE — fold into the next turn as a digest</td></tr>
            <tr><td><code>+msg</code></td><td>APPEND — own turn, after current work</td></tr>
            <tr><td><code>todo: msg</code></td><td>CAPTURE — no turn; to the todo board (also <code>?msg</code>)</td></tr>
          </tbody>
        </table>
        <CopyBlock title="run the frontier seat" code={`abstract-claude mct              # default workspace
abstract-claude mct --model claude-opus-4-8`}
        />
      </section>

      <section id="ts-station" className="docs-sec">
        <h2 className="docs-h2">Station integration</h2>
        <p>
          <a href="#/station">hugpy Station</a>&rsquo;s frontier seat runs on abstract-claude: the
          <code>mct</code> backend is <code>abstract-claude mct</code> and <code>claude-code</code> is{' '}
          plain <code>claude</code> — one implementation, no bundled copy to drift. The retired
          <code>hugpy-agent mct</code> broker was removed from the keeper stack.
        </p>

        <div id="ts-seats" className="docs-h3-block">
          <h3 className="docs-h3">Seat provisioning</h3>
          <p>
            A fresh box has none of the seat CLIs, so only the shell surface would show. On install
            the station runs an ordered, idempotent bootstrap (<code>seat-provision.sh</code>) that
            makes frontier + local operational and authed:
          </p>
          <ol>
            <li>python3 present? else install it</li>
            <li>pip present? else <code>ensurepip</code> / apt</li>
            <li>venv active/exists? else <code>python3 -m venv</code></li>
            <li>pip upgrade needed? else <code>pip install -U pip</code></li>
            <li>install <code>abstract-claude</code>, <code>hugpy-agent</code>, <code>claude</code> → <code>~/.local/bin</code></li>
            <li>store the durable token in both the abstract-claude store and{' '}
              <code>~/.claude-oauth.env</code> (the keeper sources it) — same login as the toolserver</li>
          </ol>
        </div>

        <div id="ts-install" className="docs-h3-block">
          <h3 className="docs-h3">Secure install</h3>
          <p>
            The station is served from <code>dev.hugpy.ai/station/</code>, gated LAN/WireGuard-open
            with an off-LAN key. The one-liner fetches the sha256-verified latest build and installs
            it; pass the durable token once so the seats come up authed.
          </p>
          <CopyBlock
            title="install (LAN)"
            code={`# plain install (provisions seats in the background):
curl -fsSL https://dev.hugpy.ai/station/install.sh | sudo bash

# with the durable token → seats authed, no /login:
curl -fsSL https://dev.hugpy.ai/station/install.sh \\
  | sudo HUGPY_STATION_OAUTH=sk-ant-oat01-... bash

# off-LAN needs the installer key:
curl -fsSL -u installer:$KEY https://dev.hugpy.ai/station/install.sh | sudo -E bash`}
          />
        </div>
      </section>

      <section id="ts-dbarm" className="docs-sec">
        <h2 className="docs-h2">The DB arm</h2>
        <p>
          Beyond tools, the toolserver is becoming <strong>hugpy&rsquo;s central persistence
          layer</strong>: it owns the Postgres schemas, migrations, and read/aggregate API,
          and the fleet&rsquo;s scattered SQLite files and per-VM boards are consolidating into
          that one database. This kills a whole class of trouble &mdash; SQLite on a virtiofs
          mount is exactly where WAL-checkpoint rewrites and reset file-times bite. Everything
          below (the board, cross-box pings, and model metrics) lives in that DB; direct
          Postgres access is fine on the LAN, so writers reach it without an API hop.
        </p>
      </section>

      <section id="ts-todo" className="docs-sec">
        <h2 className="docs-h2">Central todo board</h2>
        <p>
          The <code>todo</code> category is a DB-backed board in the dedicated <code>toolserver</code>
          {' '}database — central and cross-seat, so the console, the <code>mct</code> capture lane,
          and any session share one list instead of a per-workspace file. It mirrors the station&rsquo;s
          schema (<code>id, type, status, priority, text, note, by, ts</code>) and is a scoped write
          path, separate from the read-only <code>db</code> tools.
        </p>
        <CopyBlock
          title="use the board"
          code={`curl -s https://toolserver.hugpy.ai/todo/add  -d '{"text":"ship the docs page","type":"todo","priority":"high"}'
curl -s https://toolserver.hugpy.ai/todo/list
curl -s https://toolserver.hugpy.ai/todo/done -d '{"id":"t1"}'`}
        />
        <p>
          Rows now carry a <code>locus</code> (whose board an item is on), so one table
          holds every keeper&rsquo;s slice &mdash; see <em>Central board &amp; mirror</em> below.
        </p>
      </section>

      <section id="ts-comms" className="docs-sec">
        <h2 className="docs-h2">comms.ping — reach a busy keeper</h2>
        <p>
          A plain board item reaches a keeper only from a lull; a <code>comms.ping</code> is
          the one primitive that jumps that gate. It posts a high-priority <code>[ping]</code>
          {' '}<code>request</code> on the target&rsquo;s board (carrying <code>locus=&lt;to&gt;</code>),
          and keeper-relay delivers it on the prompt board-event lane at the next idle
          boundary &mdash; not after the quiet hold. If the target relay is dry/stopped the
          ping still lands durably on the board (the backstop), so delivery is never silently
          assumed.
        </p>
        <CopyBlock
          title="poke a keeper"
          code={`curl -s https://toolserver.hugpy.ai/comms/ping -d '{"to":"hugpy","text":"build done, please look","from_":"ae-agent","ref":"t42"}'
curl -s https://toolserver.hugpy.ai/comms/inbox -d '{"to":"hugpy"}'`}
        />
      </section>

      <section id="ts-board" className="docs-sec">
        <h2 className="docs-h2">Central board &amp; mirror</h2>
        <p>
          Each keeper&rsquo;s board is still the atomic <code>~/todo.json</code> file it always
          was &mdash; that stays local truth, untouched. A per-VM <strong>board-mirror</strong>
          {' '}watchdog reflects it into a JSONB <code>board</code> table (indexed
          <code> locus / item_id / status / type</code>, full item in <code>data</code>), so a
          single agent gets one SQL view across every locus. The mirror is read-only on the
          file and <strong>fail-open</strong> on the DB: no driver, no reachability, or a
          half-written file just idles &mdash; the board keeps working and nothing on the
          delivery path depends on it. Delivery timing itself is now driven by the keeper being
          <em> between tasks</em> (its own task/status attribution), not a clock.
        </p>
      </section>

      <section id="ts-metrics" className="docs-sec">
        <h2 className="docs-h2">Model metrics</h2>
        <p>
          Central records a measured row on every real load and call &mdash; upload time and
          tok/s per (model, variant, worker, temperature), a per-model and per-(model, task)
          call EMA, and a durable compute-action log. These are moving off the SQLite file into
          the toolserver&rsquo;s Postgres (the <code>metrics</code> category), a
          behaviour-identical port (same EMA, same tables) that keeps the <em>metrics must never
          break serving</em> rule: every write is best-effort and a DB outage silently no-ops
          rather than touching the serving path.
        </p>
        <CopyBlock
          title="read the metrics"
          code={`curl -s https://toolserver.hugpy.ai/metrics/actions       -d '{"limit":50,"action":"call"}'
curl -s https://toolserver.hugpy.ai/metrics/calls_by_task
curl -s https://toolserver.hugpy.ai/metrics/call          -d '{"model":"Qwen2.5-VL-7B-Instruct-GGUF"}'`}
        />
      </section>
    </article>
  )
}
