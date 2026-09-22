// Using the console — task pages. Every step is written against the real UI:
// the labels, chips, and pills quoted here are the exact strings the console
// renders (verified in the components on 2026-07-05, tiers-v3 semantics as
// released in 0.1.133). If the UI changes, change these pages in the same PR.
import { CopyBlock } from './docParts'

/* =====================================================================
   PAGE · Chat with a model
   ===================================================================== */
export const CHAT_TOC = [
  ['pick', 'Pick a model & open chat', []],
  ['talk', 'Send, stream, stop', []],
  ['attach', 'Attach images & files', []],
  ['commands', 'Slash commands', []],
  ['allocation', 'See who answered', []],
]

export function ConsoleChatPage() {
  return (
    <>
      <div className="docs-crumbs">Using the console <span>/</span> Chat with a model</div>
      <h1>Chat with a model</h1>
      <p className="docs-lede">
        Goal: hold a streaming conversation with any model you&apos;ve downloaded — right in the console.
      </p>

      <section id="pick" className="docs-sec">
        <h2>Pick a model &amp; open chat</h2>
        <ol className="docs-steps-list">
          <li>Open the <strong>Models</strong> tab. The table lists every model hugpy knows about; by
            default it&apos;s filtered to status <em>installed</em> (models whose files are on disk).</li>
          <li>Click a model&apos;s row number (the <code>#</code> column) to expand its detail row.
            <br /><em>You should see</em> an action strip starting with <strong>💬 Chat</strong>.</li>
          <li>Click <strong>💬 Chat</strong>. If the button is greyed out, hover it — it says either
            &ldquo;Install model first&rdquo; (download it) or &ldquo;Not a chat model&rdquo; (this model&apos;s
            main job isn&apos;t conversation — embeddings, speech, image generation and the like are
            called through the API instead).</li>
        </ol>
        <p>
          <em>You should see</em> a chat panel open beside the table. Its header shows the model&apos;s
          engine and task; a <strong>🖼 VL</strong> tag appears when the model can look at images —
          that ability is read from the model&apos;s <em>full</em> task list, so a chat model that also
          does vision still gets the tag.
        </p>
      </section>

      <section id="talk" className="docs-sec">
        <h2>Send, stream, stop</h2>
        <ol className="docs-steps-list">
          <li>Type in the message box and press <strong>Enter</strong> (Shift+Enter makes a newline), or
            click <strong>↑ Send</strong>.
            <br /><em>You should see</em> the reply stream in token by token, with the header showing
            <strong> ● generating…</strong>.</li>
          <li>To halt a reply mid-stream, click <strong>⏹ Stop</strong>. The partial answer is kept and
            marked with ⏹ — and the GPU actually stops working on it, wherever the request ran.</li>
        </ol>
        <p className="docs-note">
          Conversations are saved in your browser per model — they survive tab switches and reloads
          (the header shows <strong>● saved · N msgs</strong>). <strong>✕ clear</strong> wipes the
          current conversation. Answers run unbounded by default: the model keeps going until it is
          actually done, not until a token cap.
        </p>
      </section>

      <section id="attach" className="docs-sec">
        <h2>Attach images &amp; files</h2>
        <ol className="docs-steps-list">
          <li>Click <strong>📎</strong> and pick a file.</li>
          <li>Images go straight to the model with your next message — this needs a model with the
            <strong> 🖼 VL</strong> tag, and the model must be served by an engine that can see (see the
            <a className="docs-inline-link" href="#worker-troubleshooting/mmproj"> vision fix-it entry</a> if
            image questions come back blind).</li>
          <li>Other files (text, code, …) are uploaded and their contents are folded into the
            conversation server-side.</li>
        </ol>
      </section>

      <section id="commands" className="docs-sec">
        <h2>Slash commands</h2>
        <table className="docs-table">
          <thead><tr><th>Command</th><th>Does</th></tr></thead>
          <tbody>
            <tr><td><code>/system &lt;text&gt;</code></td><td>Sets a system prompt for the rest of the conversation (the header shows <em>sys</em>).</td></tr>
            <tr><td><code>/clear</code></td><td>Clears the conversation.</td></tr>
            <tr><td><code>/tokens &lt;N&gt;</code></td><td>Caps replies at N tokens (default is unbounded with auto-continue).</td></tr>
          </tbody>
        </table>
      </section>

      <section id="allocation" className="docs-sec">
        <h2>See who answered</h2>
        <p>
          The <strong>allocation</strong> banner under the chat header names the machine that served
          your last message: a worker&apos;s name, or <strong>local (this node)</strong> when the central
          server answered itself. If you placed a model on a GPU box but the banner keeps saying
          local, the worker isn&apos;t serving it yet — check its state pills in the
          <strong> Compute</strong> tab (<a className="docs-inline-link" href="#console-glossary">Reading
          the panels</a>).
        </p>
      </section>

      <div className="docs-pager">
        <a className="docs-pager-card" href="#console-models"><span className="d">Next</span><span className="t">Find &amp; download models →</span></a>
      </div>
    </>
  )
}

/* =====================================================================
   PAGE · Find & download models
   ===================================================================== */
export const MODELS_TOC = [
  ['search', 'Search Hugging Face', []],
  ['options', 'Choose what to install', []],
  ['watch', 'Watch the download', []],
  ['vision', 'Vision models: the sidecar', []],
  ['civitai', 'Image checkpoints (Civitai)', []],
  ['manage', 'Manage what you have', []],
]

export function ConsoleModelsPage() {
  return (
    <>
      <div className="docs-crumbs">Using the console <span>/</span> Find &amp; download models</div>
      <h1>Find &amp; download models</h1>
      <p className="docs-lede">
        Goal: search Hugging Face (or Civitai), pick a variant that fits your disk, and download it
        into hugpy&apos;s model library.
      </p>

      <section id="search" className="docs-sec">
        <h2>Search Hugging Face</h2>
        <ol className="docs-steps-list">
          <li>Open the <strong>Add models 🤗</strong> tab. The source toggle at the top offers
            <strong> 🤗 Hugging Face</strong> and <strong>🧩 Civitai</strong> — stay on Hugging Face for
            language, vision, and speech models.</li>
          <li>Pick a task from the task dropdown (it starts on <em>text-generation</em>) and optionally
            type a name filter. Results appear as you type.
            <br /><em>You should see</em> a table of repos with size, download counts, and publish date —
            click any column header to sort.</li>
        </ol>
      </section>

      <section id="options" className="docs-sec">
        <h2>Choose what to install</h2>
        <ol className="docs-steps-list">
          <li>Click <strong>⬇ Options</strong> (or the ▸ arrow) on a result.
            <br /><em>You should see</em> the repo&apos;s install options — for GGUF repos that&apos;s one radio
            button per quantization (a smaller, compressed copy of the model; smaller files need less
            memory but lose a little quality), with a recommended one preselected.</li>
          <li>Options marked <em>won&apos;t fit</em> are disabled — they are bigger than your free disk.</li>
          <li>Click <strong>⬇ Add to local</strong>.</li>
        </ol>
      </section>

      <section id="watch" className="docs-sec">
        <h2>Watch the download</h2>
        <p>
          A progress bar appears in place of the button: percent, bytes, and speed. Downloads survive
          hiccups — a stalled transfer shows <strong>⚠ stalled — resuming…</strong> and retries on its
          own; <strong>✕ cancel</strong> stops it and <strong>↻ retry</strong> resumes from where it
          stopped. When it finishes you&apos;ll see <strong>✓ installed</strong>, and the model appears in
          the <strong>Models</strong> tab with a <strong>✓ ready</strong> badge.
        </p>
      </section>

      <section id="vision" className="docs-sec">
        <h2>Vision models: the sidecar</h2>
        <p>
          A vision-capable GGUF is a <strong>pair</strong> of files: the model itself plus a small
          projector file (<code>mmproj-*.gguf</code>) that translates images for it. Some downloads
          fetch only the model — it will then show <em>files incomplete</em> and refuse to go to
          workers. The fix is one file copy; see{' '}
          <a className="docs-inline-link" href="#worker-troubleshooting/mmproj">Vision GGUFs — missing
          mmproj</a>.
        </p>
      </section>

      <section id="civitai" className="docs-sec">
        <h2>Image checkpoints (Civitai)</h2>
        <ol className="docs-steps-list">
          <li>Flip the source toggle to <strong>🧩 Civitai</strong> — the home of Stable Diffusion
            checkpoints.</li>
          <li>Search, filter by base model (SD 1.5 / SDXL 1.0 / SD 2.1), and click
            <strong> ⬇ → comfy</strong>.
            <br /><em>You should see</em> a percentage while it streams in, then
            <strong> ✓ registered</strong> — the checkpoint lands in the shared
            <code> /checkpoints</code> folder and registers itself as a <code>comfy-…</code> model
            automatically. What to do with it next:{' '}
            <a className="docs-inline-link" href="#console-images">Generate images</a>.</li>
        </ol>
        <p className="docs-note">
          A ⚠ next to a result means its base model is outside the family the standard image
          template serves (SD 1.x / 2.x / SDXL) — it may not load.
        </p>
      </section>

      <section id="manage" className="docs-sec">
        <h2>Manage what you have</h2>
        <ul className="docs-routes">
          <li><strong>Filters</strong> — the Models tab&apos;s search matches name, repo, task, and engine;
            the task dropdown matches <em>every</em> task a model advertises, not just its main one, so
            multi-talent models show up under each thing they can do.</li>
          <li><strong>Badges</strong> — <strong>✓ ready</strong> (files complete), <strong>◐ partial</strong>
            (incomplete — expand the row and <strong>⬇ Resume download</strong>), <strong>✗ missing</strong>
            (registered but not downloaded).</li>
          <li><strong>Free disk</strong> — expand a row for <strong>🗑 Delete files</strong> (keeps the
            registry entry) or <strong>⊘ Prune entry</strong> (removes a never-downloaded ghost).</li>
          <li><strong>Discover models</strong> — re-scans model storage and re-registers anything on disk
            that fell out of the list. Slower than the ↻ reload; use it after moving files around
            outside the console.</li>
        </ul>
      </section>

      <div className="docs-pager">
        <a className="docs-pager-card" href="#console-workers"><span className="d">Next</span><span className="t">Put models on workers →</span></a>
      </div>
    </>
  )
}

/* =====================================================================
   PAGE · Put models on workers
   ===================================================================== */
export const WORKERS_TOC = [
  ['join', 'Add a worker box', []],
  ['admit', 'Admit it', []],
  ['load', 'Load models onto one worker', []],
  ['group', 'Put one model on many workers', []],
  ['verify', 'Confirm it took', []],
  ['space', 'Take models off & free space', []],
]

export function ConsoleWorkersPage() {
  return (
    <>
      <div className="docs-crumbs">Using the console <span>/</span> Put models on workers</div>
      <h1>Put models on workers</h1>
      <p className="docs-lede">
        Goal: join a GPU box to your fleet and decide which models live on it — the console guards
        the fit so you can&apos;t overfill a machine by accident.
      </p>

      <section id="join" className="docs-sec">
        <h2>Add a worker box</h2>
        <ol className="docs-steps-list">
          <li>Open the <strong>Compute</strong> tab and expand
            <strong> ＋ Add &amp; manage workers</strong>.</li>
          <li>Copy the one-line install command (<code>curl -fsSL …/install.sh | bash</code>) and run it
            on the new box. Central bakes its own reachable address into the script — no per-worker
            configuration.
            <br /><em>You should see</em> — if a ⚠ warning appears saying the address is
            <code> localhost</code>, you&apos;re browsing the console on central itself; use the LAN
            address the warning suggests so the worker box can actually reach central.</li>
          <li><em>Optional, for locked-down fleets:</em> issue an enrollment token first
            (<strong>+ Issue enrollment token</strong>). The token is shown <strong>once</strong> — copy
            the combined command it gives you. Revoking a token later shuts down every worker that
            joined with it.</li>
        </ol>
      </section>

      <section id="admit" className="docs-sec">
        <h2>Admit it</h2>
        <p>
          A newly joined box appears as a row with a <strong>pending</strong> pill — it does not serve
          anything yet; admission is your approval gate. Click <strong>✓ admit</strong> to let it serve,
          or <strong>⛔ block</strong> to evict a box for good (its agent exits and won&apos;t respawn).
          <em> You should see</em> the pill turn <strong>approved</strong> and a
          <strong> ⏳ N pending</strong> chip disappear from the panel header.
        </p>
      </section>

      <section id="load" className="docs-sec">
        <h2>Load models onto one worker</h2>
        <ol className="docs-steps-list">
          <li>On the worker&apos;s row, click <strong>＋ load a model</strong>.
            <br /><em>You should see</em> a sortable table of every model not already on this worker,
            with a search box, a task dropdown, and an
            <strong> ⚡ allow duplicate allocation</strong> checkbox.</li>
          <li>Filter if you like — the task dropdown matches <em>any</em> task a model advertises, not
            just its main one.</li>
          <li>Tick the models you want. Models that already live on another worker are locked and show
            <em> on &lt;worker&gt;</em> in the &ldquo;Allocated on&rdquo; column — that&apos;s the
            anti-duplicate guard. Flip the <strong>⚡</strong> breaker only when you deliberately want a
            second copy (for example to fan work out across boxes).</li>
          <li>Click <strong>Allocate N models to this worker</strong>. Every model is
            <em> fit-guarded</em>: checked against the worker&apos;s free VRAM (GPU memory), RAM, and disk
            before anything moves — a model that won&apos;t fit is refused with the reason, not loaded
            into a crash. You can force past a refusal, at your own OOM risk.</li>
        </ol>
        <p className="docs-note">
          Placing a model on a worker is called a <em>designation</em> in some messages: the model is
          assigned to that box, its files are pulled over (that transfer is
          <em> provisioning</em> — you&apos;ll see <strong>⏳ pulling</strong> with a percentage), and it
          warms up ready to serve.
        </p>
      </section>

      <section id="group" className="docs-sec">
        <h2>Put one model on many workers</h2>
        <ol className="docs-steps-list">
          <li>Above the worker rows, click
            <strong> ⧉ Assign a model to a group of workers…</strong></li>
          <li>Pick the model, then tick target workers (or <strong>Select all eligible</strong>).
            Workers that already hold it show <strong>✓ already here</strong> and are never re-sent —
            no double-booking; offline, blocked, and not-yet-admitted boxes are locked out too.</li>
          <li>Click <strong>Dedicate to N workers</strong>.
            <br /><em>You should see</em> a per-worker result list — ✓ or ✗ with the refusal reason.
            Re-running retries only the failures.</li>
        </ol>
      </section>

      <section id="verify" className="docs-sec">
        <h2>Confirm it took</h2>
        <p>
          On the worker&apos;s row, the model appears in its <strong>Serving:</strong> list with a state
          pill that tells the live truth: <strong>⏳ pulling 42%</strong> (files transferring) →
          <strong> 🔶 heating</strong> (weights loading into memory) → <strong>🔥 serving</strong> or
          <strong> 📌 loaded</strong> (ready). Then open a chat with the model — the allocation banner
          should name this worker. Every pill is decoded in{' '}
          <a className="docs-inline-link" href="#console-glossary">Reading the panels</a>.
        </p>
      </section>

      <section id="space" className="docs-sec">
        <h2>Take models off &amp; free space</h2>
        <ul className="docs-routes">
          <li><strong>⏏</strong> on a model — unload it from memory. It stays assigned and reloads on
            the next request. <strong>⏏ free VRAM</strong> in the row header unloads everything.</li>
          <li><strong>×</strong> on a model — unassign it from this worker entirely. Refused while the
            model is 📌 pinned (<a className="docs-inline-link" href="#console-serving/pin">what pinning
            means</a>) — unpin first.</li>
          <li><strong>🗑 reclaim</strong> next to the disk chip — deletes files of models that are on
            disk but no longer assigned. It previews the list and never touches anything assigned,
            loaded, or 📌 pinned.</li>
        </ul>

        <h3>The 💾 storage bar</h3>
        <p>
          Under the resource-pool bar, each worker shows a <strong>💾 storage</strong> bar — how much
          model cache the box holds on its model-root disk (<em>e.g. &ldquo;142 / 180 GB cached · 12
          models&rdquo;</em>), with the reserved free space marked. Below it is the per-model list:
          each model&apos;s size, when it was <em>last served</em>, and a badge. <strong>Protected</strong>
          models — 🔥 loaded / serving, 📌 pinned, 🔒 static, or currently assigned — are greyed and can
          never be deleted. Cold, unassigned files show as <strong>○ evictable</strong>.
        </p>
        <p>
          If a worker crosses its storage budget (its free disk drops below the reserve, or an explicit
          per-worker cap is exceeded) it shows a <strong>⚠ over budget</strong> flag and an
          <strong> eviction proposal</strong>: exactly which cold, unprotected models would be freed to
          get back under budget, least-recently-served first. <strong>Nothing is deleted automatically.</strong>
          The proposal is a preview until you click <strong>✓ Approve &amp; free</strong>; on approval,
          central re-checks the list and the worker re-proves every guard before deleting a single file,
          so a model that became loaded, assigned, or pinned in the meantime is skipped (and shown as
          skipped). Protected files are never touched.
        </p>
      </section>

      <div className="docs-pager">
        <a className="docs-pager-card" href="#console-serving"><span className="d">Next</span><span className="t">Keep models serving →</span></a>
      </div>
    </>
  )
}

/* =====================================================================
   PAGE · Keep models serving (tiers v3, as released in 0.1.133)
   ===================================================================== */
export const SERVING_TOC = [
  ['residency', 'The two policies', []],
  ['pin', '📌 Pin: make it permanent', []],
  ['pinall', 'Pin all · Unpin all', []],
  ['activate', '▶ Activate: serve it now', []],
  ['apply', 'The ⏳ applying window', []],
  ['pills', 'Policy vs. live state', []],
  ['allocations', 'What’s allocated', []],
  ['budget', 'The resource-pool bar', []],
  ['slots', 'Slots: the seats', []],
]

export function ConsoleServingPage() {
  return (
    <>
      <div className="docs-crumbs">Using the console <span>/</span> Keep models serving</div>
      <h1>Keep models serving</h1>
      <p className="docs-lede">
        Goal: decide which models a worker keeps loaded and ready versus loads on demand — and make
        a model&apos;s home on a worker permanent when you mean it.
      </p>

      <section id="residency" className="docs-sec">
        <h2>The two policies</h2>
        <p>
          Every model on a worker has a <em>residency</em> policy — how it competes for the worker&apos;s
          memory. There are exactly two:
        </p>
        <table className="docs-table">
          <thead><tr><th>Policy</th><th>Meaning</th></tr></thead>
          <tbody>
            <tr><td><strong>⏲ on-demand</strong> <em>(the default)</em></td><td>Loads when called; once loaded it <strong>holds its seat until another model needs it</strong>. Seats never sit idle on purpose — the least-recently-used on-demand occupant is the one bumped when a different model gets called.</td></tr>
            <tr><td><strong>🔒 static</strong></td><td>A locked seat: never swapped out, no matter who else gets called. Lasts as long as the model stays assigned to the worker (unassigning clears it) — combine with 📌 pin to make it permanent.</td></tr>
          </tbody>
        </table>
        <ol className="docs-steps-list">
          <li>In the <strong>Compute</strong> tab, find the model on its worker row and click its
            residency tag (<strong>⏲ on-demand</strong> or <strong>🔒 static</strong>).
            <br /><em>You should see</em> a two-option picker open — clicking the tag never changes
            anything by itself.</li>
          <li>Pick the other option. Picking the current one just closes the menu, so every change is
            a deliberate act.</li>
        </ol>
        <p className="docs-note">
          One honest nuance: a model resident in the worker&apos;s own process (the <strong>📌 loaded</strong>
          pill, not a slot) that is on-demand also ages out after ~15 idle minutes to give its memory
          back. Slot seats don&apos;t age out — they only change hands when another model needs the seat.
        </p>
      </section>

      <section id="pin" className="docs-sec">
        <h2>📌 Pin: make the allocation survive restarts</h2>
        <p>
          The <strong>📌</strong> button next to a model makes its <em>allocation survive restarts</em>.
          An allocation is <em>routing</em> — which worker answers for that model. Pin makes that routing
          durable and nothing else. While pinned:
        </p>
        <ul className="docs-routes">
          <li><strong>Unassign is refused</strong> — the × button is disabled, and the API answers
            <em> &ldquo;pinned to &lt;worker&gt; — unpin first&rdquo;</em>.</li>
          <li><strong>The allocation and residency choice survive</strong> restarts and prunes —
            everything short of unpinning.</li>
          <li><strong>Files are NOT protected.</strong> Pin does not download the model and does not keep
            its bytes on disk — a pinned model can be evicted to make room, and its weights re-download on
            the next call. The pin (routing) is untouched by the eviction.</li>
        </ul>
        <p>
          Want the files always present too? Use <strong>🔒 static</strong> — the only tier that downloads
          the model eagerly and never evicts it. <strong>🔒 static + 📌 pin</strong> gives you a locked
          seat that is always on disk, on a worker it can never silently leave — the setting for the one
          model you always want answering instantly. An unpinned 📌 shows as <strong>📌?</strong> — click
          to pin, click again to unpin.
        </p>
      </section>

      <section id="pinall" className="docs-sec">
        <h2>Pin all · Unpin all</h2>
        <p>
          When you want a whole worker&apos;s line-up to stay put — every model designated to it made
          permanent in one go — use the <strong>📌 pin all</strong> button next to the worker&apos;s
          <strong> Serving:</strong> list. It pins every model currently on that worker in a single
          change (one agent restart, not one per model).
        </p>
        <ol className="docs-steps-list">
          <li>Click <strong>📌 pin all</strong> on the worker row. <em>You should see</em> a confirm
            dialog telling you how many models will be pinned — because pinning is sticky: each one then
            refuses unassign (&ldquo;unpin first&rdquo;) until you reverse it.</li>
          <li>Confirm. The worker shows <strong>⏳ applying…</strong> for a few seconds; when it comes
            back, each model carries a solid <strong>📌</strong>. If any model couldn&apos;t be pinned,
            a summary lists which and why — the rest still pinned.</li>
        </ol>
        <p>
          <strong>Unpin all</strong> is the undo: <strong>📌✕ unpin all</strong> clears the pin on every
          model on that worker, so they can be unassigned or reaped again. Use it when you&apos;re about
          to reshuffle a worker you&apos;d previously locked down.
        </p>
      </section>

      <section id="activate" className="docs-sec">
        <h2>▶ Activate: serve it now</h2>
        <p>
          A model shown <strong>○ cold</strong> is assigned but not loaded — it loads on the first
          request (which pays the wait). To make it serve <em>right now</em> instead, click
          <strong> ▶ activate</strong> on its row. Activating <em>allocates the worker&apos;s resources
          to the model</em> so it becomes resident and ready — engine-agnostic: on a slotted worker a
          GGUF model takes a slot; a transformers model loads into the worker&apos;s RAM. Either way the
          goal is the same — no cold-start on the next call.
        </p>
        <ol className="docs-steps-list">
          <li>Find a <strong>○ cold</strong> (or <strong>✗ missing</strong>) model on a worker row and
            click <strong>▶ activate</strong>. It shows <strong>⏳ activating…</strong> while the worker
            seats it — a cold load can take tens of seconds, and if the model&apos;s files aren&apos;t
            local yet they transfer first.</li>
          <li>Watch the state pill: it moves through <strong>⏳ pulling</strong> / <strong>🔶 heating</strong>
            to <strong>🔥 serving</strong> or <strong>📌 loaded</strong>. The card reflects it on the next
            heartbeat — it keeps working in the background even if the button already returned.</li>
        </ol>
        <p className="docs-note">
          If the worker can&apos;t fit the model, the server says so (and offers to force it); nothing is
          hidden client-side. Activate is available on every worker regardless of engine — a slot is just
          a resource allocation to a model.
        </p>
      </section>

      <section id="apply" className="docs-sec">
        <h2>The ⏳ applying window</h2>
        <p>
          Residency, pin, and slot-count changes are stored on the worker itself, and the worker&apos;s
          agent restarts to apply them. After a change, the row shows
          <strong> ⏳ applying…</strong> and that worker&apos;s policy controls grey out for up to ~10
          seconds while the restart happens and the next heartbeat confirms the new config. If a
          click during the window fails, it isn&apos;t broken — retry in a few seconds.
        </p>
      </section>

      <section id="pills" className="docs-sec">
        <h2>Policy vs. live state</h2>
        <p>
          The residency tag is a <em>policy</em> — what you asked for. The state pills are the
          <em> live truth</em> — what is happening right now:
        </p>
        <table className="docs-table">
          <thead><tr><th>Pill</th><th>Live state</th></tr></thead>
          <tbody>
            <tr><td><strong>🔥 serving</strong></td><td>In a slot, healthy, routable — requests reach it instantly.</td></tr>
            <tr><td><strong>⚡ answering</strong></td><td>Its slot is processing a request <em>right now</em>.</td></tr>
            <tr><td><strong>📌 loaded</strong></td><td>Resident in the worker&apos;s own process (ready, but not in a slot).</td></tr>
            <tr><td><strong>○ cold</strong></td><td>Assigned; loads on the first request or the next warm pass.</td></tr>
            <tr><td><strong>⏳ pulling 42%</strong></td><td>Files transferring to the worker, with live progress.</td></tr>
            <tr><td><strong>🔶 heating</strong></td><td>Weights loading into memory right now.</td></tr>
            <tr><td><strong>✗ missing</strong></td><td>Assigned but files absent — the worker re-pulls on its own; if it never resolves, see <a className="docs-inline-link" href="#worker-troubleshooting/missing-files">the fix-it entry</a>.</td></tr>
          </tbody>
        </table>
        <p className="docs-note">
          &ldquo;Serving&rdquo; is only ever a state, never a policy — no setting is called serving. An
          on-demand model can be 🔥 serving for days if nothing else wants its seat.
        </p>
      </section>

      <section id="allocations" className="docs-sec">
        <h2>What&apos;s allocated</h2>
        <p>
          An <em>allocation</em> is simply the worker&apos;s resources given to a model so it&apos;s
          resident and serving — that&apos;s all a &ldquo;slot&rdquo; is, regardless of engine. So the
          worker card shows every allocation in one honest picture: a GGUF model holding a slot seat and
          a transformers model held in the worker&apos;s RAM appear side by side, each with a subtle
          badge — <strong>🎰 slot</strong> or <strong>🧠 RAM</strong> — next to its name. A worker with
          no seats configured (for example a CPU-only, transformers-only box) shows its RAM-resident
          models as allocations and no empty slot rows.
        </p>
        <p>
          <strong>Which number means &ldquo;occupied&rdquo;.</strong> A model&apos;s{' '}
          <em>model_bytes</em> is its size <em>on disk</em> — an upper bound on what it could ever
          take, never a claim about what it is holding right now. Occupancy is only ever a worker-side
          measurement: <em>ram_resident_bytes</em> (an in-process model&apos;s measured host RAM),{' '}
          <em>rss_anon</em> (a slot child&apos;s truly-pinned RAM — plain VmRSS also counts the
          mmap&apos;d model file&apos;s reclaimable pages and overstates RAM many times over), and{' '}
          <em>vram_bytes</em> (the per-process nvidia-smi read). Where a figure is prefixed{' '}
          <strong>~</strong> and shown muted, it is an <em>estimate</em> derived from the declared
          placement times the file size — kept because on a box whose nvidia-smi is broken it is the
          only signal there is, but never summed together with measurements.
        </p>
      </section>

      <section id="budget" className="docs-sec">
        <h2>The resource-pool bar</h2>
        <p>
          A worker has <em>one</em> resource pool — its VRAM plus RAM, minus a small reserve the box
          keeps for itself — drawn as a single budget bar on the worker card. Resident models fill the
          bar; the empty tail is what&apos;s still free. There is no fixed number of
          &ldquo;slots&rdquo;: however many models fit in the pool right now <em>are</em> the slots, so
          the count rises and falls as models load and unload.
        </p>
        <p>Read the bar this way:</p>
        <ul className="docs-routes">
          <li>Each <strong>resident chip</strong> shows its <em>footprint</em> (how much of the pool it
            holds), its <em>engine kind</em> — <strong>🎰 slot</strong> or <strong>🧠 RAM</strong> —
            and its <em>live state</em> (<strong>🔥 serving</strong> / <strong>⚡ answering</strong> /
            <strong> 🔶 heating</strong>, decoded in{' '}
            <a className="docs-inline-link" href="#console-serving/pills">Policy vs. live state</a>).</li>
          <li><strong>0 resident</strong> means the pool is simply <em>idle</em> — nothing is loaded, or
            nothing currently fits. It is not a broken or offline worker; the next request (or an
            <a className="docs-inline-link" href="#console-serving/activate"> ▶ activate</a>) fills it.</li>
        </ul>
        <p className="docs-note">
          In this release the budget bar is <strong>read-only</strong> — it shows how the pool is spent,
          but you can&apos;t resize the pool or hand-evict a chip from it here yet. Tuning the reserve
          and manual eviction arrive in a later slice.
        </p>
      </section>

      <section id="slots" className="docs-sec">
        <h2>Slots: the seats</h2>
        <p>
          A <em>slot</em> is a seat that hosts one running model in its own crash-isolated process —
          a model in a slot can be routed to directly and can&apos;t take the worker down with it. The
          worker&apos;s <strong>Slots:</strong> row shows each seat: <strong>🔥 serving</strong> /
          <strong> ⚡ answering</strong> / <strong>⏳ warming</strong> with the occupant&apos;s name, or
          <strong> ○ slot N</strong> when empty. Empty seats fill themselves from the worker&apos;s
          assigned models (static ones first). The <strong>🎛 slots: N</strong> chip sets how many
          seats the box runs (0–16; 0 = in-process only) — changing it goes through the same
          ⏳ applying restart.
        </p>
      </section>

      <div className="docs-pager">
        <a className="docs-pager-card" href="#console-images"><span className="d">Next</span><span className="t">Generate images →</span></a>
      </div>
    </>
  )
}

/* =====================================================================
   PAGE · Generate images
   ===================================================================== */
export const IMAGES_TOC = [
  ['kinds', 'Two kinds of image models', []],
  ['comfy', 'Checkpoint path (comfy-*)', []],
  ['diffusers', 'Diffusers path (sd / transformers)', []],
  ['media', 'The media arm', []],
  ['no-worker', '“no live GPU worker is serving …”', []],
]

export function ConsoleImagesPage() {
  return (
    <>
      <div className="docs-crumbs">Using the console <span>/</span> Generate images</div>
      <h1>Generate images</h1>
      <p className="docs-lede">
        Goal: get an image model running on one of your GPU workers and understand the two routes an
        image can take — and the one error everyone hits first.
      </p>

      <section id="kinds" className="docs-sec">
        <h2>Two kinds of image models</h2>
        <table className="docs-table">
          <thead><tr><th>Kind</th><th>Where it comes from</th><th>What runs it</th></tr></thead>
          <tbody>
            <tr><td><strong>comfy checkpoints</strong> (<code>comfy-…</code>)</td><td>Civitai (<a className="docs-inline-link" href="#console-models/civitai">⬇ → comfy</a>) or your own <code>/checkpoints</code> folder</td><td>A ComfyUI install on a worker box — the worker with the <strong>🧩 comfy</strong> chip</td></tr>
            <tr><td><strong>diffusers models</strong> (Stable Diffusion, FLUX, …)</td><td>Hugging Face, like any other model</td><td>A GPU worker the model is placed on, like a chat model</td></tr>
          </tbody>
        </table>
      </section>

      <section id="comfy" className="docs-sec">
        <h2>Checkpoint path (comfy-*)</h2>
        <ol className="docs-steps-list">
          <li>Pull a checkpoint from Civitai — it self-registers as a <code>comfy-…</code> model
            (<a className="docs-inline-link" href="#console-models/civitai">how</a>).</li>
          <li>Check the <strong>Compute</strong> tab: the worker that will serve it must show the
            <strong> 🧩 comfy</strong> chip, which means a ComfyUI process is alive on that box. hugpy
            never starts ComfyUI itself — it adopts one that is running.</li>
          <li>If every <code>comfy-…</code> request fails with <em>Connection refused</em>, ComfyUI is
            down on that box — see{' '}
            <a className="docs-inline-link" href="#worker-troubleshooting/comfy-refused">the fix-it
            entry</a> to start (or install) it.</li>
        </ol>
      </section>

      <section id="diffusers" className="docs-sec">
        <h2>Diffusers path (sd / transformers)</h2>
        <ol className="docs-steps-list">
          <li>Download the model from Hugging Face as usual
            (task <em>text-to-image</em> or <em>image-to-image</em> in the Add models tab).</li>
          <li>Place it on a GPU worker
            (<a className="docs-inline-link" href="#console-workers/load">Put models on workers</a>) and
            wait for its pill to reach a ready state.</li>
          <li>Rough edge, honestly: these models need heavyweight Python packages (diffusers, torch,
            sometimes bitsandbytes for compressed checkpoints) installed <em>on the worker</em>. A
            worker set up for chat models will fail image loads with import errors — see{' '}
            <a className="docs-inline-link" href="#worker-troubleshooting/no-torch">the missing-torch
            fix-it entry</a> for installing extras into a worker from the console side.</li>
        </ol>
      </section>

      <section id="media" className="docs-sec">
        <h2>The media arm</h2>
        <p>
          Image and scene generation is driven through hugpy&apos;s media endpoints
          (<code>/api/video/jobs/generate_image</code>, <code>…/generate_scene</code>) — that&apos;s what
          the media-intelligence chat and any API caller use; scenes chain frames into video. The
          console&apos;s part is upstream: the <strong>Media</strong> column in the Models tab. Tick a
          chat-capable model&apos;s checkbox to offer it in the media chat&apos;s model dropdown, and use the
          ★ to make one the default.
        </p>
      </section>

      <section id="no-worker" className="docs-sec">
        <h2>&ldquo;no live GPU worker is serving …&rdquo;</h2>
        <p>
          The most common image-generation refusal reads:
          <em> &ldquo;no live GPU worker is serving &lsquo;X&rsquo; (worker offline or still warming);
          refusing local CPU generation on central.&rdquo;</em>
        </p>
        <p>
          It means: you have a worker fleet, but no online worker currently serves this image model —
          and hugpy deliberately refuses to grind out the image on central&apos;s CPU (a multi-gigabyte
          model on the API server melts it for everyone). It is a <em>retryable</em> refusal, not a
          crash. Fix it in order:
        </p>
        <ol className="docs-steps-list">
          <li>Open <strong>Compute</strong> and find the model. Not assigned anywhere? Place it on a GPU
            worker (<a className="docs-inline-link" href="#console-workers/load">how</a>).</li>
          <li>Assigned but <strong>⏳ pulling</strong> / <strong>🔶 heating</strong>? It&apos;s still warming —
            retry when the pill settles.</li>
          <li>Worker offline or <strong>✗ unreachable</strong>? See{' '}
            <a className="docs-inline-link" href="#worker-troubleshooting/unreachable">the fix-it
            entry</a>.</li>
        </ol>
        <p className="docs-note">
          Single machine, no fleet? Then local generation stays allowed — this refusal only exists
          when workers are registered. <code>HUGPY_VIDEOGEN_LOCAL=always</code> overrides it if you
          truly want central doing the work.
        </p>
      </section>

      <div className="docs-pager">
        <a className="docs-pager-card" href="#console-discord"><span className="d">Next</span><span className="t">Discord &amp; comms sessions →</span></a>
      </div>
    </>
  )
}

/* =====================================================================
   PAGE · Discord & comms sessions
   ===================================================================== */
export const DISCORD_TOC = [
  ['bindings', 'Bind a model to a channel', []],
  ['bridges', 'Supervised bridges', []],
  ['sessions', 'Mint a comms session', []],
  ['endpoints', 'What the token can do', []],
  ['revoke', 'Revoke', []],
  // Linked from the /fleet screen's per-block "Docs ↗" buttons
  // (react/agents_ui/src/FleetOverview.tsx → DOCS_HREFS). Keep bot-own and
  // bot-hugpy stable — they are the two deep anchors those buttons address.
  ['bot-links', 'Run your own bot (members)', [
    ['bot-own', 'Your own Discord bot'],
    ['bot-hugpy', 'The hugpy discord bot'],
  ]],
]

export function ConsoleDiscordPage() {
  return (
    <>
      <div className="docs-crumbs">Using the console <span>/</span> Discord &amp; comms sessions</div>
      <h1>Discord &amp; comms sessions</h1>
      <p className="docs-lede">
        Goal: put your models in Discord — from a simple model↔channel binding to a scoped token an
        outside agent can use to speak in exactly one channel.
      </p>
      <p className="docs-note">
        All of this lives in the <strong>API</strong> tab. The Discord bot itself
        (<code>hugpy bot</code>) must be running and invited to your server for messages to flow.
      </p>

      <section id="bindings" className="docs-sec">
        <h2>Bind a model to a channel</h2>
        <ol className="docs-steps-list">
          <li>Open <strong>💬 Discord — model ↔ channel/user bindings</strong>.</li>
          <li>Pick a model, then a channel and/or a user from the dropdowns (they list what the bot can
            see; you can also enter an ID manually — enable Developer Mode in Discord and right-click →
            Copy ID).</li>
          <li>Click <strong>+ Bind</strong>.
            <br /><em>You should see</em> the binding row appear: <em>🧠 model → #channel</em>.
            @-mentions in that channel (or from that user) now go to the bound model, and the row&apos;s
            compose box pushes a message outward into Discord.</li>
        </ol>
      </section>

      <section id="bridges" className="docs-sec">
        <h2>Supervised bridges</h2>
        <p>
          <strong>🔗 Console ↔ Discord bridges</strong> are bindings with you in the loop: inbound
          messages generate a draft reply per your directive, and the mode decides whether it sends
          automatically or waits for your <strong>✓ send</strong> / ✏️ edit / <strong>✕</strong> in a
          merged transcript. Use a binding for a hands-off bot, a bridge when you want approval power
          over what the model says.
        </p>
      </section>

      <section id="sessions" className="docs-sec">
        <h2>Mint a comms session</h2>
        <p>
          A comms session is a <em>scoped bearer token</em>: whoever holds it can read one Discord
          channel and post to it — nothing else on your API. It&apos;s how you give a terminal agent a
          voice in a channel without giving it your console.
        </p>
        <ol className="docs-steps-list">
          <li>Open <strong>🎟️ Comms sessions</strong>. Pick a channel, optionally a label, a lifetime in
            hours (blank = no expiry), and an instructions line (folded into the paste-block so the
            agent knows when to speak).</li>
          <li>Click <strong>+ Mint</strong>.
            <br /><em>You should see</em> a boxed paste-block with the token baked into an endpoint URL —
            shown <strong>once</strong>; the server keeps only a hash. Copy it now.</li>
          <li>Paste the block into the agent&apos;s session. It contains the endpoint and usage
            instructions in one piece.</li>
        </ol>
      </section>

      <section id="endpoints" className="docs-sec">
        <h2>What the token can do</h2>
        <table className="docs-table">
          <thead><tr><th>Call</th><th>Does</th></tr></thead>
          <tbody>
            <tr><td><code>POST &lt;endpoint&gt;/send</code></td><td>Post to the channel — JSON <code>{'{"content":"…"}'}</code>, ≤1900 chars, delivered within ~8s.</td></tr>
            <tr><td><code>GET &lt;endpoint&gt;/messages?since=&lt;ts&gt;</code></td><td>Read replies since a timestamp (poll ~30s).</td></tr>
            <tr><td><code>GET &lt;endpoint&gt;</code></td><td>A usage refresher for the agent.</td></tr>
          </tbody>
        </table>
        <p className="docs-note">
          The endpoint URL uses the console&apos;s own origin — an agent running off-network needs a host
          it can reach (tailnet or public), not a LAN address.
        </p>
      </section>

      <section id="revoke" className="docs-sec">
        <h2>Revoke</h2>
        <p>
          The sessions table shows every session — <em>live</em>, <em>expired</em>, or <em>revoked</em> —
          with creation and last-used times. Click <strong>revoke</strong> on a live one and any agent
          holding its token loses access immediately.
        </p>
      </section>

      <section id="bot-links" className="docs-sec">
        <h2>Run your own bot (members)</h2>
        <p>
          Everything above is the <em>operator&apos;s</em> bot — one bot account, bound to the operator&apos;s
          guild. A signed-in member does not get a share of it. Instead, the
          {' '}<a className="docs-inline-link" href="/fleet">Fleet screen</a> generates the credentials a bot
          you own needs to call this deployment as <strong>you</strong>.
        </p>
        <p className="docs-note">
          <strong>What is never issued.</strong> The shared <code>DISCORD_TOKEN</code> is the operator&apos;s
          bot-account credential for the whole guild — it is not handed out, ever. Neither is a comms
          session: a session is bound to a channel id the caller names, and nothing here can prove you
          control the channel behind an id, so minting one stays operator-only. You bring your own bot
          token; hugpy issues the API key.
        </p>

        <h3 id="bot-own" className="docs-h3-plain">Your own Discord bot</h3>
        <p>
          Use this when you have written (or want to write) your own bot — <code>discord.py</code>,
          <code> discord.js</code>, anything — and want it to think with this fleet.
        </p>
        <ol className="docs-steps-list">
          <li>Create a bot application at
            {' '}<a className="docs-inline-link" href="https://discord.com/developers/applications" target="_blank" rel="noreferrer">discord.com/developers/applications</a>,
            copy its bot token, and invite it to your server from that page.</li>
          <li>On <a className="docs-inline-link" href="/fleet">/fleet</a>, open
            <strong> Connect your Discord bot</strong>, label it, and press <strong>Generate</strong>.</li>
          <li>Copy the <code>.env</code> block it prints. The key is shown <strong>once</strong>.</li>
        </ol>
        <CopyBlock title=".env" code={`DISCORD_TOKEN=<your own discord bot token>
HUGPY_BASE_URL=https://your-hugpy/api
HUGPY_API_KEY=hp_REPLACE_ME
OPENAI_BASE_URL=https://your-hugpy/v1
OPENAI_API_KEY=hp_REPLACE_ME`} />
        <p>
          The same key works two ways: as <code>X-API-Key</code> against the hugpy API, and as an
          OpenAI-compatible bearer against <code>/v1</code> — so an existing OpenAI SDK bot only needs its
          base URL and key changed.
        </p>

        <h3 id="bot-hugpy" className="docs-h3-plain">The hugpy discord bot</h3>
        <p>
          Use this when you want the bot arm hugpy already ships (<code>hugpy bot</code>) — chat, model
          switching, uploads, image generation — running on your own machine and your own bot account.
          The Fleet screen&apos;s <strong>Run the hugpy discord bot</strong> block generates the
          <code> HUGPY_API_KEY</code> that instance authenticates with.
        </p>
        <CopyBlock title="install and run" code={`python3 -m venv ~/hugpy-bot/venv
~/hugpy-bot/venv/bin/pip install --upgrade abstract_hugpy

# .env next to where you run it (see the block above for the values)
~/hugpy-bot/venv/bin/hugpy bot --env ./.env`} />
        <p className="docs-note">
          The bot reads <code>DISCORD_TOKEN</code> and <code>HUGPY_API_KEY</code> from that
          <code> .env</code>; <code>HUGPY_BOT_API_KEY</code> overrides the key for this arm alone. The
          full CLI is under <a className="docs-inline-link" href="#cli/bot">Reference → CLI</a>, and every
          variable it reads is in
          {' '}<a className="docs-inline-link" href="#architecture/env">Environment variables</a>.
        </p>

        <h3 className="docs-h3-plain">Scope and revocation</h3>
        <ul className="docs-routes">
          <li><strong>Account-scoped.</strong> A member&apos;s key carries the product scopes only
            (<code>v1</code>, <code>ml</code>) — never <code>full</code>, never
            <code> agent-register</code>. An API key of any scope is never an operator credential.</li>
          <li><strong>Attributed.</strong> Your username is recorded on the key. The list on /fleet shows
            only your own keys.</li>
          <li><strong>Revocable.</strong> Revoke one from the same block (or, as an operator, from
            <strong> API access</strong>); the bot using it stops authenticating immediately.</li>
          <li><strong>Invite link.</strong> If the deployment has a Discord application id configured
            (<code>HUGPY_DISCORD_CLIENT_ID</code>), the generated block also offers a one-click OAuth
            invite. Without it the block says so and points you at your own application page.</li>
        </ul>
      </section>

      <div className="docs-pager">
        <a className="docs-pager-card" href="#console-glossary"><span className="d">Next</span><span className="t">Reading the panels →</span></a>
      </div>
    </>
  )
}

/* =====================================================================
   PAGE · Reading the panels (glossary)
   ===================================================================== */
export const GLOSSARY_TOC = [
  ['model-states', 'Model state pills (Compute)', []],
  ['model-policy', 'Policy tags & controls', []],
  ['worker-chips', 'Worker identity chips', []],
  ['resource-chips', 'Resource & config chips', []],
  ['models-tab', 'Models-tab badges', []],
  ['fixdoc', 'The 📖 link', []],
]

export function ConsoleGlossaryPage() {
  return (
    <>
      <div className="docs-crumbs">Using the console <span>/</span> Reading the panels</div>
      <h1>Reading the panels</h1>
      <p className="docs-lede">
        Every chip, pill, and badge in the console, one plain-language line each. Hovering any of
        them in the console shows the same story in its tooltip.
      </p>

      <section id="model-states" className="docs-sec">
        <h2>Model state pills (Compute tab)</h2>
        <table className="docs-table">
          <thead><tr><th>Pill</th><th>Means</th></tr></thead>
          <tbody>
            <tr><td><strong>🔥 serving</strong></td><td>Loaded in a slot and routable — answers instantly.</td></tr>
            <tr><td><strong>⚡ answering</strong></td><td>Processing a request at this very moment.</td></tr>
            <tr><td><strong>⏳ warming</strong></td><td>A slot occupant still loading — not healthy yet.</td></tr>
            <tr><td><strong>📌 loaded</strong></td><td>Resident in the worker&apos;s own process (ready, not in a slot).</td></tr>
            <tr><td><strong>○ cold</strong></td><td>Assigned to the worker; loads on first request or the next warm pass.</td></tr>
            <tr><td><strong>⏳ pulling 42%</strong></td><td>Files transferring to the worker, live percent.</td></tr>
            <tr><td><strong>🔶 heating</strong></td><td>Weights loading into VRAM/RAM right now.</td></tr>
            <tr><td><strong>🌡 hot</strong></td><td>Files on the worker’s own drive, not loaded — lifts into VRAM/RAM on first request.</td></tr>
            <tr><td><strong>○ cold</strong></td><td>Files on central storage (llm_storage) only — they copy to the worker on the first call (lazy download). Normal resting state for discovered models.</td></tr>
            <tr><td><strong>○ missing</strong></td><td>Files nowhere — neither the worker nor central storage has them (stale catalog row or phantom assignment); 📖 links the fix.</td></tr>
            <tr><td><strong>○ slot N</strong></td><td>An empty seat, waiting for an occupant.</td></tr>
          </tbody>
        </table>
      </section>

      <section id="model-policy" className="docs-sec">
        <h2>Policy tags &amp; controls (per model on a worker)</h2>
        <table className="docs-table">
          <thead><tr><th>Control</th><th>Means</th></tr></thead>
          <tbody>
            <tr><td><strong>⏲ on-demand</strong></td><td>Residency policy (default): loads on call, keeps its seat until another model needs it. Click to open the picker.</td></tr>
            <tr><td><strong>🔒 static</strong></td><td>Residency policy: locked seat, never swapped out, downloaded eagerly and kept on disk (the only tier that protects files from eviction). Also appears as a small 🔒 on the slot holding it.</td></tr>
            <tr><td><strong>📌 / 📌?</strong></td><td>Pin toggle: 📌 = the allocation (routing to this worker) survives restarts, unassign refused. Does NOT download the model or protect its files — a pinned model can be evicted and re-pulls on call. 📌? = not pinned, click to pin.</td></tr>
            <tr><td><strong>⚙ autofit</strong> (or <em>max GPU / CPU only / …G VRAM</em>)</td><td>This model&apos;s GPU/CPU budget on the worker — click to change.</td></tr>
            <tr><td><strong>⏏</strong></td><td>Unload from memory (stays assigned; reloads on demand).</td></tr>
            <tr><td><strong>×</strong></td><td>Unassign from this worker (disabled while pinned).</td></tr>
            <tr><td><strong>⏳ applying…</strong></td><td>The worker&apos;s agent is restarting (~10s) to apply your last policy change — its controls are paused until the new config reports in.</td></tr>
          </tbody>
        </table>
      </section>

      <section id="worker-chips" className="docs-sec">
        <h2>Worker identity chips (row header)</h2>
        <table className="docs-table">
          <thead><tr><th>Chip</th><th>Means</th></tr></thead>
          <tbody>
            <tr><td><strong>approved / pending / blocked</strong></td><td>The admission gate — only approved workers serve; pending awaits your ✓ admit; blocked is evicted for good.</td></tr>
            <tr><td><strong>🏷 general</strong> (or a pool name)</td><td>Which traffic pool this worker serves — a named pool serves <em>only</em> requests tagged for it. Click to set/clear.</td></tr>
            <tr><td><strong>rpc</strong></td><td>A shard backend: lends its GPU to split oversized models across machines; doesn&apos;t serve whole models itself.</td></tr>
            <tr><td><strong>🧩 comfy</strong></td><td>A ComfyUI process is alive on this box — image checkpoints route here.</td></tr>
            <tr><td><strong>⚠ CPU-only engine</strong></td><td>Its GGUF engine was built without GPU support — models run on CPU while the GPU idles. 📖 links the rebuild fix.</td></tr>
            <tr><td><strong>✓ reachable / ✗ unreachable</strong></td><td>Result of the <em>ping</em> button — can central dial this worker right now?</td></tr>
            <tr><td><strong>⏳ N pending</strong> (panel header)</td><td>Boxes waiting for your approval.</td></tr>
            <tr><td><strong>registry error</strong></td><td>The console&apos;s poll of central failed (console↔central hop, not the workers). Hover for the cause; it clears itself.</td></tr>
            <tr><td><strong>this host</strong></td><td>The central server itself, shown as the first worker row — its compute is the local slot pool.</td></tr>
          </tbody>
        </table>
      </section>

      <section id="resource-chips" className="docs-sec">
        <h2>Resource &amp; config chips</h2>
        <table className="docs-table">
          <thead><tr><th>Chip</th><th>Means</th></tr></thead>
          <tbody>
            <tr><td><strong>🖥 GPU bar</strong></td><td>Per-GPU VRAM used/total, with free on hover.</td></tr>
            <tr><td><strong>🧠 RAM … free · VRAM … free · 💾 disk … free</strong></td><td>What&apos;s left on the box — the fit guard checks new models against all three.</td></tr>
            <tr><td><strong>box caps</strong></td><td>Hard ceilings configured on the box itself; central can only set limits at or below them.</td></tr>
            <tr><td><strong>central limits</strong></td><td>Limits you set from here (⚙ limits) — the worker adopts them on its next heartbeat.</td></tr>
            <tr><td><strong>🎛 slots: N</strong></td><td>How many model seats this box runs — click to change (0–16, applies via agent restart).</td></tr>
            <tr><td><strong>spill: autofit</strong></td><td>The worker&apos;s default GPU/CPU split mode for models without an explicit budget.</td></tr>
            <tr><td><strong>✓ fit-guarded</strong></td><td>In the load table: every allocation is pre-checked against free VRAM+RAM+disk — refused with a reason, never OOM&apos;d.</td></tr>
            <tr><td><strong>⚡ allow duplicate allocation</strong></td><td>The anti-duplicate breaker: off = models already on another worker are locked; on = deliberately place a second copy.</td></tr>
          </tbody>
        </table>
      </section>

      <section id="models-tab" className="docs-sec">
        <h2>Models-tab badges</h2>
        <table className="docs-table">
          <thead><tr><th>Badge</th><th>Means</th></tr></thead>
          <tbody>
            <tr><td><strong>✓ ready</strong></td><td>Files complete on disk — chattable / assignable.</td></tr>
            <tr><td><strong>◐ partial</strong></td><td>Some files present — expand the row and resume the download.</td></tr>
            <tr><td><strong>✗ missing</strong></td><td>Registered but not downloaded.</td></tr>
            <tr><td><strong>Media ☑ + ★/☆</strong></td><td>Offer this model in the media chat&apos;s dropdown; ★ = its default model.</td></tr>
            <tr><td><strong>✓ fits in VRAM / ◐ 60% on GPU</strong></td><td>Per-worker fit forecast in the detail row; <em>probe</em> actually loads it once and reports the truth.</td></tr>
          </tbody>
        </table>
      </section>

      <section id="fixdoc" className="docs-sec">
        <h2>The 📖 link</h2>
        <p>
          A <strong>📖</strong> next to any warning is a direct link to the doc section that fixes it —
          it opens in a new tab so the console keeps its state. If a warning has no 📖, the fix isn&apos;t
          written yet; the <a className="docs-inline-link" href="#worker-troubleshooting">Fix it</a> page
          is the place to look.
        </p>
      </section>

      <div className="docs-pager">
        <a className="docs-pager-card" href="#installation"><span className="d">Next</span><span className="t">Run it yourself →</span></a>
      </div>
    </>
  )
}
