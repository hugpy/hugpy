import { useEffect, useState, useCallback, useRef } from 'react'
import { fetchJson } from '../../api'
import { hugpyFetch } from '../../runtime/config'
import './AgentNodesPanel.css'

// P3.3 — the operator console panel for agent nodes.
//
// Reads the P3.1 / P3.1b operator-gated routes (all under the /api mount):
//   GET  /api/agent/nodes                 — full roster + live health
//   POST /api/agent/<id>/dispatch {task}  — queue a task -> 201 with its {seq}
//   GET  /api/agent/nodes/<id>/tasks       — recent task history
//   GET  /api/agent/<id>/tasks/<seq>      — one task's row (status/result)
//
// The roster and recent task history are recovered from central, so completed
// work remains visible across page reloads. Modeled on WorkersPanel / PhoneBrickPanel (collapsible bar, ~10s
// roster poll, error surfaced in the header). The /agent/* routes are not
// deployed everywhere yet, so the roster read degrades to a clear
// "routes not deployed" state rather than a crash or a forever-spinner.

const TERMINAL = ['done', 'error']

// A node heartbeats ~every 30s; treat >3 missed beats as stale.
const LIVE_WINDOW_SECS = 90
// Poll cadences: the roster slowly (health drift), in-flight tasks fast.
const ROSTER_POLL_MS = 10_000
const TASK_POLL_MS = 2_500

// last-seen relative time from an epoch-seconds clock.
function fmtAgo(epoch) {
  if (!epoch) return 'never'
  const secs = Math.max(0, Date.now() / 1000 - Number(epoch))
  if (secs < 5) return 'just now'
  if (secs < 90) return `${Math.round(secs)}s ago`
  const m = secs / 60
  if (m < 60) return `${Math.round(m)}m ago`
  const h = m / 60
  if (h < 24) return `${Math.round(h)}h ago`
  return `${Math.round(h / 24)}d ago`
}

function isLive(node) {
  return node.last_seen != null &&
    (Date.now() / 1000 - Number(node.last_seen)) <= LIVE_WINDOW_SECS
}

// Current-task readout is EXACT-STRING per the central heartbeat contract:
//   current_task === ""   -> the node reported "no task" (idle)
//   current_task === "<s>" -> the node is working task <s>
//   current_task == null   -> the heartbeat kept the PRIOR value (never written,
//                             or a partial beat) — this is "no report", NOT idle.
// (Central's heartbeat only writes non-null fields, so null must not be read as
// idle — it may be a stale busy marker. See HANDOFF-p31b baseline quirk #1.)
function taskLabel(node) {
  const ct = node.current_task
  if (ct === '') return { text: 'idle', cls: 'an-ct-idle', title: 'node reported no current task (current_task === "")' }
  if (ct == null) return { text: 'no report', cls: 'an-ct-unknown', title: 'no current_task reported yet (null = heartbeat kept prior value — not necessarily idle)' }
  return { text: `▶ task ${ct}`, cls: 'an-ct-busy', title: `node reports it is working task seq ${ct}` }
}

function statusClass(status) {
  if (status === 'idle') return 'an-st-idle'
  if (status === 'busy' || status === 'running') return 'an-st-busy'
  if (status === 'enrolled') return 'an-st-enrolled'
  if (status === 'offline') return 'an-st-offline'
  return 'an-st-other'
}

function taskStatusClass(status) {
  if (status === 'done') return 'an-ts-done'
  if (status === 'error') return 'an-ts-error'
  return 'an-ts-queued'   // queued / anything not terminal
}

function taskLabelText(task) {
  if (!task || typeof task !== 'object') return String(task || '(empty task)')
  return task.instruction || task.prompt || task.kind || JSON.stringify(task)
}

// One dispatched-task card: prompt echo, live status, spinner while queued, and
// the final result (which may end with central's "…[truncated N bytes]" marker).
function TaskCard({ rec }) {
  const v = rec.view || {}
  const status = v.status || 'queued'
  const running = !TERMINAL.includes(status)
  return (
    <div className={`an-task an-task-${status}`}>
      <div className="an-task-head">
        <span className={`an-ts-pill ${taskStatusClass(status)}`}>
          {running && <span className="an-spin" />}
          {status}
        </span>
        <span className="an-task-seq" title="dispatch sequence (this node's monotonic task cursor)">seq {rec.seq}</span>
        <span className="an-task-prompt" title={rec.prompt}>{rec.prompt}</span>
        {v.created_at != null && <span className="an-task-fin">· queued {fmtAgo(v.created_at)}</span>}
        {v.finished_at != null && (
          <span className="an-task-fin" title="finished_at (from central)">· {fmtAgo(v.finished_at)}</span>
        )}
      </div>
      {running && (
        <div className="an-task-wait">Dispatched — waiting for the node to pull, run, and report…</div>
      )}
      {v.task != null && (
        <details className="an-task-payload">
          <summary>Task payload</summary>
          <pre className="an-task-result">{JSON.stringify(v.task, null, 2)}</pre>
        </details>
      )}
      {status === 'done' && (
        <pre className="an-task-result">{v.result != null && v.result !== '' ? v.result : '(empty result)'}</pre>
      )}
      {status === 'error' && (
        <pre className="an-task-result an-task-result-err">{v.result || 'node reported an error (no detail)'}</pre>
      )}
    </div>
  )
}

// One node row: identity + health, a dispatch input, and the session's task
// cards for this node.
function NodeRow({ node, tasks, onDispatch }) {
  const [prompt, setPrompt] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState(null)
  const live = isLive(node)
  const tl = taskLabel(node)

  const send = useCallback(async () => {
    const text = prompt.trim()
    if (!text || busy) return
    setBusy(true); setErr(null)
    try {
      await onDispatch(node, text)
      setPrompt('')
    } catch (e) {
      setErr(e.message || 'dispatch failed')
    } finally {
      setBusy(false)
    }
  }, [prompt, busy, node, onDispatch])

  return (
    <div className={`an-node${live ? '' : ' an-node-stale'}${node.revoked ? ' an-node-revoked' : ''}`}>
      <div className="an-node-head">
        <span className={`an-dot ${live ? 'an-dot-live' : 'an-dot-stale'}`}
              title={live ? 'heartbeat fresh' : 'no recent heartbeat (stale / offline)'} />
        <span className="an-name" title={node.id}>{node.name || node.id}</span>
        <span className={`an-status ${statusClass(node.status)}`}>{node.status || 'unknown'}</span>
        <span className={`an-ct ${tl.cls}`} title={tl.title}>{tl.text}</span>
        {node.version && <span className="an-ver" title="node version">v{node.version}</span>}
        <span className="an-seen" title="last heartbeat">seen {fmtAgo(node.last_seen)}</span>
        {node.revoked && <span className="an-revoked" title="node credential revoked">revoked</span>}
      </div>

      {(node.host || (node.capabilities && node.capabilities.length > 0)) && (
        <div className="an-node-meta">
          {node.host && <span className="an-host" title="reported host">{node.host}</span>}
          {Array.isArray(node.capabilities) && node.capabilities.map(c => (
            <span key={c} className="an-cap">{c}</span>
          ))}
        </div>
      )}

      <div className="an-dispatch">
        <input
          className="an-dispatch-input"
          type="text"
          value={prompt}
          placeholder="Prompt to dispatch to this node…"
          disabled={busy || node.revoked}
          onChange={e => setPrompt(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') send() }}
        />
        <button className="an-dispatch-btn" onClick={send}
                disabled={busy || node.revoked || !prompt.trim()}
                title="POST /agent/<id>/dispatch {task:{prompt}}">
          {busy ? 'Dispatching…' : '▶ Dispatch'}
        </button>
      </div>
      {err && <div className="an-dispatch-err">{err}</div>}

      {tasks.length > 0 && (
        <div className="an-task-list">
          {tasks.map(t => <TaskCard key={t.key} rec={t} />)}
        </div>
      )}
    </div>
  )
}

export default function AgentNodesPanel({ embedded = false }) {
  const [nodes, setNodes] = useState([])
  const [open, setOpen] = useState(false)
  // roster state machine: what the last /agent/nodes read told us.
  //   loading | ok | unavailable (routes not deployed) | auth | error
  const [state, setState] = useState({ kind: 'loading', msg: null })
  // Recent task history loaded from central, plus newly dispatched tasks.
  // Each: { key, nodeId, nodeName, seq, prompt, view, dispatchedAt }
  const [tasks, setTasks] = useState([])
  const aliveRef = useRef(true)

  // Roster read — done via raw hugpyFetch so we can branch on the status code:
  // a 404/501 on this collection endpoint is the "routes not deployed" signal.
  const loadTaskHistory = useCallback(async (nodeList) => {
    const perNode = await Promise.all(nodeList.map(async node => {
      try {
        const data = await fetchJson(
          `/api/agent/nodes/${encodeURIComponent(node.id)}/tasks?limit=100`)
        return (data.tasks || []).map(view => ({
          key: `${node.id}:${view.seq}`,
          nodeId: node.id,
          nodeName: node.name || node.id,
          seq: view.seq,
          prompt: taskLabelText(view.task),
          view,
          dispatchedAt: Number(view.created_at || 0) * 1000,
        }))
      } catch {
        return []
      }
    }))
    if (aliveRef.current) setTasks(perNode.flat())
  }, [])

  const loadNodes = useCallback(async () => {
    let r
    try {
      r = await hugpyFetch('/api/agent/nodes')
    } catch (e) {
      if (aliveRef.current) setState({ kind: 'error', msg: e.message || 'network error' })
      return
    }
    const body = await r.text()
    if (!aliveRef.current) return
    if (r.ok) {
      let data = []
      try { data = JSON.parse(body) } catch { data = [] }
      const nodeList = Array.isArray(data) ? data : []
      setNodes(nodeList)
      void loadTaskHistory(nodeList)
      setState({ kind: 'ok', msg: null })
      return
    }
    if (r.status === 404 || r.status === 501) {
      setState({ kind: 'unavailable', msg: 'agent routes not deployed on this central yet' })
    } else if (r.status === 401 || r.status === 403) {
      setState({ kind: 'auth', msg: 'operator authentication required to view agent nodes' })
    } else {
      let msg = body.trim()
      try { const j = JSON.parse(body); msg = j.error || j.detail || j.message || msg } catch { /* keep text */ }
      setState({ kind: 'error', msg: msg || `HTTP ${r.status}` })
    }
  }, [loadTaskHistory])

  useEffect(() => {
    aliveRef.current = true
    loadNodes()
    const t = setInterval(loadNodes, ROSTER_POLL_MS)
    return () => { aliveRef.current = false; clearInterval(t) }
  }, [loadNodes])

  // Poll every in-flight (non-terminal) dispatched task until it finalizes.
  // Re-runs whenever `tasks` changes, so once a task hits done/error it drops
  // out of `active` and its polling stops; when none are active, no interval.
  useEffect(() => {
    const active = tasks.filter(t => !TERMINAL.includes(t.view && t.view.status))
    if (!active.length) return
    const tick = async () => {
      await Promise.all(active.map(async t => {
        try {
          const row = await fetchJson(
            `/api/agent/${encodeURIComponent(t.nodeId)}/tasks/${encodeURIComponent(t.seq)}`)
          setTasks(prev => prev.map(x => (x.key === t.key ? { ...x, view: row } : x)))
        } catch { /* transient (or routes just went away) — keep the last view */ }
      }))
    }
    const id = setInterval(tick, TASK_POLL_MS)
    return () => clearInterval(id)
  }, [tasks])

  // Dispatch a prompt to a node. Returns the created task view (throws on
  // non-2xx so the row can show the reason inline). Captures seq for tracking.
  const dispatch = useCallback(async (node, prompt) => {
    const created = await fetchJson(
      `/api/agent/${encodeURIComponent(node.id)}/dispatch`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task: { prompt } }),
      })
    setTasks(prev => [
      {
        key: `${node.id}:${created.seq}`,
        nodeId: node.id,
        nodeName: node.name || node.id,
        seq: created.seq,
        prompt,
        view: created,
        dispatchedAt: Date.now(),
      },
      ...prev,
    ])
    return created
  }, [])

  const tasksFor = useCallback(
    (id) => tasks.filter(t => t.nodeId === id), [tasks])

  const liveCount = nodes.filter(isLive).length
  const headerNote = {
    loading: null,
    ok: null,
    unavailable: 'routes not deployed',
    auth: 'operator auth required',
    error: 'roster error',
  }[state.kind]

  return (
    <div className="agentnodes-panel">
      <div className={`an-bar${embedded ? ' an-bar-static' : ''}`}
           onClick={embedded ? undefined : () => setOpen(o => !o)}>
        <span className="an-title">🕸 Agent Nodes — dispatch &amp; watch</span>
        <span className="an-count">{liveCount} live / {nodes.length} total</span>
        {headerNote && (
          <span className={`an-headnote an-headnote-${state.kind}`} title={state.msg || ''}>{headerNote}</span>
        )}
        {!embedded && <span className="an-toggle">{open ? '▾' : '▸'}</span>}
      </div>

      {(embedded || open) && (
        <div className="an-body">
          <div className="an-howto">
            Nodes enroll by running <code>hugpy-agent serve --node</code> with{' '}
            <code>HUGPY_AGENT_CENTRAL</code> pointed at this console. Dispatched
            prompts are pulled and run by the node; results stream back here as
            each task finalizes.
          </div>

          {state.kind === 'unavailable' && (
            <div className="an-banner an-banner-unavailable">
              The <code>/agent/*</code> routes are not deployed on this central yet.
              {state.msg ? <> ({state.msg})</> : null} The panel will populate once
              the P3.1/P3.1b blueprint is live.
            </div>
          )}
          {state.kind === 'auth' && (
            <div className="an-banner an-banner-auth">
              {state.msg}. Sign in as the operator (or set the operator token) to
              view and dispatch to agent nodes.
            </div>
          )}
          {state.kind === 'error' && (
            <div className="an-banner an-banner-error" title={state.msg || ''}>
              Could not read the node roster: {state.msg || 'unknown error'}
            </div>
          )}
          {state.kind === 'loading' && (
            <div className="an-banner an-banner-loading">Loading node roster…</div>
          )}

          {state.kind === 'ok' && nodes.length === 0 && (
            <div className="an-empty">No agent nodes have enrolled yet.</div>
          )}

          {nodes.map(n => (
            <NodeRow key={n.id} node={n} tasks={tasksFor(n.id)} onDispatch={dispatch} />
          ))}
        </div>
      )}
    </div>
  )
}
