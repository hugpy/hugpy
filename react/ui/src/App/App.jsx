import { useState, useEffect, useCallback, useRef } from 'react'
import { fetchJson } from './../api'
import { useFeed } from './../runtime/feeds'
// App.jsx
import { Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider, useAuth } from './../Auth/AuthProvider'
import { PrivateRoute } from './../Auth/PrivateRoute'
import { LoginForm } from './../Auth/LoginForm'
import Docs from './../pages/Docs/Docs'

import {ModelTable,ChatPanel,HFSearch,WorkersPanel,PhoneBrickPanel,AgentNodesPanel,DiscordPanel,BridgePanel,SessionsPanel,ApiAccess,Landing,CentralConnect,AssessmentPanel,StatusBar,Navbar,SettingsPanel,EvictionsPanel,PriorityGroupsPanel,TemplatesPanel,MetricsPanel,CallsPanel,ReviewPanel} from './../components';
import EvictionFeed from './../components/EvictionsPanel/EvictionFeed'
import ModelPicker from './../components/ModelPicker/ModelPicker'
import { useChats } from './../components/ChatPanel/useChats'
import useSessionState from './../hooks/useSessionState'
import ShowroomConsole from './../showroom'
import StationDemo from '../showroom/StationDemo'
import { readSavedCentral, probeLocalCentral, isDemoHost } from './../runtime/localCentral'
// The sitewide Help widget — one shared, framework-free module, mounted
// identically by all four surfaces (see ui_shared/help/helpWidget.js).
import { mountHelpWidget, setHelpWidgetOverride } from './../../../ui_shared/help/helpWidget'
// The operator's hugpy help agent (logs + code + tests) — the console's Help.
import HelpPanel, { openHelpPanel } from './../components/HelpPanel'
import './App.css'

// NOTE: the embedded Console tab (KeeperConsole -> @hugpy/console + xterm) is
// temporarily removed — it wasn't working and weighed the UI down. The
// component files under components/KeeperConsole/ are kept; re-add the lazy
// import, the 'console' tab entry, and its render block to restore it.
export function Console({ banner = null }) {
  const [models, setModels]         = useState([])
  const [loading, setLoading]       = useState(true)
  const [error, setError]           = useState(null)
  const [activeChat, setActiveChat] = useState(() => {
    try { return localStorage.getItem('hugpy.activeChat') || null } catch { return null }
  })
  // The Compute tab can pop the SAME ChatPanel as a FLOATING overlay (position:
  // fixed, out of flow) so it never compresses/reflows the workers grid — unlike
  // the Models tab, where that panel is an inline 44% flex column. This flag is
  // only the overlay's open/closed state; the conversation itself lives in the
  // lifted chat store below (useChats + activeChat), so hiding the overlay never
  // drops history or an in-flight stream (the t142 lift). Closed by default and
  // deliberately NOT persisted — it's a transient view toggle, not layout state.
  const [computeChatOpen, setComputeChatOpen] = useState(false)
  const [jobs, setJobs]             = useState({})     // id -> job (backend Job.to_dict shape)
  const [workers, setWorkers]       = useState([])     // for the model-list "run on worker" submenu
  // Overview placement: a permanent bar on top for desktop; a dedicated
  // "Overview" tab on mobile (where a tall static header would crowd the screen).
  const initialMobile = typeof window !== 'undefined' && window.matchMedia('(max-width: 640px)').matches
  const [isMobile, setIsMobile] = useState(initialMobile)
  // Restore the tab you were on across reloads (same convention as
  // hugpy.activeChat below); unknown/stale ids fall back to the default.
  const [activeTab, setActiveTab] = useState(() => {  // overview(mobile) | models | add | compute | status | api
    const fallback = initialMobile ? 'overview' : 'status'
    const known = ['overview', 'status', 'compute', 'models', 'add', 'api', 'nodes', 'evictions', 'metrics', 'calls', 'review', 'settings']
    try {
      // `?tab=<id>` (shareable links, e.g. the Metrics panel's ?tab=metrics&model=…) wins over the saved tab.
      const fromUrl = new URLSearchParams(window.location.search).get('tab')
      if (known.includes(fromUrl)) return fromUrl
      const saved = localStorage.getItem('hugpy.activeTab')
      return known.includes(saved) ? saved : fallback
    } catch { return fallback }
  })
  useEffect(() => {
    try { localStorage.setItem('hugpy.activeTab', activeTab) } catch { /* ignore (private mode / quota) */ }
    try {  // keep a `?tab=` already in the URL in step, so a reload lands where you are
      const q = new URLSearchParams(window.location.search)
      if (q.has('tab') && q.get('tab') !== activeTab) {
        q.set('tab', activeTab)
        window.history.replaceState(window.history.state, '', `${window.location.pathname}?${q}${window.location.hash}`)
      }
    } catch { /* non-browser host */ }
  }, [activeTab])
  useEffect(() => {
    const mq = window.matchMedia('(max-width: 640px)')
    const h = e => setIsMobile(e.matches)
    mq.addEventListener('change', h)
    return () => mq.removeEventListener('change', h)
  }, [])
  // The Overview tab only exists on mobile; if we grow to desktop while it's
  // active, fall back to Status (the Overview is the static bar there).
  useEffect(() => {
    if (!isMobile && activeTab === 'overview') setActiveTab('status')
  }, [isMobile, activeTab])
  const { signOut } = useAuth()

  // Per-model, localStorage-backed conversations. Lives here (not inside
  // ChatPanel) so chats survive tab switches, route changes, and reloads, and
  // each model keeps its own thread instead of bleeding into the next.
  // `chats` (the whole per-model map) rides down to ChatPanel so its top-right
  // conversation switcher can list every populated thread from this one store.
  const { chats, getMessages, setMessages: setChatMessages } = useChats()
  const setActiveMessages = useCallback(
    (updater) => { if (activeChat) setChatMessages(activeChat, updater) },
    [activeChat, setChatMessages],
  )

  // Persist which chat is open so it reopens exactly where you left it.
  useEffect(() => {
    try {
      if (activeChat) localStorage.setItem('hugpy.activeChat', activeChat)
      else localStorage.removeItem('hugpy.activeChat')
    } catch { /* ignore (private mode / quota) */ }
  }, [activeChat])

  // A POLL MUST NEVER DESTROY GOOD DATA (2026-09-10). One degraded reply used
  // to wipe the model list, and the Models pane was gated on `error`, so a
  // single transient failure blanked the whole tab. Keep the last good list;
  // `error` only when we never got one.
  const modelsRef = useRef(models)
  modelsRef.current = models
  const refreshModels = useCallback(() => {
    fetchJson('/api/models')
      .then(data => {
        if (Array.isArray(data)) { setModels(data); setError(null) }
        setLoading(false)
      })
      .catch(e => { if (!modelsRef.current.length) setError(e.message); setLoading(false) })
  }, [])

  const refreshWorkers = useCallback(() => {
    fetchJson('/api/llm/workers')
      .then(data => { if (Array.isArray(data)) setWorkers(data) })   // same rule: a bad reply keeps the roster
      .catch(() => {})
  }, [])

  useEffect(() => { refreshModels() }, [refreshModels])
  // Workers come from the ONE live subscription (runtime/feeds.js): central
  // pushes the roster whenever a heartbeat changes it, so the "answering" pill
  // is as fresh as the beat without a 4 s poll per tab (2026-09-10).
  // refreshWorkers() stays for the optimistic refetch right after a verb.
  const fWorkers = useFeed('workers', null)
  useEffect(() => { if (Array.isArray(fWorkers)) setWorkers(fWorkers) }, [fWorkers])

  // The model roster (/api/models) has no push channel of its own: its Status
  // column (installed / partial / missing — a RESIDENCY fact) was fetched once
  // on mount and then only after a tracked job or an explicit verb. A residency
  // change from anywhere else — another operator, the API, the reconciler, a
  // hot-cache eviction — never reached the table, so status sat stale for a long
  // time (the "severe lag", 2026-09-12). Residency changes are exactly what the
  // live serving/downloads feeds report, so refetch the (~10 ms) list when
  // either ticks, coalesced (a burst of ticks → one GET). refreshModels already
  // keeps the last good list on a degraded reply, so this can never blank the tab.
  const fServing = useFeed('serving', null)
  const fDownloads = useFeed('downloads', null)
  useEffect(() => {
    const t = setTimeout(refreshModels, 400)
    return () => clearTimeout(t)
  }, [fServing, fDownloads, refreshModels])
  // Safety net: even with the feed down (fallback poll also failing) the roster
  // must not drift indefinitely — a slow unconditional refresh backstops it.
  useEffect(() => {
    const t = setInterval(refreshModels, 30_000)
    return () => clearInterval(t)
  }, [refreshModels])

  // Assign a model to a worker from the model list (one-click).
  const assignWorker = useCallback((worker, modelKey) => {
    fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/assign`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_key: modelKey }),
    }).then(refreshWorkers).catch(e => alert(`Assign failed: ${e.message}`))
  }, [refreshWorkers])

  // Live VRAM-fit probe: load the model on the worker's GPU, report fit.
  const probeWorker = useCallback(async (worker, modelKey) => {
    try {
      return await fetchJson(`/api/llm/workers/${encodeURIComponent(worker.id)}/probe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_key: modelKey }),
      })
    } catch (e) {
      return { ok: false, fit: false, error: e.message }
    }
  }, [])

  // Poll active jobs. The effect used to depend on `jobs`, so every 1 s reply
  // re-created the interval — one timer per reply, piling up per active job
  // (2026-09-10). Read jobs through a ref and keep ONE interval while active.
  const jobsRef = useRef(jobs)
  jobsRef.current = jobs
  const anyActiveJob = Object.values(jobs).some(j => j.status === 'queued' || j.status === 'running')
  useEffect(() => {
    if (!anyActiveJob) return
    const timer = setInterval(() => {
      Object.values(jobsRef.current)
        .filter(j => j.status === 'queued' || j.status === 'running')
        .forEach(j => {
          fetchJson(`/api/jobs/${j.id}`)
            .then(updated => {
              if (!updated || !updated.id) return
              setJobs(prev => ({ ...prev, [updated.id]: { ...prev[updated.id], ...updated } }))
              if (updated.status === 'completed') refreshModels()
            })
            .catch(() => {})
        })
    }, 2000)
    return () => clearInterval(timer)
  }, [anyActiveJob, refreshModels])

  // registered-model download: POST returns Job.to_dict() (has `id`, no model_key in body — we know it)
  const handleDownload = useCallback((modelKey) => {
    fetchJson(`/api/models/${encodeURIComponent(modelKey)}/download`, { method: 'POST' })
      .then(job => setJobs(prev => ({
        ...prev,
        [job.id]: { ...job, model_key: job.model_key ?? modelKey },
      })))
      .catch(e => alert(`Download failed: ${e.message}`))
  }, [])

  const handleChat = useCallback((modelKey) => { setActiveChat(modelKey) }, [])

  // ── Models tab vertical split (model list / live eviction feed) ───────────
  // Height of the feed as a % of the pane. Session-scoped like the tab's other
  // layout choices: a fresh visit gets the default, your drag sticks while you
  // work. 0 = collapsed (double-click the handle toggles).
  const [rawFeedPct, setFeedPct] = useSessionState('hugpy.models.feedPct', 32)
  // A stale or corrupted session value must not render the pane as `NaN%` and
  // collapse the model list — fall back to the default and clamp to the same
  // band the drag enforces.
  const feedPct = Number.isFinite(Number(rawFeedPct))
    ? Math.max(0, Math.min(85, Number(rawFeedPct)))
    : 32
  const modelsPaneRef = useRef(null)

  // Same drag idiom as the AssessmentPanel column resizer: window-level
  // listeners so the pointer can leave the 5px handle mid-drag, and a body
  // cursor so the whole page reads as "resizing".
  const startFeedResize = useCallback((e) => {
    e.preventDefault()
    const pane = modelsPaneRef.current
    if (!pane) return
    const onMove = (ev) => {
      const r = pane.getBoundingClientRect()
      if (!r.height) return
      const pct = ((r.bottom - ev.clientY) / r.height) * 100
      // Clamp so neither half can be dragged out of existence; snap the last
      // few percent to a true collapse so "drag it away" actually works.
      setFeedPct(pct < 6 ? 0 : Math.min(85, pct))
    }
    const onUp = () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    document.body.style.cursor = 'row-resize'
    document.body.style.userSelect = 'none'
  }, [setFeedPct])

  const handleDelete = useCallback((modelKey) => {
    if (!confirm(`Delete downloaded files for ${modelKey}?`)) return
    fetchJson(`/api/models/${encodeURIComponent(modelKey)}`, { method: 'DELETE' })
      .then(() => refreshModels())
      .catch(e => alert(`Delete failed: ${e.message}`))
  }, [refreshModels])

  // Prune = remove a NOT-installed model's registry entry (the "ghost" rows from
  // discovery that stick around). Distinct from Delete, which only removes files.
  const handlePrune = useCallback((modelKey) => {
    if (!confirm(`Prune "${modelKey}" from the model registry?\n\nThis removes the catalog entry (no files are on disk). Re-discovery or re-adding the model brings it back.`)) return
    fetchJson(`/api/models/${encodeURIComponent(modelKey)}/prune`, { method: 'POST' })
      .then(() => refreshModels())
      .catch(e => alert(`Prune failed: ${e.message}`))
  }, [refreshModels])

  // Media = whether a model is offered in the media-intelligence chat dropdown.
  // Optimistic flip so the checkbox responds instantly; revert on error.
  const handleSetMedia = useCallback((modelKey, enabled) => {
    setModels(prev => prev.map(m =>
      (m.model_key ?? m.key) === modelKey ? { ...m, media: enabled } : m))
    fetchJson(`/api/models/${encodeURIComponent(modelKey)}/media`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled }),
    }).catch(e => {
      alert(`Media toggle failed: ${e.message}`)
      setModels(prev => prev.map(m =>
        (m.model_key ?? m.key) === modelKey ? { ...m, media: !enabled } : m))
    })
  }, [])

  // The single, server-side default media model (first/preselected in the media
  // chat dropdown). One global default: set one true, clear the rest. Concrete
  // and shared — not a per-browser guess.
  const handleSetMediaDefault = useCallback((modelKey, isDefault) => {
    setModels(prev => prev.map(m => {
      const key = m.model_key ?? m.key
      if (key === modelKey) return { ...m, media_default: isDefault }
      return isDefault && m.media_default ? { ...m, media_default: false } : m
    }))
    fetchJson(`/api/models/${encodeURIComponent(modelKey)}/media-default`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ default: isDefault }),
    }).catch(e => {
      alert(`Default media model failed: ${e.message}`)
      refreshModels()  // re-sync from server on failure (optimistic update was wrong)
    })
  }, [refreshModels])

  const cancelJob = useCallback((jobId) => {
    fetchJson(`/api/jobs/${jobId}/cancel`, { method: 'POST' })
      .then(() => {
        setJobs(prev => prev[jobId]
          ? { ...prev, [jobId]: { ...prev[jobId], status: 'cancelled' } }
          : prev)
      })
      .catch(e => alert(`Cancel failed: ${e.message}`))
  }, [])

  // Resume a failed/cancelled download from its partial files. Flipping status
  // back to 'running' re-arms the poller above.
  const retryJob = useCallback((jobId) => {
    fetchJson(`/api/jobs/${jobId}/retry`, { method: 'POST' })
      .then(res => {
        if (res && res.retried === false) { alert(`Retry failed: ${res.reason || 'unknown'}`); return }
        setJobs(prev => prev[jobId]
          ? { ...prev, [jobId]: { ...prev[jobId], status: 'running', error: null, stalled: false } }
          : prev)
      })
      .catch(e => alert(`Retry failed: ${e.message}`))
  }, [])

  // freeform /llm/repos/download returns {...Job.to_dict(), model_key}; HFSearch tags hub_id
  const handleHFJobStarted = useCallback((rawJob) => {
    const jobId = rawJob.id ?? rawJob.job_id
    if (!jobId) return
    setJobs(prev => ({
      ...prev,
      [jobId]: {
        ...rawJob,
        id: jobId,
        model_key: rawJob.model_key,
        hub_id: rawJob.hub_id,
        status: rawJob.status ?? 'queued',
      },
    }))
    refreshModels()
  }, [refreshModels])


  // Inside the console the floating Help button opens the help agent too.
  useEffect(() => {
    setHelpWidgetOverride(() => openHelpPanel())
    return () => setHelpWidgetOverride(null)
  }, [])

  const jobsByModel = {}
  Object.values(jobs).forEach(j => { if (j.model_key) jobsByModel[j.model_key] = j })

  const jobsByHub = {}
  Object.values(jobs).forEach(j => { if (j.hub_id) jobsByHub[j.hub_id] = j })

  const pendingByHub = {}
  Object.values(jobs).forEach(j => { if (j.hub_id) pendingByHub[j.hub_id] = true })
  models.forEach(m => { if (m.hub_id && m.status === 'installed') pendingByHub[m.hub_id] = true })

  return (
    <div className="layout">
      {/* Shared top nav (brand + links), consistent with every other page.
          Sign out rides along as a page-specific action. In the showroom demo
          the DEMO banner is passed down as `banner` so it rides in the nav's
          own sticky header stack, directly beneath the nav (operator directive
          2026-07-21) — instead of floating above the nav in a separate context. */}
      <Navbar banner={banner}>
        <button onClick={() => openHelpPanel()} title="Ask the hugpy help agent (logs, code, tests)" className="hugpy-navbar-action">Help</button>
        <button onClick={signOut} title="Sign out of the console" className="hugpy-navbar-action">Sign out</button>
      </Navbar>
      <HelpPanel tab={activeTab} model={activeChat || ''} error={error || ''} />

      {/* Desktop: the Overview is a permanent bar on top (stats, queue, fleet,
          peers). Mobile: it's the "Overview" tab below instead. */}
      {!isMobile && <StatusBar models={models} workers={workers} />}

      <nav className="tabbar" role="tablist">
        {[
          ...(isMobile ? [{ id: 'overview', label: 'Overview', badge: null }] : []),
          { id: 'status',  label: 'Status',     badge: null },
          { id: 'compute', label: 'Compute',    badge: workers.filter(w => w.status === 'online').length || null },
          { id: 'models',  label: activeChat ? 'Models 💬' : 'Models', badge: models.filter(m => m.status === 'installed').length || null },
          { id: 'add',     label: 'Add models 🤗', badge: null },
          { id: 'api',     label: 'API',        badge: null },
          { id: 'nodes',   label: 'Agent Nodes', badge: null },
          { id: 'evictions', label: 'Evictions', badge: null },
          { id: 'metrics', label: 'Metrics',    badge: null },
          { id: 'calls',   label: 'Calls',      badge: null },
          { id: 'review',  label: 'Grader',     badge: null },
          { id: 'settings',label: 'Settings',    badge: null },
        ].map(t => (
          <button
            key={t.id}
            role="tab"
            aria-selected={activeTab === t.id}
            className={`tab ${activeTab === t.id ? 'tab-active' : ''}`}
            onClick={() => setActiveTab(t.id)}
          >
            {t.label}
            {t.badge != null && <span className="tab-badge">{t.badge}</span>}
          </button>
        ))}
      </nav>

      <div className="tab-body">
        {activeTab === 'overview' && isMobile && (
          <div className="tab-pane">
            <StatusBar models={models} workers={workers} />
          </div>
        )}

        {activeTab === 'models' && (
          <div className="body">
            {/* Vertical split: the model list on top, the live eviction feed
                below. The operator calls a model from chat (activeChat) and
                watches THAT model's eviction behaviour on the same screen —
                the feed highlights it rather than filtering to it. */}
            <section className="models-pane" data-chat-open={!!activeChat}>
              {/* Explicit, ORDERED model fallback lists. Model management is
                  the right home: a priority group decides WHICH model key a
                  request resolves to, and its members are rows in the table
                  right below it (including blocked ones, which a group skips
                  and can never launder). Collapsed by default — it sits above
                  the list, so it must cost nothing when unused. */}
              <PriorityGroupsPanel models={models} />
              <TemplatesPanel workers={workers} />
              {/* The list/feed split lives in its own host so the two
                  percentage heights still sum to ONE box. Measuring and
                  resizing against the host (not the whole pane) is what keeps
                  the drag math honest now that something sits above it. */}
              <div className="models-pane-split-host" ref={modelsPaneRef}>
              <div className="models-pane-list"
                   style={{ height: `${100 - feedPct}%` }}>
                {loading && <div className="placeholder">Loading models…</div>}
                {!loading && error && !models.length && <div className="placeholder error">API error: {error}</div>}
                {!loading && models.length > 0 && (
                  <ModelTable
                    models={models}
                    jobsByModel={jobsByModel}
                    activeChat={activeChat}
                    onDownload={handleDownload}
                    onChat={handleChat}
                    onDelete={handleDelete}
                    onPrune={handlePrune}
                    onSetMedia={handleSetMedia}
                    onSetMediaDefault={handleSetMediaDefault}
                    onCancel={cancelJob}
                    onRetry={retryJob}
                    workers={workers}
                    onAssignWorker={assignWorker}
                    onProbeWorker={probeWorker}
                    onRefresh={refreshModels}
                  />
                )}
              </div>

              <div
                className="models-split"
                role="separator"
                aria-orientation="horizontal"
                aria-label="Resize the eviction feed"
                title="Drag to resize · double-click to collapse"
                onMouseDown={startFeedResize}
                onDoubleClick={() => setFeedPct(p => (p <= 4 ? 32 : 0))}
              >
                <span className="models-split-grip" />
              </div>

              <div className="models-pane-feed" style={{ height: `${feedPct}%` }}>
                <EvictionFeed selectedModel={activeChat} />
              </div>
              </div>
            </section>

            {activeChat && (
              <ChatPanel
                key={activeChat}
                modelKey={activeChat}
                model={models.find(m => (m.model_key ?? m.key) === activeChat)}
                messages={getMessages(activeChat)}
                setMessages={setActiveMessages}
                onClose={() => setActiveChat(null)}
                chats={chats}
                models={models}
                onSwitchChat={handleChat}
              />
            )}
          </div>
        )}

        {activeTab === 'add' && (
          <div className="tab-pane">
            <HFSearch
              embedded
              models={models}
              onJobStarted={handleHFJobStarted}
              onCancelJob={cancelJob}
              onRetryJob={retryJob}
              pendingByHub={pendingByHub}
              jobsByHub={jobsByHub}
            />
          </div>
        )}

        {activeTab === 'compute' && (
          <div className="tab-pane">
            {/* The open affordance for the floating chat. It only toggles the
                overlay's visibility; the overlay itself is fixed-position (below)
                so opening it costs the compute layout no space. */}
            <div className="compute-chat-toolbar">
              <button
                type="button"
                className={`compute-chat-toggle ${computeChatOpen ? 'is-open' : ''}`}
                aria-pressed={computeChatOpen}
                aria-expanded={computeChatOpen}
                onClick={() => setComputeChatOpen(o => !o)}
                title={computeChatOpen ? 'Close the floating chat' : 'Chat with a model without leaving Compute'}
              >
                {computeChatOpen ? '✕ Close chat' : `💬 Chat${activeChat ? ' ●' : ''}`}
              </button>
            </div>
            {/* Each pool is a collapsible row — click its header bar to expand. */}
            <WorkersPanel models={models} />
            {/* SlotsPanel retired 2026-07-03 (daylight item 3): central presents
                as a worker row inside WorkersPanel with its own load/unload. */}
            <PhoneBrickPanel />

            {/* FLOATING chat overlay — the SAME <ChatPanel> the Models tab
                renders, driven by the SAME lifted state (activeChat + the
                useChats store), so streaming, model switching and localStorage
                persistence are identical and survive closing/reopening the
                overlay (onClose only hides it; it never nulls activeChat). It's
                position:fixed (see App.css .compute-chat-overlay), so it floats
                over the workers grid and never reflows it. */}
            {computeChatOpen && (
              <aside className="compute-chat-overlay" role="dialog" aria-label="Model chat">
                {activeChat ? (
                  <ChatPanel
                    key={activeChat}
                    modelKey={activeChat}
                    model={models.find(m => (m.model_key ?? m.key) === activeChat)}
                    messages={getMessages(activeChat)}
                    setMessages={setActiveMessages}
                    onClose={() => setComputeChatOpen(false)}
                    chats={chats}
                    models={models}
                    onSwitchChat={handleChat}
                  />
                ) : (
                  /* No conversation open yet: from Compute there's no model table
                     to start one, so offer the same ModelPicker other panels use.
                     Picking a model sets activeChat (App's own opener), and the
                     ChatPanel above takes over. */
                  <div className="compute-chat-empty">
                    <div className="compute-chat-empty-head">
                      <strong>Chat</strong>
                      <button className="btn-close" onClick={() => setComputeChatOpen(false)} title="Close chat">✕</button>
                    </div>
                    <div className="compute-chat-empty-body">
                      <p>Pick a model to start chatting — its conversation is saved and shared with the Models tab.</p>
                      <ModelPicker
                        models={models}
                        onPick={(mk) => handleChat(mk)}
                        placeholder="Pick a model…"
                        autoFocus
                      />
                    </div>
                  </div>
                )}
              </aside>
            )}
          </div>
        )}

        {activeTab === 'status' && (
          <div className="tab-pane">
            <AssessmentPanel models={models} workers={workers} />
          </div>
        )}

        {activeTab === 'evictions' && (
          <div className="tab-pane">
            <EvictionsPanel />
          </div>
        )}

        {activeTab === 'metrics' && (
          <div className="tab-pane">
            <MetricsPanel models={models} />
          </div>
        )}

        {activeTab === 'calls' && (
          <div className="tab-pane">
            {/* Who called what, from where, when, served by whom (2026-09-10). */}
            <CallsPanel />
          </div>
        )}

        {activeTab === 'review' && (
          <div className="tab-pane">
            {/* Model grader/tester: surfaces the review pipeline (screen →
                smoke-load → judge) over /api/llm/review/*. */}
            <ReviewPanel />
          </div>
        )}

        {activeTab === 'settings' && (
          <div className="tab-pane">
            <SettingsPanel workers={workers} />
          </div>
        )}

        {activeTab === 'api' && (
          <div className="tab-pane">
            {/* Each is a collapsible row — click its header bar to expand. */}
            <ApiAccess models={models} />
            <DiscordPanel models={models} />
            <BridgePanel models={models} />
            <SessionsPanel />
          </div>
        )}

        {activeTab === 'nodes' && (
          <div className="tab-pane">
            {/* P3.3: operator console for agent nodes (register/heartbeat via the
                node daemon; dispatch + watch runs via the /agent/* routes).
                Promoted from the Compute tab to its own tab (operator, 2026-07-15). */}
            <AgentNodesPanel />
          </div>
        )}

      </div>
    </div>
  )
}




// /console gate: a reachable same-origin backend (dev, or a local `hugpy serve`
// opened at its own origin) → the full console. No same-origin backend (the
// public brochure) → probe the visitor's OWN localhost central before settling
// for the showroom.
function ConsoleGate() {
  const { reachable } = useAuth()
  // Dedicated demo host (demo.hugpy.ai): ALWAYS the showroom — even if a
  // same-origin backend is reachable — and skip the localhost-central probe, so
  // a demo visitor never lands in a live console there.
  if (isDemoHost()) return <ShowroomConsole><Console /></ShowroomConsole>
  // Same-origin backend reachable (dev.hugpy.ai, or a local `hugpy serve` opened
  // at its own origin) → the live console directly.
  if (reachable) return <Console />
  // Backendless front door (the public hugpy.ai brochure on :7003, no /api).
  // Before falling back to the demo, see if THIS visitor is running their own
  // `hugpy serve` on localhost:7002 — if so, hand them the real console at its
  // own origin (where the API is same-origin → no CORS). Only if nothing answers
  // → the explorable SHOWROOM (the real console wrapped in demo mode).
  return <LocalCentralGate />
}

// Brochure-only: probe the visitor's own localhost central, then route.
//  • central up   → navigate to it (open the real local console at its origin)
//  • central down → the showroom demo
// `?showroom=1` (or `?demo=1`) forces the demo without probing — a stable way to
// reach the showroom even when a local central is running.
function LocalCentralGate() {
  const force = (() => {
    try {
      const q = new URLSearchParams(window.location.search)
      return q.has('showroom') || q.has('demo')
    } catch { return false }
  })()
  const [stage, setStage] = useState(force ? 'showroom' : 'probing') // 'probing' | 'showroom'
  useEffect(() => {
    if (force) return
    let cancelled = false
    const central = readSavedCentral()
    probeLocalCentral(central).then((up) => {
      if (cancelled) return
      if (up) window.location.assign(central)   // open the real local console
      else setStage('showroom')
    })
    return () => { cancelled = true }
  }, [force])
  if (stage === 'probing') {
    return (
      <div className="console-probe" role="status" aria-live="polite">
        <span className="console-probe-dot" />
        Looking for your local hugpy central…
      </div>
    )
  }
  return <ShowroomConsole><Console /></ShowroomConsole>
}

export default function App() {
  // The floating Keeper Help button, on every screen of this SPA. Mounted from
  // the ONE shared, dependency-free module (react/ui_shared/help/helpWidget.js)
  // that the media/video/fleet arms mount too — a plain DOM module rather than a
  // component, so the four arms' separate React installs never have to agree on
  // a version (see that file's header). It renders its own fixed-position node
  // outside this tree, so it survives every route change; the effect only mounts
  // and unmounts it.
  useEffect(() => mountHelpWidget({ apiBase: '/api', surface: 'console' }), [])
  return (
    <AuthProvider>
      <Routes>
        {/* Front door: the welcome/Landing page is the index for every face —
            prod hugpy.ai, dev, and a local `hugpy serve`. The console is an
            explicit click away at /console, never the bare root, so visitors
            always land on the welcome page first (the agreed posture). */}
        <Route path="/" element={<Landing />} />
        <Route path="/welcome" element={<Navigate to="/" replace />} />
        <Route path="/login" element={<LoginForm />} />
        <Route path="/docs" element={<Docs />} />
        {/* demo.hugpy.ai/station — the hugpy-station showroom (canned SPA + live
            agent terminal), a React page instead of a bare nginx alias. Any
            other host sends /station to the front door. */}
        <Route path="/station/*" element={isDemoHost() ? <StationDemo /> : <Navigate to="/" replace />} />
        <Route element={<PrivateRoute />}>
          <Route path="/console/*" element={<ConsoleGate />} />
        </Route>
        {/* Unknown paths fall back to the front door, not a blank console. */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AuthProvider>
  )
}