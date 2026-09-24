// helpBus.js — how the rest of the console reaches the Help panel without
// importing it: a window CustomEvent to open it (optionally with a prompt and
// an explicit error/context), and a tiny "last error" slot fed by
// `hugpy:api-error` events (dispatched by api.ts fetchJson on any API failure)
// so the context strip can carry the last error the console actually showed.

export const OPEN_EVENT = 'hugpy:help-open'
export const API_ERROR_EVENT = 'hugpy:api-error'

let lastError = ''
let listening = false

function listen() {
  if (listening || typeof window === 'undefined') return
  listening = true
  window.addEventListener(API_ERROR_EVENT, (e) => {
    const d = (e && e.detail) || {}
    lastError = [d.status ? `HTTP ${d.status}` : '', d.url || '', d.message || ''].filter(Boolean).join(' · ')
  })
}
listen()

export function getLastConsoleError() { return lastError }
export function noteConsoleError(text) { lastError = String(text || '') }

// openHelpPanel({ prompt?, error?, context?: {k: v}, send?: bool })
export function openHelpPanel(detail = {}) {
  if (typeof window === 'undefined') return false
  window.dispatchEvent(new CustomEvent(OPEN_EVENT, { detail }))
  return true
}
