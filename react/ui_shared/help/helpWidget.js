/*
 * ui_shared/help/helpWidget.js — the ONE floating Help widget for every
 * dev.hugpy.ai surface.
 *
 * WHY IT IS PLAIN DOM AND NOT A REACT COMPONENT
 * ---------------------------------------------
 * The navbar is shared as a DATA module (links.js) with a per-arm React
 * rendering copied into each arm, because each arm renders links its own way
 * (react-router <Link> vs <a>, Tailwind vs hand CSS). The Help widget has no
 * such per-arm variation — it is the SAME chrome-level panel everywhere, and it
 * is deliberately outside the page's layout. Copying a React component four
 * times would mean four copies to fix and would couple the widget to each arm's
 * React version (the main SPA is React 18 on webpack; the arms are React 18 on
 * Vite, each with their own node_modules). A dependency-free DOM module has one
 * copy, one behavior, and no version coupling at all: every arm's mount site is
 * a single `mountHelpWidget()` call in an effect / bootstrap.
 *
 * WHAT IT DOES
 *   * Ask tab   — streams POST {apiBase}/chat/stream (the SAME endpoint and SSE
 *                 shape the console's ChatPanel consumes) with a Keeper system
 *                 preamble: answer questions, explain errors, guide through
 *                 common problems.
 *   * "Include page context" — appends the current URL plus the last few JS
 *                 console errors from a ring buffer (window.onerror,
 *                 unhandledrejection, console.error) to the outgoing message.
 *   * Request a fix — POST {apiBase}/keeper/help/report, which files the
 *                 request as a PENDING message on the Keeper's Discord bridge
 *                 for operator approval. The widget shows the returned ticket id.
 *
 * WHAT IT NEVER DOES
 *   Execute anything. There is no action path in this file — no navigation it
 *   is told to take, no fetch it is told to make, no setting it can write. A
 *   proposed action becomes a ticket an operator approves or rejects in the
 *   existing console flow. That boundary is the whole point of the widget.
 *
 * Every request uses credentials:'include' and expects the member gate: a 401
 * swaps the body for a sign-in prompt pointing at the main SPA's /login.
 */
// Shared stylesheet, imported the same way ui_shared/navbar/navbar.css is by
// every arm's Navbar copy — one file, all four bundlers, no per-arm theming.
import "./helpWidget.css";

export const HELP_WIDGET_ID = "hugpy-help-widget";
const ERR_BUFFER_MAX = 8;
const CTX_ERR_MAX = 4;

// The controller of the widget currently mounted on this page, or null. There
// is at most ONE widget per page (mountHelpWidget is idempotent on
// HELP_WIDGET_ID), so a module-level slot is the whole registry — no event bus,
// no globals, still dependency-free. Set on mount, cleared on unmount.
let activeWidget = null;

// THE DIRECT LINE (2026-08-20): Ask goes to POST /keeper/help/ask — the hugpy
// VM's keeper seat (the same B the station console talks to), grounded
// server-side in live fleet state. No client-side preamble, no substitute
// model, no pretend streaming. `offline: true` replies are B's deterministic
// state readout and are labelled as such — honest degradation, never fake.

// --------------------------------------------------------------------------
// JS-error ring buffer. Installed once per page; captures what the browser
// console saw so "include page context" can attach it to a question.
// --------------------------------------------------------------------------
const errorRing = [];
let ringInstalled = false;

function pushError(text) {
  const line = String(text || "").slice(0, 400);
  if (!line) return;
  if (errorRing[errorRing.length - 1] === line) return; // collapse repeats
  errorRing.push(line);
  while (errorRing.length > ERR_BUFFER_MAX) errorRing.shift();
}

function installErrorRing() {
  if (ringInstalled || typeof window === "undefined") return;
  ringInstalled = true;
  window.addEventListener("error", (e) => {
    const where = e && e.filename ? ` (${e.filename}:${e.lineno || 0})` : "";
    pushError(`${(e && e.message) || "script error"}${where}`);
  });
  window.addEventListener("unhandledrejection", (e) => {
    const r = e && e.reason;
    pushError(`unhandled rejection: ${(r && (r.message || r)) || "unknown"}`);
  });
  // console.error is wrapped, never replaced: the original is always called
  // first so devtools output is byte-identical with or without the widget.
  try {
    const original = console.error;
    console.error = function (...args) {
      try {
        pushError(args.map((a) => (a && a.message) || String(a)).join(" "));
      } catch { /* never let the buffer break logging */ }
      return original.apply(console, args);
    };
  } catch { /* frozen console — the listeners above still cover most cases */ }
}

function pageContext() {
  const lines = ["page: " + (typeof location !== "undefined" ? location.href : "?")];
  const recent = errorRing.slice(-CTX_ERR_MAX);
  if (recent.length) {
    lines.push("recent JS console errors:");
    recent.forEach((e) => lines.push("  - " + e));
  } else {
    lines.push("recent JS console errors: none");
  }
  return lines.join("\n");
}

// --------------------------------------------------------------------------
// Tiny DOM helpers (no framework, no deps).
// --------------------------------------------------------------------------
function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  Object.entries(attrs).forEach(([k, v]) => {
    if (v === false || v == null) return;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : String(v));
  });
  (Array.isArray(children) ? children : [children]).forEach((c) => {
    if (c) node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  });
  return node;
}

/**
 * Mount the Help widget on the current page.
 *
 * @param {Object}  [opts]
 * @param {string}  [opts.apiBase="/api"]  Same-origin API base (the arms and the
 *                                         SPA all proxy /api to central).
 * @param {string}  [opts.surface=""]      Free-form label for the surface doing
 *                                         the mounting ("console", "media",
 *                                         "video", "fleet") — shown in the panel
 *                                         header and included in a filed ticket.
 * @param {string}  [opts.modelKey]        Optional explicit chat model. Omitted
 *                                         by default so central picks its own.
 * @param {string}  [opts.loginHref="/login"] Where the 401 prompt points.
 * @returns {() => void} unmount
 */
export function mountHelpWidget(opts = {}) {
  if (typeof document === "undefined") return () => {};
  const apiBase = (opts.apiBase || "/api").replace(/\/$/, "");
  const surface = opts.surface || "";
  const modelKey = opts.modelKey || "";
  const loginHref = opts.loginHref || "/login";

  // Idempotent: React 18 StrictMode double-invokes effects in development, and
  // two of these would stack two buttons in the same corner.
  const existing = document.getElementById(HELP_WIDGET_ID);
  if (existing) return () => existing.remove();

  installErrorRing();

  const root = el("div", { id: HELP_WIDGET_ID });
  let open = false;
  let tab = "ask";
  let busy = false;
  const history = [];        // [{role:'user'|'assistant', content}]
  let includeCtx = false;
  // Set by openHelpWidget({prompt}); consumed by the next renderAsk, which
  // drops it into the textarea (and sends it when autoSend was asked for).
  // Held as state rather than passed through render() so a re-render for any
  // other reason can't resurrect a prompt that was already placed.
  let pendingPrompt = "";
  let pendingSend = false;

  // ---- render ------------------------------------------------------------
  function render() {
    root.textContent = "";
    root.appendChild(open ? panel() : fab());
  }

  function fab() {
    return el("button", {
      class: "hgh-fab", type: "button",
      "aria-label": "Open help",
      onclick: () => { open = true; render(); },
    }, [el("span", { class: "hgh-fab-dot" }), el("span", { text: "Help" })]);
  }

  function panel() {
    const body = el("div", { class: "hgh-body" });
    const foot = el("div", { class: "hgh-foot" });
    const wrap = el("div", { class: "hgh-panel" }, [
      el("div", { class: "hgh-head" }, [
        el("div", {}, [
          el("div", { class: "hgh-title", text: "Keeper help" }),
          el("div", { class: "hgh-sub", text: surface ? `${surface} · ask or request a fix` : "ask or request a fix" }),
        ]),
        el("button", {
          class: "hgh-x", type: "button", "aria-label": "Close help",
          onclick: () => { open = false; render(); },
        }, "×"),
      ]),
      el("div", { class: "hgh-tabs", role: "tablist" }, [
        tabButton("ask", "Ask"),
        tabButton("fix", "Request a fix"),
      ]),
      body, foot,
    ]);
    if (tab === "ask") renderAsk(body, foot);
    else renderFix(body, foot);
    return wrap;
  }

  function tabButton(key, label) {
    return el("button", {
      class: "hgh-tab", type: "button", role: "tab",
      "aria-selected": tab === key ? "true" : "false",
      onclick: () => { tab = key; render(); },
    }, label);
  }

  function say(body, role, text) {
    const node = el("div", { class: `hgh-msg ${role}`, text });
    body.appendChild(node);
    body.scrollTop = body.scrollHeight;
    return node;
  }

  function signInPrompt(body) {
    body.textContent = "";
    const msg = el("div", { class: "hgh-msg err" }, [
      el("div", { text: "You need to be signed in to use Keeper help." }),
    ]);
    msg.appendChild(el("div", { class: "hgh-hint" }, [
      "Members get the studio, media and chat surfaces. ",
      el("a", { href: loginHref, text: "Sign in" }),
      ".",
    ]));
    body.appendChild(msg);
  }

  // ---- Ask tab -----------------------------------------------------------
  function renderAsk(body, foot) {
    if (!history.length) {
      say(body, "keeper",
        "Ask me about anything on this site — an error you're seeing, how a "
        + "screen works, or what to try next. I can explain, but I can't change "
        + "anything: use “Request a fix” for that.");
    }
    history.forEach((m) => say(body, m.role === "user" ? "user" : "keeper", m.content));

    const input = el("textarea", {
      rows: 2, placeholder: "What's going wrong?", "aria-label": "Your question",
    });
    const send = el("button", { class: "hgh-btn", type: "button", text: busy ? "…" : "Send", disabled: busy });
    const ctxBox = el("input", { type: "checkbox" });
    ctxBox.checked = includeCtx;
    ctxBox.addEventListener("change", () => { includeCtx = ctxBox.checked; });

    const submit = () => {
      const text = input.value.trim();
      if (!text || busy) return;
      input.value = "";
      void ask(body, send, text);
    };
    send.addEventListener("click", submit);
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); }
    });

    foot.appendChild(input);
    foot.appendChild(el("div", { class: "hgh-row" }, [
      el("label", { class: "hgh-check" }, [ctxBox, el("span", { text: "Include page context" })]),
      el("span", { style: "flex:1" }),
      send,
    ]));
    foot.appendChild(el("div", { class: "hgh-hint", text: "Direct line to the hugpy keeper, grounded in live fleet state. Server-side actions go through “Request a fix”." }));

    // A prompt handed in by openHelpWidget: PRE-FILL, don't auto-send by
    // default. The person still sees the question, can edit it, and presses
    // Send — the widget never speaks for them unless the caller asked (send:true).
    if (pendingPrompt) {
      const preset = pendingPrompt;
      pendingPrompt = "";
      input.value = preset;
      if (pendingSend) {
        pendingSend = false;
        input.value = "";
        setTimeout(() => void ask(body, send, preset), 0);
      }
    }
    setTimeout(() => { input.focus(); input.setSelectionRange(input.value.length, input.value.length); }, 0);
  }

  async function ask(body, send, text) {
    busy = true;
    send.disabled = true;
    send.textContent = "…";
    const outgoing = includeCtx ? `${text}\n\n--- page context ---\n${pageContext()}` : text;
    history.push({ role: "user", content: text });
    say(body, "user", text);
    if (includeCtx) {
      const ctx = el("div", { class: "hgh-ctx", text: pageContext() });
      body.appendChild(ctx);
    }
    const bubble = say(body, "keeper", "asking the hugpy keeper…");
    let acc = "";
    try {
      // Direct line: one JSON round-trip to the keeper seat. The keeper reads
      // its MCT ledger + live fleet facts server-side; nothing is faked here.
      const resp = await fetch(`${apiBase}/keeper/help/ask`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: outgoing, history: history.slice(0, -1) }),
      });
      if (resp.status === 401) { signInPrompt(body); return; }
      if (!resp.ok) throw new Error(`${resp.status} ${(await resp.text()).slice(0, 200)}`);
      const data = await resp.json();
      acc = data.reply || "";
      if (data.offline) {
        acc = `[keeper brain offline — deterministic state readout]\n${acc}`;
      }
      bubble.textContent = acc || "the keeper returned no answer";
      if (!data.ok) bubble.className = "hgh-msg err";
      history.push({ role: "assistant", content: acc });
    } catch (e) {
      bubble.className = "hgh-msg err";
      bubble.textContent = `Could not reach the Keeper: ${(e && e.message) || e}`;
    } finally {
      busy = false;
      send.disabled = false;
      send.textContent = "Send";
      body.scrollTop = body.scrollHeight;
    }
  }

  // ---- Request-a-fix tab -------------------------------------------------
  function renderFix(body, foot) {
    body.appendChild(el("div", { class: "hgh-note" },
      "Destructive or uncertain actions are never performed from here. This "
      + "files your request with the Keeper for an OPERATOR to approve or reject "
      + "— nothing happens until they do."));

    const desc = el("textarea", { rows: 3, placeholder: "What's wrong / what do you need?", "aria-label": "Description" });
    const action = el("textarea", { rows: 2, placeholder: "Optional: the fix you think is needed", "aria-label": "Proposed action" });
    const ctxBox = el("input", { type: "checkbox" });
    ctxBox.checked = true;
    const status = el("div");

    body.appendChild(el("div", { class: "hgh-label", text: "Description" }));
    body.appendChild(desc);
    body.appendChild(el("div", { class: "hgh-label", text: "Proposed action" }));
    body.appendChild(action);
    body.appendChild(status);

    const submit = el("button", { class: "hgh-btn", type: "button", text: "Submit for approval" });
    submit.addEventListener("click", async () => {
      const description = desc.value.trim();
      if (!description) { desc.focus(); return; }
      submit.disabled = true;
      submit.textContent = "Submitting…";
      status.textContent = "";
      try {
        const resp = await fetch(`${apiBase}/keeper/help/report`, {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            description,
            proposed_action: action.value.trim(),
            page_url: typeof location !== "undefined" ? location.href : "",
            context: ctxBox.checked
              ? (surface ? `surface: ${surface}\n` : "") + pageContext()
              : "",
          }),
        });
        if (resp.status === 401) { signInPrompt(body); return; }
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(data.error || `${resp.status}`);
        desc.value = "";
        action.value = "";
        status.appendChild(el("div", { class: "hgh-msg ok" }, [
          el("div", { text: "Submitted for approval." }),
          el("div", { class: "hgh-hint", text: `ticket ${data.ticket_id || "?"} · status ${data.status || "pending"}` }),
        ]));
      } catch (e) {
        status.appendChild(el("div", { class: "hgh-msg err", text: `Could not file the request: ${(e && e.message) || e}` }));
      } finally {
        submit.disabled = false;
        submit.textContent = "Submit for approval";
        body.scrollTop = body.scrollHeight;
      }
    });

    foot.appendChild(el("div", { class: "hgh-row" }, [
      el("label", { class: "hgh-check" }, [ctxBox, el("span", { text: "Include page context" })]),
      el("span", { style: "flex:1" }),
      submit,
    ]));
  }

  // Expose the open/prefill verb to the rest of the page (see openHelpWidget).
  // Nothing else about this widget is reachable from outside — no send, no
  // ticket filing, no history read. Opening a panel with a question typed in
  // is the whole API surface.
  activeWidget = {
    open(o = {}) {
      tab = o.tab === "fix" ? "fix" : "ask";
      open = true;
      pendingPrompt = o.prompt ? String(o.prompt) : "";
      pendingSend = !!o.send;
      render();
      return true;
    },
  };

  render();
  document.body.appendChild(root);
  return () => {
    root.remove();
    activeWidget = null;
  };
}

/**
 * Open the mounted Help widget's Ask tab, optionally with a question already
 * typed into its input.
 *
 * Added 2026-08-06 for the /fleet screen's per-component "? Help" buttons: a
 * caller wants "open Keeper help asking about THIS block", and before this the
 * only entry point was the floating button, which opens empty. The prompt is
 * PRE-FILLED, not sent, unless `send:true` — the person reads and presses Send.
 *
 * @param {Object}  [opts]
 * @param {string}  [opts.prompt]  Question to place in the input.
 * @param {string}  [opts.tab]     "ask" (default) | "fix".
 * @param {boolean} [opts.send]    Send the prompt immediately (default false).
 * @returns {boolean} false when no widget is mounted on this page (the caller
 *                    can then fall back to its own affordance); true otherwise.
 */
export function openHelpWidget(opts = {}) {
  if (!activeWidget) return false;
  return activeWidget.open(opts);
}

export default mountHelpWidget;
