// Showroom theme registry — the canned demo doubles as a live style sandbox.
//
// The console is token-driven (src/index.css :root). A theme RE-DECLARES those
// tokens scoped to the showroom wrapper's `.sr-theme-<id>` class (cascading to
// the whole console without touching any component), optionally loads web fonts,
// and adds scoped overrides for the few panels that still hardcode colors.
// Add a theme = add an object here. `?theme=<id>` selects it; the switcher in the
// demo banner persists the choice.

export const DEFAULT_THEME = 'default'

export const THEMES = [
  {
    id: 'default',
    label: 'hugpy',
    fonts: [],
    vars: {},
    css: '',
  },
  {
    id: 'voyager',
    label: 'Voyager II',
    fonts: [
      'https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,300;0,6..72,500;0,6..72,700;1,6..72,400&family=JetBrains+Mono:wght@400;500;700&display=swap',
    ],
    vars: {
          "--bg": "#0e0d0a",
          "--surface": "#181612",
          "--surface-1": "#1d1a15",
          "--surface-2": "#211d18",
          "--surface-3": "#2a251d",
          "--border": "#2c2820",
          "--border-strong": "#423a2c",
          "--muted": "#807866",
          "--text": "#e8e3d6",
          "--bright": "#f5f0e4",
          "--accent": "#d4a857",
          "--accent-dim": "#6b5530",
          "--green": "#8faa5e",
          "--yellow": "#e0a44a",
          "--red": "#e25b3a",
          "--blue": "#6ea3c4",
          "--purple": "#d4a857",
          "--cyan": "#d4a857",
          "--success": "#8faa5e",
          "--danger": "#e25b3a",
          "--warning": "#e0a44a",
          "--radius": "4px",
          "--radius-lg": "8px",
          "--font-mono": "'JetBrains Mono', ui-monospace, monospace"
    },
    css: `/* ══════════════════════════════════════════════════════════════════════════
   VOYAGER II — warm-dark · gold · archival showroom theme
   All selectors scoped under .sr-theme-voyager. Token colors handled by 'vars'.
   ══════════════════════════════════════════════════════════════════════════ */

/* ── (a) global field: warm document ground + faint gold/cool nebulae ─────── */
.sr-theme-voyager {
  background: #0e0d0a;
  min-height: 100%;
  background-image:
    radial-gradient(ellipse 1200px 600px at 80% -10%, rgba(212, 168, 87, 0.06), transparent 60%),
    radial-gradient(ellipse 800px 500px at -10% 110%, rgba(110, 163, 196, 0.04), transparent 60%);
  background-attachment: fixed;
}

/* ── (b) serif display accents (Newsreader) — data/tables stay JetBrains Mono ─ */
/* brand/logo + section titles + panel § headings -> serif */
.sr-theme-voyager .hugpy-navbar-brand,
.sr-theme-voyager .brandmark-word,
.sr-theme-voyager .brandmark-sub,
.sr-theme-voyager .landing-brand,
.sr-theme-voyager .docs-brand,
.sr-theme-voyager .landing-section-title,
.sr-theme-voyager .section-title,
.sr-theme-voyager .cc-h1,
.sr-theme-voyager .docs-h3,
.sr-theme-voyager .docs-h3-plain,
.sr-theme-voyager .mt-serve-title,
.sr-theme-voyager .wp-title,
.sr-theme-voyager .sp-title,
.sr-theme-voyager .chat-title,
.sr-theme-voyager .pb-title,
.sr-theme-voyager .kc-title,
.sr-theme-voyager .br-title,
.sr-theme-voyager .dc-title,
.sr-theme-voyager .rp-title {
  font-family: 'Newsreader', Georgia, 'Times New Roman', serif;
  letter-spacing: 0.01em;
  font-weight: 500;
}

/* italic for display headings — where it reads well (brand wordmark + big titles) */
.sr-theme-voyager .hugpy-navbar-brand,
.sr-theme-voyager .brandmark-word,
.sr-theme-voyager .landing-brand,
.sr-theme-voyager .landing-section-title,
.sr-theme-voyager .section-title,
.sr-theme-voyager .cc-h1,
.sr-theme-voyager .wp-title,
.sr-theme-voyager .sp-title,
.sr-theme-voyager .chat-title {
  font-style: italic;
  font-weight: 400;
}

/* tab labels — serif but upright + tracked for legibility at small size */
.sr-theme-voyager .tabbar .tab {
  font-family: 'Newsreader', Georgia, serif;
  font-style: normal;
  font-weight: 500;
  letter-spacing: 0.02em;
}
.sr-theme-voyager .tabbar .tab.tab-active { font-weight: 700; }

/* hard-keep data / tables / code / counters / secrets in JetBrains Mono */
.sr-theme-voyager .model-table,
.sr-theme-voyager .hf-results-table,
.sr-theme-voyager .docs-table,
.sr-theme-voyager .table-wrap,
.sr-theme-voyager .wp-token-secret,
.sr-theme-voyager .tab-badge,
.sr-theme-voyager .section-count,
.sr-theme-voyager code,
.sr-theme-voyager pre,
.sr-theme-voyager kbd {
  font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, monospace;
  font-style: normal;
  letter-spacing: normal;
}

/* ── WorkersPanel · voyager showroom overrides (hardcoded colors only) ─────── */

/* online dot glow — green glow → sage (bg var(--green) handled globally) */
.sr-theme-voyager .wp-online .wp-dot {
  box-shadow: 0 0 6px rgba(143, 170, 94, 0.7);
}

/* ping result chips — green/red → sage/hot */
.sr-theme-voyager .wp-ping-ok {
  background: rgba(143, 170, 94, 0.1);
  border-color: rgba(143, 170, 94, 0.35);
}
.sr-theme-voyager .wp-ping-bad {
  background: rgba(226, 91, 58, 0.1);
  border-color: rgba(226, 91, 58, 0.35);
}

/* prune (destructive) — red → hot */
.sr-theme-voyager .wp-prune {
  background: rgba(226, 91, 58, 0.1);
  border-color: rgba(226, 91, 58, 0.35);
}
.sr-theme-voyager .wp-prune:hover:not(:disabled) {
  background: rgba(226, 91, 58, 0.2);
}

/* install card outline — green → sage */
.sr-theme-voyager .wp-install {
  border-color: rgba(143, 170, 94, 0.3);
}

/* fleet "loaded" rollup pill — amber */
.sr-theme-voyager .wp-fleet-loaded {
  background: rgba(224, 164, 74, 0.08);
  border-color: rgba(224, 164, 74, 0.35);
}

/* worker role badge — purple → gold accent (text color is var(--purple), global) */
.sr-theme-voyager .wp-role {
  background: rgba(212, 168, 87, 0.1);
  border-color: rgba(212, 168, 87, 0.35);
}

/* free-all VRAM control — blue → cool */
.sr-theme-voyager .wp-free-all {
  background: rgba(110, 163, 196, 0.1);
  border-color: rgba(110, 163, 196, 0.35);
}
.sr-theme-voyager .wp-free-all:hover:not(:disabled) {
  background: rgba(110, 163, 196, 0.2);
}

/* per-GPU VRAM bar free portion — green → sage (fill gradient is token-based) */
.sr-theme-voyager .wp-vram-bar {
  background: rgba(143, 170, 94, 0.18);
}

/* per-model state pills + borders — amber / cool */
.sr-theme-voyager .wp-pill-serving {
  background: rgba(224, 164, 74, 0.08);
  border-color: rgba(224, 164, 74, 0.35);
}
.sr-theme-voyager .wp-pill-pulling {
  background: rgba(110, 163, 196, 0.08);
  border-color: rgba(110, 163, 196, 0.35);
}
.sr-theme-voyager .wp-model.wp-st-serving {
  border-color: rgba(224, 164, 74, 0.4);
}
.sr-theme-voyager .wp-model.wp-st-pulling {
  border-color: rgba(110, 163, 196, 0.4);
}

/* admission gate pills — solid green/olive/red → tinted sage/amber/hot */
.sr-theme-voyager .wp-adm-pill-approved {
  background: rgba(143, 170, 94, 0.18);
  color: #a8c47a;
}
.sr-theme-voyager .wp-adm-pill-pending {
  background: rgba(224, 164, 74, 0.16);
  color: #e0a44a;
}
.sr-theme-voyager .wp-adm-pill-blocked {
  background: rgba(226, 91, 58, 0.16);
  color: #e8917a;
}
.sr-theme-voyager .wp-worker.wp-adm-pending {
  outline-color: #e0a44a;
}

/* admit / block / pending action chips */
.sr-theme-voyager .wp-admit {
  background: rgba(143, 170, 94, 0.2);
  color: #a8c47a;
}
.sr-theme-voyager .wp-block {
  background: rgba(226, 91, 58, 0.2);
  color: #e8917a;
}
.sr-theme-voyager .wp-pending-chip {
  background: rgba(224, 164, 74, 0.16);
  color: #e0a44a;
}

/* enrollment tokens — amber accents + warm darks */
.sr-theme-voyager .wp-token-new {
  border-color: #e0a44a;
  background: #2a251d;
}
.sr-theme-voyager .wp-token-secret {
  background: #0e0d0a;
}
.sr-theme-voyager .wp-token-revoke {
  background: rgba(226, 91, 58, 0.2);
  color: #e8917a;
}
.sr-theme-voyager .wp-token-state {
  color: #e8917a;
}

/* SlotsPanel — voyager remap of HARDCODED colors only.
   (var(--text/--border/--surface-N/--muted/--accent/--red) are global tokens — left untouched) */

/* greens (#3fb950 online/ready/serving) -> sage #8faa5e */
.sr-theme-voyager .sp-st-serving { border-left-color: #8faa5e; }
.sr-theme-voyager .sp-pill-serving { color: #8faa5e; }
.sr-theme-voyager .sp-dedicated { color: #8faa5e; border-color: #8faa5e; }
.sr-theme-voyager .sp-unit-up { color: #8faa5e; border-color: #8faa5e; }
.sr-theme-voyager .sp-tag-gen { color: #8faa5e !important; border-color: #8faa5e !important; }

/* ambers/yellows (#d29922 loading/warning) -> amber #e0a44a */
.sr-theme-voyager .sp-st-loading { border-left-color: #e0a44a; }
.sr-theme-voyager .sp-pill-loading { color: #e0a44a; }
.sr-theme-voyager .sp-loading-tag { color: #e0a44a; border-color: #e0a44a; }
.sr-theme-voyager .sp-slot-progress-bar { background: #e0a44a; }
.sr-theme-voyager .sp-warn-banner { border-color: #e0a44a; background: rgba(224, 164, 74, 0.10); }
.sr-theme-voyager .sp-tag-warn { color: #e0a44a !important; border-color: #e0a44a !important; }

/* ───────── ChatPanel — hardcoded-color → voyager remaps ───────── */

/* streaming-status pill: green tint (76,175,80) → sage (143,170,94) */
.sr-theme-voyager .chat-status.is-streaming {
  background: rgba(143, 170, 94, 0.12);
}

/* user message bubble: cyan tint (86,212,232) → gold (212,168,87) */
.sr-theme-voyager .msg-user .msg-content {
  background: rgba(212, 168, 87, 0.09);
  border-color: rgba(212, 168, 87, 0.22);
}

/* error message bubble: red tint (248,113,113) → hot (226,91,58) */
.sr-theme-voyager .msg-error .msg-content {
  background: rgba(226, 91, 58, 0.07);
  border-color: rgba(226, 91, 58, 0.25);
}

/* send button: dark ink on gold + cyan hover → brighter gold */
.sr-theme-voyager .btn-send {
  color: #0e0d0a;
}
.sr-theme-voyager .btn-send:hover:not(:disabled) {
  background: #e6c074;
  border-color: #e6c074;
}

/* stop button: dark ink on hot + light-red hover → light hot */
.sr-theme-voyager .btn-stop {
  color: #0e0d0a;
}
.sr-theme-voyager .btn-stop:hover:not(:disabled) {
  background: #ef7d5f;
  border-color: #ef7d5f;
}

/* ── voyager re-skin: showroom demo banner ─────────────────────────────── */
.sr-theme-voyager .sr-banner {
  color: #e8e3d6;
  background: linear-gradient(90deg, #211d18 0%, #181612 100%);
  border-bottom: 1px solid #423a2c;
}
.sr-theme-voyager .sr-badge {
  color: #181612;
  background: #e0a44a;
}
.sr-theme-voyager .sr-text { color: #b8b1a0; }
.sr-theme-voyager .sr-text strong { color: #e8e3d6; }
.sr-theme-voyager .sr-text em { color: #e0a44a; }

.sr-theme-voyager .sr-cmd {
  border: 1px solid #423a2c;
  background: #0e0d0a;
  color: #e8e3d6;
}
.sr-theme-voyager .sr-cmd:hover { border-color: #6b5530; background: #181612; }
.sr-theme-voyager .sr-cmd code { color: #8faa5e; }
.sr-theme-voyager .sr-cmd-ico { color: #807866; }

.sr-theme-voyager .sr-connect {
  color: #181612;
  background: #8faa5e;
}
.sr-theme-voyager .sr-connect:hover { background: #a8c47a; }

/* ── voyager re-skin: showroom toasts ──────────────────────────────────── */
.sr-theme-voyager .sr-toast {
  color: #e8e3d6;
  background: #211d18;
  border: 1px solid #423a2c;
  border-left: 3px solid #e0a44a;
}

/* ── ModelTable: hardcoded badge/tag tints + danger hovers ─────────────── */
.sr-theme-voyager .badge-green  { background: rgba(143, 170, 94, 0.10); border-color: rgba(143, 170, 94, 0.25); }
.sr-theme-voyager .badge-red    { background: rgba(226, 91, 58, 0.08);  border-color: rgba(226, 91, 58, 0.20); }
.sr-theme-voyager .badge-yellow { background: rgba(224, 164, 74, 0.10); border-color: rgba(224, 164, 74, 0.25); }
.sr-theme-voyager .badge-blue   { background: rgba(110, 163, 196, 0.10); border-color: rgba(110, 163, 196, 0.25); }
.sr-theme-voyager .fw-llama_cpp    { background: rgba(224, 164, 74, 0.10); }
.sr-theme-voyager .fw-transformers { background: rgba(110, 163, 196, 0.10); }
.sr-theme-voyager .actions-menu .menu-danger:hover:not(:disabled) { background: rgba(226, 91, 58, 0.12); }
.sr-theme-voyager .mt-act-danger:hover:not(:disabled) { background: rgba(226, 91, 58, 0.12); }

/* ── HFSearch: hardcoded framework-tag borders + pull button ───────────── */
.sr-theme-voyager .hf-fw.fw-transformers { border-color: rgba(110, 163, 196, 0.40); }
.sr-theme-voyager .hf-fw.fw-llama_cpp    { border-color: rgba(224, 164, 74, 0.40); }
.sr-theme-voyager .hf-fw.fw-dataset      { border-color: rgba(212, 168, 87, 0.40); }
.sr-theme-voyager .btn-pull { border-color: rgba(212, 168, 87, 0.35); }
.sr-theme-voyager .btn-pull:hover:not(:disabled) { color: #0e0d0a; }

/* ── PeersBar: recolor status-dot glows (preserve the glow) ────────────── */
.sr-theme-voyager .peer-dot { box-shadow: 0 0 5px rgba(143, 170, 94, 0.60); }
.sr-theme-voyager .peer-offline .peer-dot { box-shadow: 0 0 5px rgba(226, 91, 58, 0.60); }

/* ── Navbar: hardcoded translucent dark backdrop ───────────────────────── */
.sr-theme-voyager .navbar { background: rgba(14, 13, 10, 0.85); }`,
  },
]

export function getTheme(id) {
  return THEMES.find((t) => t.id === id) || THEMES[0]
}

/** Combined CSS for a theme: the token block + its scoped extras. */
export function themeStyle(theme) {
  const entries = Object.entries(theme.vars || {})
  const varBlock = entries.length
    ? `.sr-theme-${theme.id} {\n${entries.map(([k, v]) => `  ${k}: ${v};`).join('\n')}\n}`
    : ''
  return `${varBlock}\n${theme.css || ''}`.trim()
}
