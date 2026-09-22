// Public overview of the portable hugpy-agent runtime. Keep this page focused
// on the stable operator path; implementation history belongs in the repo.
import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
// The top-nav link SET + order come from the ONE shared manifest so this arm
// can't drift from the rest of the site. Plain .js, imported via allowJs.
import { buildNavItems } from "../../ui_shared/navbar/links";
// The ONE Keeper help widget (mounted for this arm in agents_ui/entry.tsx).
// `openHelpWidget` opens that already-mounted panel with a question typed in —
// there is deliberately no second chat client on this page; the button reuses
// the widget's existing /api/chat/stream path.
import { openHelpWidget } from "../../ui_shared/help/helpWidget";

const ico = {
  width: 20,
  height: 20,
  viewBox: "0 0 24 24",
  fill: "none" as const,
  stroke: "currentColor",
  strokeWidth: 1.6,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
};

const LoopIcon = () => (
  <svg {...ico}>
    <path d="M21 12a9 9 0 1 1-3-6.7" />
    <path d="M21 3v6h-6" />
  </svg>
);
const ToolIcon = () => (
  <svg {...ico}>
    <path d="M14.7 6.3a4 4 0 0 0-5.4 5.4L4 17v3h3l5.3-5.3a4 4 0 0 0 5.4-5.4l-2.5 2.5-2-2 2.5-2.5z" />
  </svg>
);
const MemoryIcon = () => (
  <svg {...ico}>
    <path d="M6 4h9l3 3v13a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1z" />
    <path d="M9 9h6M9 13h6M9 17h3" />
  </svg>
);
const ShieldIcon = () => (
  <svg {...ico}>
    <path d="M12 3l7 3v6c0 4.5-3 8-7 9-4-1-7-4.5-7-9V6l7-3z" />
    <path d="M9.5 12l1.8 1.8L15 10" />
  </svg>
);
const BranchIcon = () => (
  <svg {...ico}>
    <circle cx="6" cy="6" r="2.2" />
    <circle cx="6" cy="18" r="2.2" />
    <circle cx="18" cy="12" r="2.2" />
    <path d="M6 8.2V15.8M8 6.8 16 11M8 17.2 16 13" />
  </svg>
);
const NodesIcon = () => (
  <svg {...ico}>
    <circle cx="5" cy="7" r="2" />
    <circle cx="19" cy="7" r="2" />
    <circle cx="12" cy="17" r="2" />
    <path d="M6.6 8.4 10.6 15.4M17.4 8.4 13.4 15.4M7 7h10" />
  </svg>
);

type CardProps = { icon: React.ReactNode; title: string; body: string };
function Card({ icon, title, body }: CardProps) {
  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.03] p-5 transition hover:border-white/20 hover:bg-white/[0.05]">
      <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-white/10 bg-black/40 text-neutral-300">
        {icon}
      </div>
      <h3 className="mt-4 text-[15px] font-medium text-white">{title}</h3>
      <p className="mt-1.5 text-sm leading-relaxed text-neutral-400">{body}</p>
    </div>
  );
}

const CAPABILITIES: CardProps[] = [
  {
    icon: <LoopIcon />,
    title: "Assess → act → observe, crash-safe",
    body:
      "Every message and tool call is journaled to SQLite (WAL) before it runs. Kill the process mid-task and `resume` replays the journal — completed side effects aren't re-run, nothing is lost.",
  },
  {
    icon: <ToolIcon />,
    title: "Workspace-jailed tools + the fleet's ML suite",
    body:
      "Local tools (files, shell, http) are rooted to the workspace and risk-classed. Fleet tools ride the same API you already have: summarize, embed, transcribe, vision, and async image/scene generation — with capacity gaps surfaced as data, never masked.",
  },
  {
    icon: <MemoryIcon />,
    title: "Plain markdown memory",
    body:
      "Long-term notes live as greppable, human-editable markdown files in the workspace — one fact per file plus an index. No vector database to run.",
  },
  {
    icon: <ShieldIcon />,
    title: "Choose the policy for the job",
    body:
      "Use readonly for inspection, auto for trusted work, or ask when a Discord operator session is configured. Every tool call still passes through the same policy and audit path.",
  },
  {
    icon: <BranchIcon />,
    title: "Run once or stay online",
    body:
      "Use `run` for one task, `chat` for an interactive session, or `serve` to consume a local queue, Discord inbox, or centrally dispatched work.",
  },
  {
    icon: <NodesIcon />,
    title: "A node fleet, not just a CLI",
    body:
      "`serve --node` enrolls a box with central — register, heartbeat, pull dispatched tasks — so an operator can address a roster of remote agents from one console, the same pattern hugpy already uses for GPU workers.",
  },
];

const INSTALL_CMD = `cd path/to/hugpy_agent
python3 -m venv "$HOME/hugpy-agent/venv"
"$HOME/hugpy-agent/venv/bin/pip" install --upgrade .

export HUGPY_BASE=https://dev.hugpy.ai/api
export HUGPY_API_KEY=hp_…
export HUGPY_MODEL=Qwen~Qwen3-Coder-Next-GGUF
export HUGPY_WORKSPACE="$HOME/hugpy-agent/workspace"

"$HOME/hugpy-agent/venv/bin/hugpy-agent" run --policy auto \
  "Use fs_write to create hello.txt containing: Hello from hugpy-agent"`;

// ───────────────────────────────────────────────────────────────────────────
// "Your fleet" — the SIGNED-IN half of this page (2026-08-06 member rollout).
//
// Everything below this comment talks to the member-gated agent routes:
//   POST /api/agent/install-links   mint a scoped install link for MY machine
//   GET  /api/agent/install-links   MY links only (the server scopes by owner)
//   GET  /api/agent/console/info    what fleet-console artifacts exist
//   GET  /api/agent/console/<file>  the .deb itself
// All four answer 401 to an anonymous caller, so 401 is not an error here — it
// is the "you are a visitor" state, and the section collapses to a sign-in
// prompt pointing at the main SPA's /login. The brochure below stays visible
// either way: this page is still the public pitch.
//
// The install-link URL and its per-OS commands are rendered STRICTLY from what
// the mint returned (`url`, `commands`). They are never rebuilt client-side —
// agent_routes._install_commands owns that string shape on purpose ("no
// consumer hand-builds this"), which is also why the LIST rows (which carry no
// url/commands) show metadata only.
// ───────────────────────────────────────────────────────────────────────────
const API = "/api";

type InstallLink = {
  link_id: string;
  label: string;
  scopes?: string[];
  status?: string;
  created_at?: number;
  expires_at?: number | null;
  max_uses?: number;
  uses_left?: number;
  url?: string;
  commands?: Record<string, string>;
  downloads?: Record<string, string>;
};

type ConsoleArtifact = {
  filename: string;
  size_bytes?: number;
  sha256?: string;
  url?: string;
} | null;

/** A bot credential generated by POST /api/discord/bot-links. `api_key`, `env`,
 *  `commands` and `notes` come back on the MINT only — the list rows are
 *  metadata (the raw key exists exactly once, in the mint response). */
type BotLink = {
  kind: string;
  title?: string;
  key_id: string;
  label?: string;
  owner?: string | null;
  scopes?: string[];
  prefix?: string;
  created_at?: number;
  expires_at?: number | null;
  status?: string;
  api_key?: string;
  api_base?: string;
  invite_url?: string;
  client_id_configured?: boolean;
  env?: string;
  commands?: Record<string, string>;
  notes?: string[];
};

// ───────────────────────────────────────────────────────────────────────────
// Per-component DOCS + HELP (2026-08-06).
//
// Every block in "Your fleet" carries the same two affordances:
//   * "Docs ↗"  — the specific docs page/anchor for THAT component, not the
//                 page-level /docs#agents link the hero already has.
//   * "? Help"  — opens the Keeper help widget with the block's question typed
//                 in. It is the SAME widget entry.tsx mounts (one chat client,
//                 one /api/chat/stream path); this only opens it prefilled.
//
// The prompts are FULL LITERALS, not built by interpolation, so the exact
// string a person sees is the exact string that ships in the bundle (and is
// greppable in dist/ during verification).
// ───────────────────────────────────────────────────────────────────────────
const HELP_PROMPTS = {
  agentInstall: "how can i install the hugpy agent",
  fleetConsole: "how can i install fleet-console",
  discordBot: "how can i install the discord bot",
  hugpyDiscordBot: "how can i install the hugpy discord bot",
} as const;

const DOCS_HREFS = {
  agentInstall: "/docs#agents/install-link",
  fleetConsole: "/docs#agents/fleet-console",
  discordBot: "/docs#console-discord/bot-own",
  hugpyDiscordBot: "/docs#console-discord/bot-hugpy",
} as const;

/** The Docs ↗ link + ? Help button every block on this screen carries.
 *  `marker` is a stable data attribute so the built bundle can be verified. */
function BlockTools({
  marker,
  docsHref,
  helpPrompt,
}: {
  marker: string;
  docsHref: string;
  helpPrompt: string;
}) {
  return (
    <div data-fleet-block={marker} className="mt-3 flex flex-wrap items-center gap-2">
      <a
        href={docsHref}
        target="_blank"
        rel="noreferrer"
        className="rounded-md border border-white/15 px-2 py-1 text-xs text-neutral-300 transition hover:border-white/30 hover:bg-white/5"
      >
        Docs ↗
      </a>
      <button
        type="button"
        aria-label={helpPrompt}
        onClick={() => {
          // openHelpWidget returns false when no widget is mounted on the page
          // (it never mounts one itself). Falling back to the docs beats a
          // button that silently does nothing.
          if (!openHelpWidget({ prompt: helpPrompt })) window.location.href = docsHref;
        }}
        className="rounded-md border border-white/15 px-2 py-1 text-xs text-neutral-300 transition hover:border-white/30 hover:bg-white/5"
      >
        ? Help
      </button>
    </div>
  );
}

class Unauthorized extends Error {}

async function memberFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${API}${path}`, { credentials: "include", ...(init || {}) });
  if (resp.status === 401) throw new Unauthorized("sign-in required");
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    const err = data as { error?: string; description?: string };
    throw new Error(err.error || err.description || `${resp.status}`);
  }
  return data as T;
}

function bytes(n?: number): string {
  if (!n && n !== 0) return "—";
  if (n < 1024) return `${n} B`;
  const units = ["KB", "MB", "GB"];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
  return `${v.toFixed(1)} ${units[i]}`;
}

function when(ts?: number | null): string {
  if (!ts) return "—";
  try { return new Date(ts * 1000).toLocaleString(); } catch { return "—"; }
}

/** Copy-to-clipboard button. Falls back to selecting nothing rather than
 *  throwing where the Clipboard API is unavailable (http:// origins). */
function CopyButton({ value, label = "Copy" }: { value: string; label?: string }) {
  const [done, setDone] = useState(false);
  return (
    <button
      type="button"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(value);
          setDone(true);
          setTimeout(() => setDone(false), 1500);
        } catch {
          setDone(false);
        }
      }}
      className="shrink-0 rounded-md border border-white/15 px-2 py-1 text-xs text-neutral-300 transition hover:border-white/30 hover:bg-white/5"
    >
      {done ? "Copied" : label}
    </button>
  );
}

function CommandBlock({ title, command }: { title: string; command: string }) {
  return (
    <div className="mt-3">
      <div className="flex items-center gap-2">
        <span className="text-xs uppercase tracking-wide text-neutral-500">{title}</span>
        <span className="flex-1" />
        <CopyButton value={command} />
      </div>
      <pre className="mt-1 overflow-x-auto rounded-lg border border-white/10 bg-black/60 p-3 text-[12px] leading-relaxed text-neutral-300">
        <code>{command}</code>
      </pre>
    </div>
  );
}

// ───────────────────────────────────────────────────────────────────────────
// The two BOT generators (2026-08-06).
//
// One backend generator, two targets — POST /api/discord/bot-links {kind}:
//   kind="discord-bot"       your OWN Discord bot application talking to hugpy
//   kind="hugpy-discord-bot" the hugpy bot arm (`hugpy bot`) run by you
// Both mint the same artifact: a scoped, owner-recorded hugpy API key plus the
// .env / commands that target needs. What the server will NOT issue — and what
// this UI therefore never promises — is the shared DISCORD_TOKEN or a
// channel-scoped comms session; see discord_routes' bot-links block for why.
//
// `invite_url` is present only where the deployment has a Discord application
// id configured. When it is absent the server says so in `notes`, and we render
// that note verbatim rather than inventing a link that would 404.
// ───────────────────────────────────────────────────────────────────────────
function BotLinkBlock({
  kind,
  heading,
  blurb,
  marker,
  docsHref,
  helpPrompt,
  defaultLabel,
  onUnauthorized,
}: {
  kind: "discord-bot" | "hugpy-discord-bot";
  heading: string;
  blurb: string;
  marker: string;
  docsHref: string;
  helpPrompt: string;
  defaultLabel: string;
  onUnauthorized: () => void;
}) {
  const [label, setLabel] = useState(defaultLabel);
  const [links, setLinks] = useState<BotLink[]>([]);
  const [minted, setMinted] = useState<BotLink | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    try {
      const data = await memberFetch<{ links: BotLink[] }>("/discord/bot-links");
      setLinks((data.links || []).filter((l) => l.kind === kind));
    } catch (e) {
      if (e instanceof Unauthorized) { onUnauthorized(); return; }
      setError((e as Error).message);
    }
  }, [kind, onUnauthorized]);

  useEffect(() => { void load(); }, [load]);

  const generate = async () => {
    setBusy(true);
    setError("");
    try {
      // `scopes` is omitted deliberately: the server default (["v1"]) is the
      // product surface a bot needs, and a member mint is clamped to
      // ("v1","ml") anyway — asking for more is a 403, not a downgrade.
      const link = await memberFetch<BotLink>("/discord/bot-links", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kind, label: label.trim() || defaultLabel }),
      });
      setMinted(link);
      void load();
    } catch (e) {
      if (e instanceof Unauthorized) { onUnauthorized(); return; }
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  // Revoke kills the key immediately; the ledger hides revoked keys, so the
  // row disappears from this list too (2026-08-13 — the backend DELETE
  // existed all along, this is its first UI).
  const revoke = async (l: BotLink) => {
    if (!window.confirm(`Revoke "${l.label || l.key_id}"? A bot using this key stops working immediately.`)) return;
    try {
      await memberFetch(`/discord/bot-links/${encodeURIComponent(l.key_id || "")}`, { method: "DELETE" });
      void load();
    } catch (e) {
      if (e instanceof Unauthorized) { onUnauthorized(); return; }
      setError((e as Error).message);
    }
  };

  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.03] p-5">
      <h3 className="text-[15px] font-medium text-white">{heading}</h3>
      <p className="mt-1.5 text-sm leading-relaxed text-neutral-400">{blurb}</p>
      <BlockTools marker={marker} docsHref={docsHref} helpPrompt={helpPrompt} />

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <input
          value={label}
          onChange={(e) => setLabel(e.target.value)}
          placeholder="label (e.g. my-bot)"
          aria-label={`${heading} label`}
          className="min-w-[10rem] flex-1 rounded-lg border border-white/15 bg-black/50 px-3 py-2 text-sm text-neutral-200 outline-none focus:border-blue-400"
        />
        <button
          type="button"
          onClick={() => void generate()}
          disabled={busy}
          className="rounded-lg bg-white px-4 py-2 text-sm font-medium text-black transition hover:bg-neutral-200 disabled:opacity-50"
        >
          {busy ? "Generating…" : "Generate"}
        </button>
      </div>

      {error && (
        <p className="mt-3 rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-300">
          {error}
        </p>
      )}

      {minted && (
        <div className="mt-5 rounded-lg border border-emerald-500/25 bg-emerald-500/[0.06] p-4">
          <div className="text-sm text-emerald-200">
            Generated — the key below is shown once and belongs to your account
            {minted.scopes && minted.scopes.length > 0
              ? ` (scopes: ${minted.scopes.join(", ")})`
              : ""}
            .
          </div>
          {minted.api_key && (
            <div className="mt-3 flex items-center gap-2">
              <code className="flex-1 overflow-x-auto rounded-md border border-white/10 bg-black/60 px-2 py-1.5 text-[12px] text-neutral-300">
                {minted.api_key}
              </code>
              <CopyButton value={minted.api_key} label="Copy key" />
            </div>
          )}
          {minted.env && <CommandBlock title=".env" command={minted.env} />}
          {minted.commands &&
            Object.entries(minted.commands).map(([name, cmd]) => (
              <CommandBlock key={name} title={name} command={cmd} />
            ))}
          {minted.invite_url && (
            <div className="mt-3 text-sm">
              <a
                href={minted.invite_url}
                target="_blank"
                rel="noreferrer"
                className="text-blue-400 underline underline-offset-4 hover:text-blue-300"
              >
                Invite the bot to your server ↗
              </a>
            </div>
          )}
          {minted.notes && minted.notes.length > 0 && (
            <ul className="mt-3 space-y-1.5 text-xs leading-relaxed text-neutral-400">
              {minted.notes.map((n) => (
                <li key={n}>· {n}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      <div className="mt-5">
        <div className="text-xs uppercase tracking-wide text-neutral-500">
          your keys ({links.length})
        </div>
        {links.length === 0 ? (
          <p className="mt-2 text-sm text-neutral-500">None yet.</p>
        ) : (
          <ul className="mt-2 divide-y divide-white/5">
            {links.map((l) => (
              <li key={l.key_id} className="flex flex-wrap items-baseline gap-x-3 py-2 text-sm">
                <span className="text-neutral-200">{l.label || l.title}</span>
                <span className="text-xs text-neutral-500">{(l.scopes || []).join(", ")}</span>
                <span className="flex-1" />
                <span className="text-xs text-neutral-500">
                  {l.prefix ? `${l.prefix}…` : ""} · {when(l.created_at)}
                </span>
                <span
                  className={
                    l.status === "active"
                      ? "rounded-full border border-emerald-500/30 px-2 py-0.5 text-[11px] text-emerald-300"
                      : "rounded-full border border-white/10 px-2 py-0.5 text-[11px] text-neutral-500"
                  }
                >
                  {l.status || "?"}
                </span>
                <button
                  type="button"
                  onClick={() => void revoke(l)}
                  className="rounded-full border border-red-500/30 px-2 py-0.5 text-[11px] text-red-300 transition hover:border-red-400/60 hover:text-red-200"
                >
                  revoke
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

function SignInPrompt() {
  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.03] px-5 py-6 text-center">
      <h3 className="text-base font-medium text-white">Sign in to set up your fleet</h3>
      <p className="mx-auto mt-2 max-w-md text-sm leading-relaxed text-neutral-400">
        Install links and the fleet-console download belong to your account. Sign
        in with your hugpy membership to mint a link for your machine.
      </p>
      <a
        href="/login"
        className="mt-4 inline-block rounded-lg bg-white px-4 py-2 text-sm font-medium text-black transition hover:bg-neutral-200"
      >
        Sign in
      </a>
    </div>
  );
}

function YourFleet() {
  const [authed, setAuthed] = useState<boolean | null>(null); // null = checking
  const [links, setLinks] = useState<InstallLink[]>([]);
  const [minted, setMinted] = useState<InstallLink | null>(null);
  const [label, setLabel] = useState("my-machine");
  const [minting, setMinting] = useState(false);
  const [error, setError] = useState("");
  const [deb, setDeb] = useState<ConsoleArtifact>(null);
  const [whl, setWhl] = useState<ConsoleArtifact>(null);
  const [consoleError, setConsoleError] = useState("");

  const loadLinks = useCallback(async () => {
    try {
      const data = await memberFetch<{ links: InstallLink[] }>("/agent/install-links");
      setLinks(data.links || []);
      setAuthed(true);
    } catch (e) {
      if (e instanceof Unauthorized) { setAuthed(false); return; }
      setAuthed(true);
      setError((e as Error).message);
    }
  }, []);

  const loadConsole = useCallback(async () => {
    try {
      const data = await memberFetch<{ deb: ConsoleArtifact; agent_whl: ConsoleArtifact }>(
        "/agent/console/info",
      );
      setDeb(data.deb || null);
      setWhl(data.agent_whl || null);
      setConsoleError("");
    } catch (e) {
      if (e instanceof Unauthorized) { setAuthed(false); return; }
      setConsoleError((e as Error).message);
    }
  }, []);

  useEffect(() => { void loadLinks(); void loadConsole(); }, [loadLinks, loadConsole]);

  const mint = async () => {
    setMinting(true);
    setError("");
    try {
      // `label` is required by the route. Scopes are omitted deliberately: the
      // server's default (["v1"]) is right for an agent box, and a member mint
      // is clamped to ("v1","ml") anyway — asking for more is a 403, not a
      // downgrade.
      const link = await memberFetch<InstallLink>("/agent/install-links", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label: label.trim() || "my-machine" }),
      });
      setMinted(link);
      void loadLinks();
    } catch (e) {
      if (e instanceof Unauthorized) { setAuthed(false); return; }
      setError((e as Error).message);
    } finally {
      setMinting(false);
    }
  };

  // 2026-08-13: members can now clean their own ledger. Revoke kills an
  // ACTIVE link (and its undelivered key); remove purges the row — a
  // used-up link's already-installed machine keeps working.
  const revokeLink = async (l: InstallLink) => {
    if (!window.confirm(`Revoke install link "${l.label}"? Its key stops working too.`)) return;
    try {
      await memberFetch(`/agent/install-links/${encodeURIComponent(l.link_id)}`, { method: "DELETE" });
      void loadLinks();
    } catch (e) {
      if (e instanceof Unauthorized) { setAuthed(false); return; }
      setError((e as Error).message);
    }
  };

  const removeLink = async (l: InstallLink) => {
    const warn = l.status === "active"
      ? `Remove ACTIVE install link "${l.label}"? Its unused key dies with it.`
      : `Remove install link "${l.label}" from the list? (An already-installed machine keeps working.)`;
    if (!window.confirm(warn)) return;
    try {
      await memberFetch(`/agent/install-links/${encodeURIComponent(l.link_id)}?purge=1`, { method: "DELETE" });
      void loadLinks();
    } catch (e) {
      if (e instanceof Unauthorized) { setAuthed(false); return; }
      setError((e as Error).message);
    }
  };

  if (authed === null) {
    return (
      <section className="mx-auto max-w-4xl px-6 pb-6 sm:px-10">
        <div className="rounded-xl border border-white/10 bg-white/[0.02] px-5 py-4 text-sm text-neutral-500">
          Checking your account…
        </div>
      </section>
    );
  }

  return (
    <section className="mx-auto max-w-4xl px-6 pb-6 sm:px-10">
      <h2 className="mb-4 text-center text-xl font-medium tracking-tight text-white">Your fleet</h2>
      {!authed ? (
        <SignInPrompt />
      ) : (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {/* ---- install link ---- */}
          <div className="rounded-xl border border-white/10 bg-white/[0.03] p-5">
            <h3 className="text-[15px] font-medium text-white">Your agent install link</h3>
            <p className="mt-1.5 text-sm leading-relaxed text-neutral-400">
              One-time link carrying a key scoped to your account. Paste the command
              for your OS on the box you want to enroll.
            </p>
            <BlockTools
              marker="agent-install-link"
              docsHref={DOCS_HREFS.agentInstall}
              helpPrompt={HELP_PROMPTS.agentInstall}
            />
            <div className="mt-4 flex flex-wrap items-center gap-2">
              <input
                value={label}
                onChange={(e) => setLabel(e.target.value)}
                placeholder="label (e.g. laptop)"
                aria-label="Install link label"
                className="min-w-[10rem] flex-1 rounded-lg border border-white/15 bg-black/50 px-3 py-2 text-sm text-neutral-200 outline-none focus:border-blue-400"
              />
              <button
                type="button"
                onClick={() => void mint()}
                disabled={minting}
                className="rounded-lg bg-white px-4 py-2 text-sm font-medium text-black transition hover:bg-neutral-200 disabled:opacity-50"
              >
                {minting ? "Generating…" : "Generate my agent install link"}
              </button>
            </div>
            {error && (
              <p className="mt-3 rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-300">
                {error}
              </p>
            )}

            {minted && (
              <div className="mt-5 rounded-lg border border-emerald-500/25 bg-emerald-500/[0.06] p-4">
                <div className="text-sm text-emerald-200">
                  Link minted — it is shown once and is good for{" "}
                  {minted.max_uses ?? 1} use{(minted.max_uses ?? 1) === 1 ? "" : "s"}.
                </div>
                {minted.url && (
                  <div className="mt-3 flex items-center gap-2">
                    <code className="flex-1 overflow-x-auto rounded-md border border-white/10 bg-black/60 px-2 py-1.5 text-[12px] text-neutral-300">
                      {minted.url}
                    </code>
                    <CopyButton value={minted.url} label="Copy link" />
                  </div>
                )}
                {minted.commands &&
                  Object.entries(minted.commands).map(([os, cmd]) => (
                    <CommandBlock key={os} title={os} command={cmd} />
                  ))}
                {minted.downloads && Object.keys(minted.downloads).length > 0 && (
                  <div className="mt-3 flex flex-wrap items-center gap-3 text-sm">
                    <span className="text-xs uppercase tracking-wide text-neutral-500">
                      downloads
                    </span>
                    {Object.entries(minted.downloads).map(([kind, href]) => (
                      <a
                        key={kind}
                        href={href}
                        className="text-blue-400 underline underline-offset-4 hover:text-blue-300"
                      >
                        {kind}
                      </a>
                    ))}
                  </div>
                )}
              </div>
            )}

            <div className="mt-5">
              <div className="text-xs uppercase tracking-wide text-neutral-500">
                your links ({links.length})
              </div>
              {links.length === 0 ? (
                <p className="mt-2 text-sm text-neutral-500">None yet.</p>
              ) : (
                <ul className="mt-2 divide-y divide-white/5">
                  {links.map((l) => (
                    <li key={l.link_id} className="flex flex-wrap items-baseline gap-x-3 py-2 text-sm">
                      <span className="text-neutral-200">{l.label}</span>
                      <span className="text-xs text-neutral-500">{(l.scopes || []).join(", ")}</span>
                      <span className="flex-1" />
                      <span className="text-xs text-neutral-500">
                        {l.uses_left ?? "?"}/{l.max_uses ?? "?"} left · {when(l.created_at)}
                      </span>
                      <span
                        className={
                          l.status === "active"
                            ? "rounded-full border border-emerald-500/30 px-2 py-0.5 text-[11px] text-emerald-300"
                            : "rounded-full border border-white/10 px-2 py-0.5 text-[11px] text-neutral-500"
                        }
                      >
                        {l.status || "?"}
                      </span>
                      {l.status === "active" && (
                        <button
                          type="button"
                          onClick={() => void revokeLink(l)}
                          className="rounded-full border border-red-500/30 px-2 py-0.5 text-[11px] text-red-300 transition hover:border-red-400/60 hover:text-red-200"
                        >
                          revoke
                        </button>
                      )}
                      <button
                        type="button"
                        onClick={() => void removeLink(l)}
                        title="Remove this row from the list"
                        className="rounded-full border border-white/15 px-2 py-0.5 text-[11px] text-neutral-400 transition hover:border-white/40 hover:text-neutral-200"
                      >
                        remove
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>

          {/* ---- fleet-console download ---- */}
          <div className="rounded-xl border border-white/10 bg-white/[0.03] p-5">
            <h3 className="text-[15px] font-medium text-white">Download fleet-console</h3>
            <p className="mt-1.5 text-sm leading-relaxed text-neutral-400">
              The desktop console for your fleet, as a Debian package. Verify the
              SHA-256 after downloading.
            </p>
            <BlockTools
              marker="fleet-console-deb"
              docsHref={DOCS_HREFS.fleetConsole}
              helpPrompt={HELP_PROMPTS.fleetConsole}
            />
            {consoleError && (
              <p className="mt-3 rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-300">
                {consoleError}
              </p>
            )}
            {!consoleError && !deb && (
              <p className="mt-4 text-sm text-neutral-500">
                No fleet-console package is staged on this deployment right now.
              </p>
            )}
            {deb && (
              <div className="mt-4 space-y-2 text-sm">
                <div className="flex flex-wrap items-baseline gap-x-3">
                  <span className="text-neutral-200">{deb.filename}</span>
                  <span className="text-xs text-neutral-500">{bytes(deb.size_bytes)}</span>
                </div>
                {deb.sha256 && (
                  <div className="flex items-center gap-2">
                    <code className="flex-1 overflow-x-auto rounded-md border border-white/10 bg-black/60 px-2 py-1.5 text-[11px] text-neutral-400">
                      sha256 {deb.sha256}
                    </code>
                    <CopyButton value={deb.sha256} />
                  </div>
                )}
                <a
                  href={`${API}/agent/console/${deb.filename}`}
                  className="mt-2 inline-block rounded-lg bg-white px-4 py-2 text-sm font-medium text-black transition hover:bg-neutral-200"
                >
                  Download fleet-console (.deb)
                </a>
              </div>
            )}
            {whl && (
              <div className="mt-5 border-t border-white/10 pt-4 text-sm">
                <div className="text-xs uppercase tracking-wide text-neutral-500">
                  paired agent wheel
                </div>
                <div className="mt-1 flex flex-wrap items-baseline gap-x-3">
                  <a
                    href={`${API}/agent/console/${whl.filename}`}
                    className="text-blue-400 underline underline-offset-4 hover:text-blue-300"
                  >
                    {whl.filename}
                  </a>
                  <span className="text-xs text-neutral-500">{bytes(whl.size_bytes)}</span>
                </div>
              </div>
            )}
          </div>

          {/* ---- your own Discord bot ---- */}
          <BotLinkBlock
            kind="discord-bot"
            heading="Connect your Discord bot"
            blurb="Credentials for a Discord bot you own, talking to this deployment as you. You bring the bot token from the Discord Developer Portal; hugpy issues the account-scoped API key it calls with."
            marker="discord-bot-link"
            docsHref={DOCS_HREFS.discordBot}
            helpPrompt={HELP_PROMPTS.discordBot}
            defaultLabel="my-discord-bot"
            onUnauthorized={() => setAuthed(false)}
          />

          {/* ---- the hugpy-discord bot arm ---- */}
          <BotLinkBlock
            kind="hugpy-discord-bot"
            heading="Run the hugpy discord bot"
            blurb="The hugpy-side bot arm (`hugpy bot`), run by you against this fleet. Generates the HUGPY_API_KEY that instance needs plus its .env and run lines — never the shared bot token."
            marker="hugpy-discord-bot-link"
            docsHref={DOCS_HREFS.hugpyDiscordBot}
            helpPrompt={HELP_PROMPTS.hugpyDiscordBot}
            defaultLabel="my-hugpy-bot"
            onUnauthorized={() => setAuthed(false)}
          />
        </div>
      )}
    </section>
  );
}

export default function FleetOverview() {
  return (
    <main className="min-h-dvh bg-[#050505] text-neutral-200">
      {/* ---- nav ---- */}
      {/* Link SET + order come from the ONE shared manifest
          (ui_shared/navbar/links.js) so the fleet arm shows the SAME five links
          as the rest of the site (previously it was missing Media + Video).
          Rendering stays local to this arm's Tailwind styling. Fleet is the
          current surface; Docs keeps its `#agents` deep-anchor via hrefByKey. */}
      {/* Sticky at the top for parity with the rest of the site — the fleet arm
          has no demo banner, so the nav alone is the fixed header (operator
          directive 2026-07-21). The window scrolls; the nav pins with a solid
          backdrop so content passes under it cleanly. */}
      <nav className="sticky top-0 z-20 flex items-center justify-between border-b border-white/10 bg-[#050505]/85 px-6 py-4 backdrop-blur sm:px-10">
        <a href="/" className="text-sm font-medium tracking-tight text-white">
          hugpy
        </a>
        <div className="flex items-center gap-5 text-sm text-neutral-400">
          {buildNavItems({
            currentKey: "fleet",
            hrefByKey: { docs: "/docs#agents" },
          }).map((item) => (
            <a
              key={item.key}
              href={item.href}
              aria-current={item.current ? "page" : undefined}
              className={item.current ? "text-white" : "hover:text-white"}
            >
              {item.label}
            </a>
          ))}
        </div>
      </nav>

      {/* ---- hero ---- */}
      <section className="mx-auto max-w-4xl px-6 pb-14 pt-16 text-center sm:px-10 sm:pt-24">
        <div className="mx-auto mb-5 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-neutral-400">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
          the agent side of the fleet
        </div>
        <h1 className="text-4xl font-medium tracking-tight text-white sm:text-5xl">
          A portable agent, running on <span className="text-neutral-400">your</span> fleet
        </h1>
        <p className="mx-auto mt-5 max-w-2xl text-balance text-base leading-relaxed text-neutral-400 sm:text-lg">
          <code className="rounded bg-white/10 px-1.5 py-0.5 text-[0.9em] text-neutral-200">hugpy-agent</code>{" "}
          is a portable runtime that turns your hugpy fleet into the inference brain of an
          agentic system — install it on any box, no local GPU required. It reasons, calls
          tools, remembers, and asks before it does anything irreversible.
        </p>
        <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
          <a
            href="/docs#agents"
            className="rounded-lg bg-white px-4 py-2.5 text-sm font-medium text-black transition hover:bg-neutral-200"
          >
            Read the docs
          </a>
          <a
            href="#install"
            className="rounded-lg border border-white/15 px-4 py-2.5 text-sm font-medium text-white transition hover:border-white/30 hover:bg-white/5"
          >
            Install
          </a>
          <a
            href="/console"
            className="rounded-lg border border-white/15 px-4 py-2.5 text-sm font-medium text-white transition hover:border-white/30 hover:bg-white/5"
          >
            Open the console
          </a>
        </div>
      </section>

      {/* ---- signed-in member actions; the brochure continues below ---- */}
      <YourFleet />

      {/* ---- nomenclature strip: distinct from the GPU worker fleet ---- */}
      <section className="mx-auto max-w-4xl px-6 pb-4 sm:px-10">
        <div className="rounded-xl border border-white/10 bg-white/[0.02] px-5 py-4 text-sm leading-relaxed text-neutral-400">
          Not the same thing as a <span className="text-neutral-200">GPU worker</span> — a worker
          lends a box's GPU to the fleet for inference. An <span className="text-neutral-200">agent</span> is
          the other end of that relationship: a runtime that <em>uses</em> the fleet's inference to
          get work done, on a box that may have no GPU at all.
        </div>
      </section>

      {/* ---- capability grid ---- */}
      <section className="mx-auto max-w-5xl px-6 py-14 sm:px-10">
        <h2 className="text-center text-xl font-medium tracking-tight text-white">
          What the runtime gives you
        </h2>
        <div className="mt-8 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {CAPABILITIES.map((c) => (
            <Card key={c.title} {...c} />
          ))}
        </div>
      </section>

      {/* ---- install ---- */}
      <section id="install" className="mx-auto max-w-3xl scroll-mt-10 px-6 py-14 sm:px-10">
        <h2 className="text-center text-xl font-medium tracking-tight text-white">
          Install, configure, run
        </h2>
        <p className="mx-auto mt-3 max-w-xl text-center text-sm leading-relaxed text-neutral-400">
          The client is a small Python package. Install it in its own venv, point it at a ready
          model on your fleet, and run a task. No GPU or local model engine is required on the
          agent box.
        </p>
        <pre className="mt-6 overflow-x-auto rounded-xl border border-white/10 bg-black/60 p-4 text-left text-[13px] leading-relaxed text-neutral-300">
          <code>{INSTALL_CMD}</code>
        </pre>
        <p className="mt-4 text-center text-sm text-neutral-500">
          Full walkthrough, node mode, and troubleshooting live in{" "}
          <a className="text-blue-400 underline underline-offset-4 hover:text-blue-300" href="/docs#agents/enroll">
            the docs
          </a>
          .
        </p>
      </section>

      {/* ---- deployment modes ---- */}
      <section className="mx-auto max-w-4xl px-6 pb-16 sm:px-10">
        <div className="flex flex-col items-center gap-3 rounded-xl border border-dashed border-white/15 bg-white/[0.02] px-6 py-10 text-center">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg border border-white/10 bg-black/40 text-neutral-400">
            <NodesIcon />
          </div>
          <h3 className="text-base font-medium text-white">Keep the same runtime online</h3>
          <p className="max-w-md text-sm leading-relaxed text-neutral-400">
            Once a one-off run works, use <code>serve</code> with a local queue, Discord inbox,
            or node mode. The service uses the same workspace, model, tools, journal, and policy.
          </p>
          <a
            href="/docs#agents/node-mode"
            className="mt-1 text-sm text-blue-400 underline underline-offset-4 hover:text-blue-300"
          >
            Choose a service mode →
          </a>
        </div>
      </section>

      <footer className="border-t border-white/10 px-6 py-8 text-center text-xs text-neutral-600 sm:px-10">
        <a className="hover:text-neutral-400" href="/">
          hugpy.ai
        </a>{" "}
        · inference you own.
      </footer>
    </main>
  );
}
