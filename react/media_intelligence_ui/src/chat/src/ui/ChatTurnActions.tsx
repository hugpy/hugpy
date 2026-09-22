/*
 * ChatTurnActions.tsx — under-message actions row.
 *
 * Icon-first ghost buttons. The switch-model, sources, and more buttons open
 * lightweight dropdown menus (close on outside-click / selection / Escape):
 *   - switch-model → re-run this turn on a different model
 *   - sources      → the files / analysis / link this answer drew on
 *   - more         → regenerate, delete
 */

import { useEffect, useRef, useState } from "react";
import { Icon } from "./Icons";

export interface TurnSource {
  label: string;
  href?: string;
}

interface ChatTurnActionsProps {
  copied: boolean;
  canCopy: boolean;
  canShare: boolean;
  busy?: boolean;
  models: { key: string; label: string }[];
  currentModelKey?: string;
  sources: TurnSource[];
  onCopy: () => void;
  onShare: () => void;
  onSwitchModel: (modelKey: string) => void;
  onRegenerate: () => void;
  onDelete: () => void;
}

type OpenMenu = "model" | "sources" | "more" | null;

export default function ChatTurnActions({
  copied,
  canCopy,
  canShare,
  busy = false,
  models,
  currentModelKey,
  sources,
  onCopy,
  onShare,
  onSwitchModel,
  onRegenerate,
  onDelete,
}: ChatTurnActionsProps): JSX.Element {
  const [open, setOpen] = useState<OpenMenu>(null);
  const rootRef = useRef<HTMLDivElement | null>(null);

  // Close on outside click / Escape.
  useEffect(() => {
    if (!open) return;
    function onDown(e: MouseEvent) {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        setOpen(null);
      }
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(null);
    }
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const hasSources = sources.length > 0;

  function toggle(menu: OpenMenu) {
    setOpen((cur) => (cur === menu ? null : menu));
  }

  return (
    <div
      ref={rootRef}
      className="relative flex flex-wrap items-center gap-0.5"
      aria-label="Response actions"
    >
      <GhostIconButton
        ariaLabel={copied ? "Copied" : "Copy response"}
        title={copied ? "Copied" : "Copy"}
        testid="copy-turn-action-button"
        disabled={!canCopy}
        onClick={onCopy}
      >
        <Icon name={copied ? "check" : "copy"} width={16} height={16} />
      </GhostIconButton>

      <GhostIconButton
        ariaLabel="Share"
        title="Share"
        testid="share-turn-action-button"
        disabled={!canShare}
        onClick={onShare}
      >
        <Icon name="share" width={16} height={16} />
      </GhostIconButton>

      <GhostIconButton
        ariaLabel="Switch model"
        title="Switch model"
        ariaHaspopup
        expanded={open === "model"}
        disabled={busy || models.length === 0}
        onClick={() => toggle("model")}
      >
        <Icon name="switch-model" width={16} height={16} />
      </GhostIconButton>

      <GhostIconButton
        ariaLabel="Sources"
        title={hasSources ? "Sources" : "No sources for this answer"}
        ariaHaspopup
        expanded={open === "sources"}
        disabled={!hasSources}
        onClick={() => toggle("sources")}
      >
        <Icon name="sources" width={16} height={16} />
      </GhostIconButton>

      <GhostIconButton
        ariaLabel="More actions"
        title="More"
        ariaHaspopup
        expanded={open === "more"}
        onClick={() => toggle("more")}
      >
        <Icon name="more" width={16} height={16} />
      </GhostIconButton>

      {open === "model" && (
        <Menu>
          <MenuHeader>Re-answer with…</MenuHeader>
          <div className="max-h-64 overflow-y-auto">
            {models.map((m) => (
              <MenuItem
                key={m.key}
                onClick={() => {
                  setOpen(null);
                  if (m.key !== currentModelKey) onSwitchModel(m.key);
                }}
              >
                <span className="truncate">{m.label}</span>
                {m.key === currentModelKey && (
                  <Icon name="check" width={14} height={14} />
                )}
              </MenuItem>
            ))}
          </div>
        </Menu>
      )}

      {open === "sources" && (
        <Menu>
          <MenuHeader>Sources</MenuHeader>
          {sources.map((s, i) =>
            s.href ? (
              <a
                key={i}
                href={s.href}
                target="_blank"
                rel="noreferrer"
                onClick={() => setOpen(null)}
                className="flex items-center gap-2 rounded-md px-2.5 py-1.5 text-[13px] text-token-text-primary hover:bg-token-surface-hover"
              >
                <span className="truncate">{s.label}</span>
              </a>
            ) : (
              <div
                key={i}
                className="flex items-center gap-2 rounded-md px-2.5 py-1.5 text-[13px] text-token-text-secondary"
              >
                <span className="truncate">{s.label}</span>
              </div>
            ),
          )}
        </Menu>
      )}

      {open === "more" && (
        <Menu>
          <MenuItem
            disabled={busy}
            onClick={() => {
              setOpen(null);
              onRegenerate();
            }}
          >
            <Icon name="switch-model" width={14} height={14} />
            <span>Regenerate</span>
          </MenuItem>
          <MenuItem
            destructive
            onClick={() => {
              setOpen(null);
              onDelete();
            }}
          >
            <Icon name="more" width={14} height={14} />
            <span>Delete</span>
          </MenuItem>
        </Menu>
      )}
    </div>
  );
}

function Menu({ children }: { children: React.ReactNode }): JSX.Element {
  return (
    <div
      role="menu"
      className="
        absolute left-0 top-full z-20 mt-1 min-w-[200px]
        rounded-xl border border-token-border-light p-1 shadow-lg
      "
      style={{ background: "var(--bg-elevated-primary)" }}
    >
      {children}
    </div>
  );
}

function MenuHeader({ children }: { children: React.ReactNode }): JSX.Element {
  return (
    <div className="px-2.5 py-1 text-[11px] uppercase tracking-wide text-token-text-tertiary">
      {children}
    </div>
  );
}

interface MenuItemProps {
  children: React.ReactNode;
  disabled?: boolean;
  destructive?: boolean;
  onClick: () => void;
}

function MenuItem({ children, disabled, destructive, onClick }: MenuItemProps): JSX.Element {
  return (
    <button
      type="button"
      role="menuitem"
      disabled={disabled}
      onClick={onClick}
      className={`
        flex w-full items-center justify-between gap-2 rounded-md px-2.5 py-1.5
        text-left text-[13px]
        hover:bg-token-surface-hover
        disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-transparent
        ${destructive ? "text-token-text-error" : "text-token-text-primary"}
      `}
      style={destructive ? { color: "var(--text-error, #ef4444)" } : undefined}
    >
      {children}
    </button>
  );
}

interface GhostIconButtonProps {
  ariaLabel: string;
  title?: string;
  testid?: string;
  disabled?: boolean;
  ariaHaspopup?: boolean;
  expanded?: boolean;
  onClick: () => void;
  children: React.ReactNode;
}

function GhostIconButton({
  ariaLabel,
  title,
  testid,
  disabled,
  ariaHaspopup,
  expanded,
  onClick,
  children,
}: GhostIconButtonProps): JSX.Element {
  return (
    <button
      type="button"
      aria-label={ariaLabel}
      title={title}
      data-testid={testid}
      aria-haspopup={ariaHaspopup ? "menu" : undefined}
      aria-expanded={ariaHaspopup ? (expanded ? "true" : "false") : undefined}
      data-state={ariaHaspopup ? (expanded ? "open" : "closed") : undefined}
      disabled={disabled}
      onClick={onClick}
      className="
        inline-flex h-8 w-8 items-center justify-center rounded-lg
        text-token-text-tertiary
        hover:bg-token-surface-hover hover:text-token-text-primary
        disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-transparent
        transition-colors duration-100
      "
    >
      {children}
    </button>
  );
}
