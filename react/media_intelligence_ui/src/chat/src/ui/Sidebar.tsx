/*
 * Sidebar.tsx — left rail.
 *
 *   - Header row: hugpy BrandMark (→ welcome screen) + close-sidebar.
 *   - Pinned items: New chat / Search chats / Library / Chat⇄Console.
 *   - Files section (this session's uploads; searchable, incl. doc full-text).
 *   - Recents: persisted conversation history (localStorage), searchable.
 *   - Profile pinned bottom → menu (clear all conversations).
 *
 * Collapsed renders a thin icon-only rail.
 */

import { useEffect, useRef, useState } from "react";
import { Icon } from "./Icons";
import type { StoredFile } from "../utilities/fileUpload";
import { fileKindGlyph } from "../utilities/fileUpload";
import type { Conversation } from "../utilities/chatHistory";

interface SidebarProps {
  collapsed: boolean;
  view: "chat" | "console";
  onToggle: () => void;
  onNewChat: () => void;
  onSelectChat: () => void;
  onSelectConsole: () => void;
  files: StoredFile[];
  onOpenFile: (id: string) => void;
  onDeleteFile: (id: string) => void;
  conversations: Conversation[];
  activeConversationId: string;
  onSelectConversation: (id: string) => void;
  onDeleteConversation: (id: string) => void;
  onClearAllConversations: () => void;
}

export default function Sidebar({
  collapsed,
  view,
  onToggle,
  onNewChat,
  onSelectChat,
  onSelectConsole,
  files,
  onOpenFile,
  onDeleteFile,
  conversations,
  activeConversationId,
  onSelectConversation,
  onDeleteConversation,
  onClearAllConversations,
}: SidebarProps): JSX.Element {
  const width = collapsed
    ? "var(--sidebar-rail-width)"
    : "var(--sidebar-width)";

  // Files search — matches name, kind, and (for documents) extracted full text.
  const [fileQuery, setFileQuery] = useState("");
  const fq = fileQuery.trim().toLowerCase();
  const filteredFiles = fq
    ? files.filter(
        (f) =>
          f.name.toLowerCase().includes(fq) ||
          f.kind.toLowerCase().includes(fq) ||
          (f.extractedText?.toLowerCase().includes(fq) ?? false),
      )
    : files;

  // Chat search — toggled by "Search chats"; matches title + message text.
  const [searchMode, setSearchMode] = useState(false);
  const [chatQuery, setChatQuery] = useState("");
  const searchRef = useRef<HTMLInputElement | null>(null);
  useEffect(() => {
    if (searchMode) searchRef.current?.focus();
  }, [searchMode]);

  const cq = chatQuery.trim().toLowerCase();
  const filteredConvos = (
    cq
      ? conversations.filter(
          (c) =>
            c.title.toLowerCase().includes(cq) ||
            c.messages.some(
              (m) =>
                (m.query ?? "").toLowerCase().includes(cq) ||
                (typeof m.response === "string"
                  ? m.response.toLowerCase().includes(cq)
                  : false),
            ),
        )
      : conversations
  )
    .slice()
    .sort((a, b) => (a.updatedAt < b.updatedAt ? 1 : -1));

  const [profileOpen, setProfileOpen] = useState(false);

  function scrollToFiles() {
    document
      .getElementById("sidebar-files")
      ?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  return (
    <aside
      id="stage-slideover-sidebar"
      className={`
        relative z-21 h-full shrink-0 overflow-hidden
        border-e border-token-border-extra-light
        bg-token-sidebar-surface-primary
        print:hidden
        transition-[width,transform] duration-150
        max-md:fixed max-md:inset-y-0 max-md:left-0 max-md:z-40
        max-md:!w-[var(--sidebar-width)] max-md:shadow-2xl
        max-md:bg-[var(--sidebar-surface-primary)]
        ${collapsed ? "max-md:-translate-x-full" : "max-md:translate-x-0"}
      `}
      style={{ width }}
      aria-label="Sidebar"
    >
      <div className="relative flex h-full flex-col">
        {/* ── header row ─────────────────────────────────────────── */}
        <div
          id="sidebar-header"
          className="flex items-center justify-between px-2 h-[var(--header-height)]"
        >
          {/* Brand mark removed — it now lives in the top nav, so showing it
              here too was a duplicate. */}
          <button
            type="button"
            onClick={onToggle}
            aria-controls="stage-slideover-sidebar"
            aria-expanded={!collapsed}
            aria-label={collapsed ? "Open sidebar" : "Close sidebar"}
            data-testid="close-sidebar-button"
            className="
              flex h-8 w-8 items-center justify-center rounded-md
              text-token-text-tertiary hover:bg-token-surface-hover outline-none ml-auto
            "
          >
            <Icon name="sidebar-toggle" width={17} height={17} />
          </button>
        </div>

        {/* ── pinned items ───────────────────────────────────────── */}
        <nav aria-label="Chat history" className="flex min-h-0 flex-1 flex-col">
          <div className={collapsed ? "px-1" : "px-2"}>
            <SidebarItem
              icon="new-chat"
              label="New chat"
              collapsed={collapsed}
              onClick={onNewChat}
              testid="sidebar-item-new-chat"
            />
            <SidebarItem
              icon="search"
              label="Search chats"
              collapsed={collapsed}
              active={searchMode}
              onClick={() => {
                if (collapsed) onToggle();
                setSearchMode((v) => !v);
              }}
              testid="sidebar-item-search"
            />
            <SidebarItem
              icon="images"
              label="Library"
              collapsed={collapsed}
              onClick={() => {
                if (collapsed) onToggle();
                scrollToFiles();
              }}
              testid="sidebar-item-library"
            />
            <SidebarItem
              icon={view === "console" ? "new-chat" : "sparkles"}
              label={view === "console" ? "Back to Chat" : "Tool Console"}
              collapsed={collapsed}
              onClick={view === "console" ? onSelectChat : onSelectConsole}
              active={view === "console"}
              testid="sidebar-item-view-toggle"
            />
          </div>

          {/* ── files section (this session's uploads) ───────────── */}
          {!collapsed && (
            <div id="sidebar-files" className="mt-2 flex min-h-0 flex-col">
              <div className="px-4 pt-3 pb-1">
                <h3 className="text-[11px] font-semibold uppercase tracking-wide text-token-text-tertiary">
                  Files
                </h3>
              </div>
              {files.length > 0 ? (
                <>
                  <div className="px-2 pb-1">
                    <input
                      type="search"
                      value={fileQuery}
                      onChange={(e) => setFileQuery(e.target.value)}
                      placeholder="Search files…"
                      aria-label="Search files"
                      className="
                        w-full rounded-md border border-token-border-light
                        bg-token-surface-secondary px-2 py-1 text-[12px]
                        text-token-text-primary outline-none
                        placeholder:text-token-text-tertiary
                      "
                    />
                  </div>
                  <div className="max-h-[30vh] overflow-y-auto px-2 pb-2">
                    {filteredFiles.map((f) => (
                      <div
                        key={f.id}
                        className="group/file flex items-center gap-1 rounded-md hover:bg-token-surface-hover"
                      >
                        <button
                          type="button"
                          onClick={() => onOpenFile(f.id)}
                          title={f.name}
                          data-file-id={f.id}
                          className="
                            flex min-w-0 flex-1 items-center gap-2 rounded-md px-2 py-1.5
                            text-left text-[12px] text-token-text-secondary
                          "
                        >
                          <span aria-hidden>{fileKindGlyph(f.kind)}</span>
                          <span className="flex-1 truncate">{f.name}</span>
                          {f.extractStatus === "extracting" && (
                            <span className="text-token-text-tertiary" title="indexing…">·</span>
                          )}
                        </button>
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            onDeleteFile(f.id);
                          }}
                          aria-label={`Delete "${f.name}"`}
                          title="Delete file (also removes it from the server)"
                          className="
                            mr-1 inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-md
                            text-token-text-tertiary opacity-0
                            hover:bg-token-surface-hover hover:text-token-text-primary
                            group-hover/file:opacity-100 focus:opacity-100
                          "
                        >
                          <span aria-hidden className="text-[14px] leading-none">×</span>
                        </button>
                      </div>
                    ))}
                    {filteredFiles.length === 0 && (
                      <div className="px-2 py-1 text-[12px] text-token-text-tertiary">
                        No matches.
                      </div>
                    )}
                  </div>
                </>
              ) : (
                <div className="px-4 pb-1 text-[12px] text-token-text-tertiary">
                  No files yet — attach with ＋.
                </div>
              )}
            </div>
          )}

          {/* ── recents section ─────────────────────────────────── */}
          {!collapsed && (
            <>
              <div className="px-4 pt-5 pb-1">
                <h3 className="text-[11px] font-semibold uppercase tracking-wide text-token-text-tertiary">
                  Recents
                </h3>
              </div>

              {searchMode && (
                <div className="px-2 pb-1">
                  <input
                    ref={searchRef}
                    type="search"
                    value={chatQuery}
                    onChange={(e) => setChatQuery(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Escape") {
                        setChatQuery("");
                        setSearchMode(false);
                      }
                    }}
                    placeholder="Search chats…"
                    aria-label="Search chats"
                    className="
                      w-full rounded-md border border-token-border-light
                      bg-token-surface-secondary px-2 py-1 text-[12px]
                      text-token-text-primary outline-none
                      placeholder:text-token-text-tertiary
                    "
                  />
                </div>
              )}

              <div className="flex-1 overflow-y-auto px-2 pb-2">
                {filteredConvos.length === 0 ? (
                  <div className="px-3 py-2 text-[12px] text-token-text-tertiary">
                    {conversations.length === 0
                      ? "No saved chats yet."
                      : "No matches."}
                  </div>
                ) : (
                  filteredConvos.map((c) => (
                    <div
                      key={c.id}
                      className={`
                        group/recent flex items-center gap-1 rounded-md
                        ${c.id === activeConversationId ? "bg-token-surface-hover" : ""}
                      `}
                    >
                      <button
                        type="button"
                        onClick={() => onSelectConversation(c.id)}
                        title={c.title}
                        className="
                          flex-1 truncate rounded-md px-3 py-2 text-left text-[13px]
                          text-token-text-primary hover:bg-token-surface-hover
                        "
                      >
                        {c.title}
                      </button>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          onDeleteConversation(c.id);
                        }}
                        aria-label={`Delete "${c.title}"`}
                        title="Delete chat"
                        className="
                          mr-1 inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-md
                          text-token-text-tertiary opacity-0
                          hover:bg-token-surface-hover hover:text-token-text-primary
                          group-hover/recent:opacity-100 focus:opacity-100
                        "
                      >
                        <Icon name="more" width={14} height={14} />
                      </button>
                    </div>
                  ))
                )}
              </div>
            </>
          )}
        </nav>

        {/* ── profile pinned bottom ──────────────────────────────── */}
        <div className={`relative ${collapsed ? "px-1 pb-2" : "px-2 pb-2"}`}>
          {profileOpen && !collapsed && (
            <div
              role="menu"
              className="
                absolute bottom-full left-2 right-2 mb-1 z-30
                rounded-xl border border-token-border-light p-1 shadow-lg
              "
              style={{ background: "var(--bg-elevated-primary)" }}
            >
              <div className="px-2.5 py-1 text-[11px] text-token-text-tertiary">
                {conversations.length} saved chat{conversations.length === 1 ? "" : "s"} · operator session
              </div>
              <button
                type="button"
                role="menuitem"
                disabled={conversations.length === 0}
                onClick={() => {
                  setProfileOpen(false);
                  onClearAllConversations();
                }}
                className="
                  flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-left text-[13px]
                  hover:bg-token-surface-hover
                  disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-transparent
                "
                style={{ color: "var(--text-error, #ef4444)" }}
              >
                Clear all conversations
              </button>
            </div>
          )}
          <SidebarItem
            icon="user"
            label="You"
            collapsed={collapsed}
            active={profileOpen}
            onClick={() => {
              if (collapsed) onToggle();
              else setProfileOpen((v) => !v);
            }}
            testid="accounts-profile-button"
          />
        </div>
      </div>
    </aside>
  );
}

interface SidebarItemProps {
  icon: string;
  label: string;
  collapsed: boolean;
  onClick: () => void;
  testid?: string;
  active?: boolean;
}

function SidebarItem({
  icon,
  label,
  collapsed,
  onClick,
  testid,
  active,
}: SidebarItemProps): JSX.Element {
  return (
    <button
      type="button"
      onClick={onClick}
      data-testid={testid}
      data-sidebar-item="true"
      aria-label={collapsed ? label : undefined}
      title={collapsed ? label : undefined}
      className={`
        __menu-item hoverable w-full text-left
        ${active ? "bg-token-surface-hover" : ""}
        ${collapsed ? "justify-center px-0" : ""}
      `}
    >
      <span className="icon"><Icon name={icon} /></span>
      {!collapsed && <span className="text-[13px] font-medium">{label}</span>}
    </button>
  );
}
