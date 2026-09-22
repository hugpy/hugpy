/*
 * ThreadHeader.tsx — top bar above the thread.
 *
 * Convo's actual header is sparse: model-switcher button on the left
 * (the model name + a chevron-down), nothing on the right for our
 * use case. Sticky to the top, blends into the page bg.
 *
 * The button itself is the dropdown trigger — we surface the prop
 * but leave the actual menu wiring to whoever bolts on the model
 * picker later. The model selector in the controls strip below the
 * composer is the canonical edit point for now.
 */

import { Icon } from "./Icons";

interface ThreadHeaderProps {
  modelLabel: string;
  sidebarCollapsed: boolean;
  plainTitle?: string;
  onReopenSidebar: () => void;
}

export default function ThreadHeader({
  modelLabel,
  sidebarCollapsed,
  plainTitle,
  onReopenSidebar,
}: ThreadHeaderProps): JSX.Element {
  return (
    <header
      className="
        sticky top-0 z-20 relative
        flex items-center justify-between
        h-[var(--header-height)] px-2
        bg-token-main-surface-primary
      "
    >
      {/* A static title (e.g. the product name) is centered across the whole
          header bar, independent of the side controls. */}
      {plainTitle && (
        <span
          className="
            pointer-events-none absolute left-1/2 top-1/2
            -translate-x-1/2 -translate-y-1/2 whitespace-nowrap
            text-[24px] font-medium tracking-[-0.015em] text-token-text-primary
          "
          style={{
            fontFamily:
              '"Newsreader", Georgia, "Times New Roman", serif',
          }}
        >
          {plainTitle}
        </span>
      )}

      <div className="flex items-center gap-1">
        {sidebarCollapsed && (
          <button
            type="button"
            onClick={onReopenSidebar}
            aria-label="Open sidebar"
            className="
              flex h-9 w-9 items-center justify-center rounded-lg
              text-token-text-primary hover:bg-token-surface-hover
            "
          >
            {/* Hamburger on mobile (the sidebar is an off-canvas drawer there);
                the panel-toggle glyph on desktop. */}
            <span className="md:hidden"><Icon name="menu" /></span>
            <span className="hidden md:inline"><Icon name="sidebar-toggle" /></span>
          </button>
        )}

        {!plainTitle && (
          <button
            type="button"
            aria-haspopup="menu"
            aria-expanded="false"
            data-state="closed"
            data-testid="model-switcher-button"
            className="
              inline-flex items-center gap-1 px-3 h-9 rounded-lg
              text-[16px] font-semibold text-token-text-primary
              hover:bg-token-surface-hover
              transition-colors duration-100
            "
            onClick={() =>
              console.info("[ThreadHeader] model switcher (stub)")
            }
          >
            <span className="truncate max-w-[260px]">{modelLabel}</span>
            <Icon name="chevron-down" width={16} height={16} />
          </button>
        )}
      </div>

      <div
        className="flex items-center gap-1"
        data-testid="thread-header-right-actions-container"
      >
        {/* placeholder for share / settings */}
      </div>
    </header>
  );
}
