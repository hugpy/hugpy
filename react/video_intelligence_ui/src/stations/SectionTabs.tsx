// Shared responsive section layout for the Frames & Generate stations.
//
// A section has three logical parts: the persistent file/media intake bar
// (`addFile`), the config knobs (`options`), and the working UI (`components`).
//
//   Mobile (<62rem): a PERSISTENT compact add-file bar on top (never a tab),
//     then a two-state [ Components | Options ] toggle (reusing the .vi-gen-mode
//     segmented-control idiom with real tab semantics — role="tablist"/"tab"/
//     "tabpanel", aria-selected, aria-controls/aria-labelledby, roving tabindex,
//     arrow-key movement, `hidden` on the inactive panel), and ONE panel below
//     showing the selected state (default Components).
//
//   Desktop (>=62rem): no toggle — the working components on the LEFT (with the
//     compact add-file bar atop that column) and the ~17rem Options rail on the
//     RIGHT, both visible at once.
//
// When `options` is omitted (crop stations — their crop params live inline in
// the Components view), there is NO toggle strip on mobile and NO right rail on
// desktop: just the persistent add-file bar over the Components content.
//
// Slots may be plain nodes or render functions `(api) => node`, where `api`
// carries `isDesktop`. Pure layout wrapper — it owns no station logic; all
// per-station state lives in the station component.
import {
  useEffect,
  useId,
  useState,
  type KeyboardEvent,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";

/** The two toggle states shown below the persistent add-file bar. */
type ToggleId = "components" | "options";

export interface SectionApi {
  isDesktop: boolean;
}

type Slot = ReactNode | ((api: SectionApi) => ReactNode);

function resolveSlot(slot: Slot | undefined, api: SectionApi): ReactNode {
  if (typeof slot === "function") return (slot as (a: SectionApi) => ReactNode)(api);
  return slot ?? null;
}

function tabElId(base: string, id: ToggleId): string {
  return `${base}-tab-${id}`;
}
function panelElId(base: string, id: ToggleId): string {
  return `${base}-panel-${id}`;
}

// matchMedia-backed media query, SSR-safe (defaults to no-match / mobile when
// window/matchMedia are unavailable). Initial state reads synchronously on the
// client so the correct layout paints on the first frame (no toggle flash).
function useMediaQuery(query: string): boolean {
  const read = () =>
    typeof window !== "undefined" && typeof window.matchMedia === "function"
      ? window.matchMedia(query).matches
      : false;
  const [matches, setMatches] = useState<boolean>(read);
  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function")
      return;
    const mql = window.matchMedia(query);
    const onChange = () => setMatches(mql.matches);
    onChange();
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, [query]);
  return matches;
}

export function SectionTabs({
  ariaLabel,
  options,
  optionsLabel = "Options",
  optionsHost,
  addFile,
  components,
  componentsLabel = "Components",
  initialTab = "components",
}: {
  ariaLabel: string;
  /** Omit when the station has NO config knobs (crop stations) — then no
      [Components|Options] toggle (mobile) and no right rail (desktop) render. */
  options?: Slot;
  optionsLabel?: string;
  /** Round 9 (additive; default OFF): when a DOM node is passed, the options are
      PORTALED into it (the arm shell's shared sidebar "Settings" tab) instead of the
      desktop right rail / mobile [Components|Options] toggle — the work column then goes
      FULL WIDTH (the movie-mirror). Stations that don't pass this (e.g. Frames) keep the
      in-place rail/toggle EXACTLY as before. Null while the Settings tab is closed —
      the options simply don't render until it opens (the knob state lives in the caller). */
  optionsHost?: HTMLElement | null;
  /** The persistent compact add-file bar (never a tab). */
  addFile: Slot;
  components: Slot;
  componentsLabel?: string;
  /** Which toggle state opens active on mobile. Defaults to Components. */
  initialTab?: ToggleId;
}) {
  const base = useId();
  const isDesktop = useMediaQuery("(min-width: 62rem)");
  const hasOptions = options != null;
  // When the caller OPTS IN to external hosting (passes the optionsHost prop at all —
  // even null while the Settings tab is closed), the options live THERE (portaled), so
  // the in-place layout behaves like the no-options case (full-width work column / no
  // mobile toggle) REGARDLESS of whether the host node exists yet. Keying on "prop was
  // passed" (not "element exists") is what stops the right rail from flickering back
  // every time the Settings tab is closed. Stations that never pass the prop
  // (optionsHost === undefined, e.g. Frames) keep the in-place rail/toggle unchanged.
  const hostOptIn = hasOptions && optionsHost !== undefined;

  // Toggle order: [ Components | Options ] (Options on the right). Only used
  // when Options exist.
  const order: { id: ToggleId; label: string }[] = [
    { id: "components", label: componentsLabel },
    { id: "options", label: optionsLabel },
  ];

  const [active, setActive] = useState<ToggleId>(() =>
    order.some((o) => o.id === initialTab) ? initialTab : "components",
  );

  const api: SectionApi = { isDesktop };
  const optionsNode = hasOptions ? resolveSlot(options, api) : null;
  const addFileNode = resolveSlot(addFile, api);
  const componentsNode = resolveSlot(components, api);
  // Round 9: when a host is supplied, portal the options into it (the shell Settings
  // tab). Renders nothing until the host node exists (Settings tab closed = no target).
  const portalOptions =
    hostOptIn && optionsHost != null && optionsNode != null
      ? createPortal(optionsNode, optionsHost)
      : null;

  // ---- Desktop ----
  if (isDesktop) {
    // No options (or options portaled to the shell Settings tab): a single full-width
    // work column (add-file bar atop the components) — no empty right rail.
    if (!hasOptions || hostOptIn) {
      return (
        <div className="vi-section-tabs vi-section-tabs--stacked">
          <section
            className="vi-section-region vi-section-region--work"
            aria-label={componentsLabel}
          >
            <div className="vi-section-addfile">{addFileNode}</div>
            {componentsNode}
          </section>
          {portalOptions}
        </div>
      );
    }
    // Options: components (with the add-file bar atop) on the left, the Options
    // rail on the right. No toggle.
    return (
      <div className="vi-section-tabs vi-section-tabs--beside">
        <section
          className="vi-section-region vi-section-region--work"
          aria-label={componentsLabel}
        >
          <div className="vi-section-addfile">{addFileNode}</div>
          {componentsNode}
        </section>
        <section
          className="vi-section-region vi-section-region--options"
          aria-label={optionsLabel}
        >
          {optionsNode}
        </section>
      </div>
    );
  }

  // ---- Mobile, no options (or options portaled to the shell Settings tab): persistent
  // add-file bar over the components only — no [Components|Options] toggle. ----
  if (!hasOptions || hostOptIn) {
    return (
      <div className="vi-section-tabs vi-section-tabs--mobile">
        <div className="vi-section-addfile">{addFileNode}</div>
        {componentsNode}
        {portalOptions}
      </div>
    );
  }

  // ---- Mobile, with options: persistent add-file bar, then a
  // [ Components | Options ] toggle over a single panel. ----
  function focusTab(index: number) {
    const t = order[index];
    if (!t) return;
    setActive(t.id);
    document.getElementById(tabElId(base, t.id))?.focus();
  }

  function onKeyDown(e: KeyboardEvent<HTMLButtonElement>, index: number) {
    switch (e.key) {
      case "ArrowRight":
      case "ArrowDown":
        e.preventDefault();
        focusTab((index + 1) % order.length);
        break;
      case "ArrowLeft":
      case "ArrowUp":
        e.preventDefault();
        focusTab((index - 1 + order.length) % order.length);
        break;
      case "Home":
        e.preventDefault();
        focusTab(0);
        break;
      case "End":
        e.preventDefault();
        focusTab(order.length - 1);
        break;
      default:
        break;
    }
  }

  const nodeFor = (id: ToggleId): ReactNode =>
    id === "options" ? optionsNode : componentsNode;

  return (
    <div className="vi-section-tabs vi-section-tabs--mobile">
      {/* PERSISTENT add-file bar — always visible, not a tab. */}
      <div className="vi-section-addfile">{addFileNode}</div>

      {/* Components | Options toggle. */}
      <div
        role="tablist"
        aria-label={ariaLabel}
        className="vi-gen-mode vi-section-tablist"
      >
        {order.map((t, i) => {
          const selected = t.id === active;
          return (
            <button
              key={t.id}
              type="button"
              role="tab"
              id={tabElId(base, t.id)}
              aria-selected={selected}
              aria-controls={panelElId(base, t.id)}
              tabIndex={selected ? 0 : -1}
              className={`vi-gen-mode-btn vi-section-tab${
                selected ? " vi-gen-mode-btn-active" : ""
              }`}
              onClick={() => setActive(t.id)}
              onKeyDown={(e) => onKeyDown(e, i)}
            >
              {t.label}
            </button>
          );
        })}
      </div>

      {/* Single panel below the toggle. */}
      {order.map((t) => (
        <div
          key={t.id}
          role="tabpanel"
          id={panelElId(base, t.id)}
          aria-labelledby={tabElId(base, t.id)}
          hidden={t.id !== active}
          tabIndex={0}
          className="vi-section-tabpanel"
        >
          {nodeFor(t.id)}
        </div>
      ))}
    </div>
  );
}
