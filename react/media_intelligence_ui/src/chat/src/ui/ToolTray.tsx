// Persistent tool tray next to the composer: the full /ml catalog (+ text-gen)
// from the live page registry, capability-grayed via GET /ml. Each tool is a
// TOGGLE (pre-selection), not an execute-on-click button: clicking highlights it,
// and the selected tools run as part of the NEXT normal chat send (on the turn's
// input), overriding the auto-router. Selections are sticky across sends until
// toggled off. File/url tools still toggle; if there's no matching input on send
// they error gracefully (toolCanRun guards in the send path).
import { useEffect, useState } from "react";
import { pagesByCategory } from "../../../console/src/imports/pages/pagesRegistry";
import type { PageSpec } from "../../../console/src/imports/pages/pageSpec";
import { loadMlCapabilities, pageReady, type MlCap } from "../utilities/capability";

interface ToolTrayProps {
  // Currently pre-selected tool keys (spec.key). Highlighted in the tray.
  selected: Set<string>;
  // Toggle a tool's pre-selection on/off.
  onToggle: (spec: PageSpec) => void;
}

export default function ToolTray({ selected, onToggle }: ToolTrayProps): JSX.Element | null {
  const [caps, setCaps] = useState<Map<string, MlCap>>(new Map());

  useEffect(() => {
    const c = new AbortController();
    loadMlCapabilities(c.signal).then(setCaps);
    return () => c.abort();
  }, []);

  const groups = pagesByCategory();
  const specs: PageSpec[] = Object.values(groups).flat();
  if (specs.length === 0) return null;

  return (
    <div
      className="hugpy-tool-tray flex flex-wrap gap-1.5 px-0.5 py-1.5 max-sm:flex-nowrap max-sm:overflow-x-auto [&::-webkit-scrollbar]:hidden"
      style={{ scrollbarWidth: "none" }}
    >
      {specs.map((spec) => {
        const cap = pageReady(spec.path, caps);
        const disabled = !cap.ready;
        const on = selected.has(spec.key);
        return (
          <button
            key={spec.key}
            type="button"
            disabled={disabled}
            aria-pressed={on}
            title={
              disabled
                ? `needs abstract_hugpy_dev[${cap.extra}]`
                : on
                  ? `“${spec.title}” will run when you send — click to remove`
                  : `Include “${spec.title}” in your next send`
            }
            onClick={() => onToggle(spec)}
            style={{
              fontSize: 12,
              padding: "3px 10px",
              borderRadius: 999,
              border: `1px solid ${on ? "#1d4ed8" : "var(--border, rgba(127,127,127,0.35))"}`,
              opacity: disabled ? 0.4 : 1,
              cursor: disabled ? "not-allowed" : "pointer",
              // Keep chips intact in the mobile single-row scroller (no squashing).
              flexShrink: 0,
              whiteSpace: "nowrap",
              // Hardcoded selected colors so a toggled-on chip is always legible.
              background: on ? "#2563eb" : "transparent",
              color: on ? "#ffffff" : "inherit",
              fontWeight: on ? 600 : 400,
            }}
          >
            {spec.title}
          </button>
        );
      })}
    </div>
  );
}
