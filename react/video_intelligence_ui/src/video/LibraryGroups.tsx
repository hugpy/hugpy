// SESSION LIBRARY GROUPS (operator ask 2026-08-13): the library pickers render
// as TYPE-CATEGORIZED, COLLAPSIBLE sections instead of one flat grid — inputs
// (uploads/frames/send-to-studio), generated images, scene outputs, movie
// outputs, studio clips. ONE component for every library surface (the house
// one-reusable-component rule); the call site supplies the item renderer so
// pick semantics stay exactly what they were. Groups with no members render
// nothing; `defaultOpen` seeds which sections start expanded (the current
// section rides in via a `key` remount on mode change).
import { useRef, useState, type ReactNode } from "react";
import type { LibraryItem, LibraryGenKind } from "./mediaLibrary";

const GROUPS: ReadonlyArray<{
  id: string;
  label: string;
  kinds: ReadonlyArray<LibraryGenKind | null>;
}> = [
  { id: "inputs", label: "📤 Inputs — uploads · frames · sent to studio", kinds: [null] },
  { id: "images", label: "🖼 Generated images", kinds: ["generate_image"] },
  { id: "scenes", label: "🎬 Scene outputs", kinds: ["generate_scene"] },
  { id: "movies", label: "🎞 Movie outputs", kinds: ["generate_movie"] },
  { id: "clips", label: "📹 Studio clips", kinds: ["studio_i2v"] },
];

export function LibraryGroups({
  items,
  defaultOpen,
  seed,
  renderItem,
}: {
  items: LibraryItem[];
  /** Group ids expanded on first render — e.g. ["inputs", "images"]. */
  defaultOpen: readonly string[];
  /** Re-seed `defaultOpen` when this changes (the active section) — the scoped
   *  tsconfig rejects `key` on locally-declared components, so remount-by-key
   *  is not available here; this is the React render-phase reset pattern. */
  seed?: string;
  renderItem: (it: LibraryItem) => ReactNode;
}) {
  const [open, setOpen] = useState<Set<string>>(() => new Set(defaultOpen));
  const seedRef = useRef(seed);
  if (seedRef.current !== seed) {
    seedRef.current = seed;
    setOpen(new Set(defaultOpen));
  }
  const toggle = (id: string) =>
    setOpen((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  return (
    <>
      {GROUPS.map((g) => {
        const members = items.filter((it) =>
          g.kinds.includes((it.genKind ?? null) as LibraryGenKind | null),
        );
        if (members.length === 0) return null;
        const isOpen = open.has(g.id);
        return (
          <div key={g.id} className="vi-lib-group">
            <button
              type="button"
              className="vi-timeline-toggle vi-lib-group-head"
              aria-expanded={isOpen}
              onClick={() => toggle(g.id)}
            >
              <span className="vi-timeline-caret" aria-hidden>
                {isOpen ? "▾" : "▸"}
              </span>{" "}
              {g.label} <span className="vi-lib-count">({members.length})</span>
            </button>
            {isOpen ? (
              <div className="vi-gen-picker-grid">{members.map(renderItem)}</div>
            ) : null}
          </div>
        );
      })}
    </>
  );
}
