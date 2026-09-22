// The player-forward VIEWER — the .vi-studio-viewer stage (mirrors the scene/video tab
// preview) that plays the selected clip. It lives at the TOP of the studio plane
// (StudioPlane), driven by the plane's lifted `selected` + `pinned`.
//
// Header-bar controls (operator ask 2026-07-12 — "Can the change be made THROUGH to
// change the initial scene or video or make it 'new'."). Before this pass the viewer had
// NO controls of its own: it silently rode the plane's auto-follow effect and, on top of
// that, silently substituted "any playable clip" whenever `selected` didn't resolve to
// one — including a DELIBERATELY cleared selection. Combined with an auto-follow effect
// that (see StudioPlane.tsx) only re-picked when the selected job vanished from the list
// entirely, the viewer would latch onto the first playable clip of the session and then
// never visibly change again — exactly the "stuck, unchangeable default" the operator
// described. Fixed on both ends:
//   • `chosen` below is now a STRICT read of `selected` — no silent substitution. A
//     cleared viewer (`selected === null`) shows the explicit empty state and only the
//     explicit empty state, until the operator picks something or resumes follow.
//   • the plane's auto-follow effect now genuinely re-syncs to the newest playable clip
//     on every list change (see that file's comment) instead of sticking forever.
// `pinned` is the one semantic switch this header reads and toggles: true once the
// operator has manually picked or cleared, false while the plane auto-follows.
//
// Layout below is INLINE style (not new app.css classes) — this slice is scoped to
// src/stations/studio/ only, and the file-local ad hoc `style={{...}}` layout is
// already the established idiom for one-off arrangement in this same directory (see
// StudioGenerateTab.tsx's picker rows / hero sections). Reused, not-new pieces (the
// button classes, the .vi-studio-src-row poster/id row, .vi-comfy-empty) still use
// their existing classes.
import { useCallback, useEffect, useState } from "react";
import {
  studioClipUrl,
  studioClipToEditorUrl,
  mediaBytesUrl,
  hugpyConfig,
} from "../../config";
import { request, okValue, errorOf, describeAppError } from "../../transport/client";
import { shortId, dims, whenText, type Clip } from "./studioShared";
import type { LibraryItem } from "../../video/mediaLibrary";
import { SourceThumb } from "./StudioGenerateTab";

export interface StudioViewerProps {
  clips: Clip[];
  /** Session-library media from the OTHER tabs (cinema movies, scene clips/frames,
   *  crops, uploads — anything with pixels), already deduped against `clips` by uri
   *  and sorted newest-first by the plane. Selected via a `lib:<asset_id>` key and
   *  played straight from its media handle (`mediaBytesUrl`). */
  library: LibraryItem[];
  /** The plane's lifted selection — a job id, a `lib:<asset_id>` library key, or null
   *  (nothing chosen / explicitly cleared). Read STRICTLY below; there is no "nearest
   *  playable" fallback any more. */
  selected: string | null;
  /** True once the operator has taken manual control (a library pick, or "Clear
   *  viewer") — the plane's auto-follow effect stands down while this is true. Drives
   *  the "↻ follow newest" toggle's shown state. */
  pinned: boolean;
  /** "▤ Play from library" panel pick — hands the chosen key (job id or
   *  `lib:<asset_id>`) back to the plane, which selects it AND pins (see
   *  StudioPlane.tsx). */
  onPick: (key: string) => void;
  /** "✕ Clear viewer" — empties the selection; the plane pins alongside it so the empty
   *  state sticks instead of being repopulated by the next poll. */
  onClear: () => void;
  /** "↻ follow newest" toggle — flips `pinned` on the plane. */
  onToggleFollow: () => void;
}

export function StudioViewer({
  clips,
  library,
  selected,
  pinned,
  onPick,
  onClear,
  onToggleFollow,
}: StudioViewerProps) {
  // Local to the picker panel only — never touches the plane's lifted selection.
  const [pickerOpen, setPickerOpen] = useState(false);
  const [previewId, setPreviewId] = useState<string | null>(null);

  // STRICT: only what `selected` names, and only while it is actually playable. No "any
  // playable clip" fallback — that silent substitution is exactly what made a cleared or
  // stuck-on-a-failed-job viewer unable to ever show empty (see the file header note).
  const libKey = (it: LibraryItem) => `lib:${it.ref.asset_id}`;
  const chosenLib = selected?.startsWith("lib:")
    ? (library.find((it) => libKey(it) === selected) ?? null)
    : null;
  const chosen =
    !chosenLib && selected
      ? (clips.find((c) => c.job_id === selected && c.playable) ?? null)
      : null;
  const libLabel = (it: LibraryItem) => it.label ?? it.origin;
  const libWhen = (it: LibraryItem) => {
    try {
      return new Date(it.addedAt).toLocaleTimeString();
    } catch {
      return "";
    }
  };

  const playable = clips.filter((c) => c.playable);

  // ── "Send to Editor" (k12) — CONSOLE-OPERATOR ONLY affordance ───────────────
  // Visibility is AUTH-DECIDED, not guessed (mirrors SharePanel): probe the same
  // operator-gated GET (keysVideoShareUrl) once on mount. null = probing (hide),
  // false = a share guest / anon (hide — the route would 403 anyway), true =
  // operator (show, but only alongside a real playable clip). A second small GET
  // here is consistent with SharePanel's own probe, not a regression.
  const [operator, setOperator] = useState<boolean | null>(null);
  const [sending, setSending] = useState(false);
  const [sentMsg, setSentMsg] = useState<string | null>(null);
  const [sendErr, setSendErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    void (async () => {
      const res = await request<unknown>(hugpyConfig.keysVideoShareUrl, {
        method: "GET",
        meta: { specKey: "studio", operation: "studio.operatorProbe" },
      });
      if (alive) setOperator(res.ok);
    })();
    return () => {
      alive = false;
    };
  }, []);

  // Clear any transient send message when the shown clip changes, so a "sent"
  // note never lingers over a different clip.
  useEffect(() => {
    setSentMsg(null);
    setSendErr(null);
  }, [selected]);

  const sendToEditor = useCallback(async () => {
    if (!chosen) return;
    setSending(true);
    setSentMsg(null);
    setSendErr(null);
    const res = await request<{ filename?: string; mode?: string }>(
      studioClipToEditorUrl(chosen.job_id),
      { method: "POST", meta: { specKey: "studio", operation: "studio.clip.toEditor" } },
    );
    setSending(false);
    if (!res.ok) {
      setSendErr(`Could not send to the editor — ${describeAppError(errorOf(res))}`);
      return;
    }
    const fn = okValue(res).filename ?? "clip.mp4";
    const msg = `✓ Sent to editor — ${fn}`;
    setSentMsg(msg);
    window.setTimeout(() => setSentMsg((cur) => (cur === msg ? null : cur)), 4000);
  }, [chosen]);

  function pick(jobId: string) {
    onPick(jobId);
    setPickerOpen(false);
    setPreviewId(null);
  }

  return (
    <section aria-label="Viewer" className="vi-studio-viewer-wrap">
      {/* ── ALWAYS-VISIBLE header bar: the viewer must never read as a fixed, unmovable
          default. It always names what's showing (or says nothing is), and always offers
          a way to pick something else, clear it, or hand control back to auto-follow. ── */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "0.6rem",
          marginBottom: "0.5rem",
        }}
      >
        <span
          style={{
            fontFamily: "var(--vi-mono)",
            fontSize: "0.85rem",
            color: "var(--vi-text)",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
          title={
            chosenLib
              ? `${libLabel(chosenLib)} · ${chosenLib.ref.asset_id}`
              : chosen
                ? chosen.job_id
                : "Nothing playing"
          }
        >
          {chosenLib
            ? `▶ ${libLabel(chosenLib)} · ${shortId(chosenLib.ref.asset_id)}`
            : chosen
              ? `▶ ${shortId(chosen.job_id)}`
              : "— nothing playing —"}
        </span>
        <div
          style={{ display: "flex", flexWrap: "wrap", gap: "0.35rem" }}
          role="group"
          aria-label="Viewer controls"
        >
          {/* CONSOLE-OPERATOR ONLY (operator === true) AND only with a real playable
              clip showing: hand this clip to the LAN Filmora inbox. Hidden for a
              share guest / anon (the route would 403 regardless). */}
          {operator === true && chosen && (
            <button
              type="button"
              className="vi-btn vi-btn-sm vi-btn-accent"
              disabled={sending}
              onClick={() => void sendToEditor()}
              title="Copy this clip, in a Filmora-ready MP4, into your editor inbox on the LAN"
            >
              {sending ? "Sending…" : "⇱ Send to Editor"}
            </button>
          )}
          <button
            type="button"
            className="vi-btn vi-btn-sm vi-btn-ghost"
            aria-expanded={pickerOpen}
            onClick={() => setPickerOpen((v) => !v)}
            title="Pick any playable clip to show in the viewer"
          >
            ▤ Play from library
          </button>
          <button
            type="button"
            className="vi-btn vi-btn-sm vi-btn-ghost"
            onClick={onClear}
            disabled={!chosen && !chosenLib && pinned}
            title="Empty the viewer — shows an explicit empty state instead of a clip"
          >
            ✕ Clear viewer
          </button>
          <button
            type="button"
            className={pinned ? "vi-btn vi-btn-sm vi-btn-ghost" : "vi-btn vi-btn-sm vi-btn-accent"}
            aria-pressed={!pinned}
            onClick={onToggleFollow}
            title={
              pinned
                ? "Off — the viewer stays on your pick. Click to resume following the newest playable clip."
                : "On — the viewer follows the newest playable clip as renders complete. Click to freeze it here."
            }
          >
            ↻ follow newest · {pinned ? "off" : "on"}
          </button>
        </div>
      </div>

      {/* Send-to-Editor transient feedback (operator only). Success auto-clears
          after a few seconds; a failure stays until the next send or clip change. */}
      {sentMsg && (
        <p
          className="vi-comfy-hint"
          role="status"
          style={{ margin: "0 0 0.5rem", color: "var(--vi-accent)" }}
        >
          {sentMsg}
        </p>
      )}
      {sendErr && (
        <p className="vi-error" role="alert" style={{ margin: "0 0 0.5rem" }}>
          {sendErr}
        </p>
      )}

      {/* "▤ Play from library" panel — the reachable select-to-viewer affordance. (The
          original StudioLibrarySection had this exact click-to-select logic, but that
          component is unmounted in BOTH plane mounts — see StudioLibraryTab.tsx and the
          StudioPlane.tsx header note. This panel is the live replacement: mounted once,
          inside the ONE shared StudioPlane, so it is reachable identically from the
          standalone Studio station AND the Generate station's studio mode.) Rows reuse
          the studio "source clip" poster/id idiom (SourceThumb + .vi-studio-src-row,
          exported from StudioGenerateTab.tsx) rather than inventing a fifth row shape. */}
      {pickerOpen && (
        <div
          style={{
            marginBottom: "0.6rem",
            padding: "0.6rem",
            border: "1px solid var(--vi-border)",
            borderRadius: "8px",
            background: "var(--vi-surface-raised)",
          }}
          role="listbox"
          aria-label="Play from library"
        >
          {playable.length === 0 && library.length === 0 ? (
            <p className="vi-comfy-hint">Nothing playable yet — generate something below.</p>
          ) : (
            <ul
              style={{
                listStyle: "none",
                margin: 0,
                padding: 0,
                maxHeight: "16rem",
                overflowY: "auto",
              }}
            >
              {/* RECENT GENERATIONS from every other tab (session library) — listed
                  FIRST, newest-first, so a fresh cinema movie / scene clip is one
                  click away (operator ask 2026-08-27: this picker used to offer only
                  the studio_i2v clip catalog). */}
              {library.map((it) => {
                const key = libKey(it);
                const isCurrent = key === selected;
                const isPreviewing = previewId === key;
                return (
                  <li key={key} className="vi-studio-src-row">
                    {it.ref.kind === "image" ? (
                      <div className="vi-studio-src-thumb" aria-hidden>
                        <img
                          src={mediaBytesUrl(it.ref.uri)}
                          alt=""
                          style={{ width: "100%", height: "100%", objectFit: "cover" }}
                        />
                      </div>
                    ) : (
                      <SourceThumb
                        uri={it.ref.uri}
                        name={libLabel(it)}
                        playing={isPreviewing}
                        onToggle={() => setPreviewId((cur) => (cur === key ? null : key))}
                      />
                    )}
                    <div className="vi-studio-src-body">
                      <div className="vi-studio-src-meta" title={it.ref.asset_id}>
                        <code>{libLabel(it)}</code>
                        <span>{it.ref.kind}</span>
                        {libWhen(it) ? <span>{libWhen(it)}</span> : null}
                        {isCurrent ? <span>· current</span> : null}
                      </div>
                      <div className="vi-studio-src-actions">
                        <button
                          type="button"
                          className="vi-btn vi-btn-sm vi-btn-accent"
                          onClick={() => pick(key)}
                          disabled={isCurrent}
                          title="Show this in the viewer"
                        >
                          {isCurrent ? "▸ Playing" : "▸ Play here"}
                        </button>
                      </div>
                    </div>
                  </li>
                );
              })}
              {playable.map((c) => {
                const uri = c.output?.uri;
                const isPreviewing = previewId === c.job_id;
                const isCurrent = c.job_id === selected;
                return (
                  <li key={c.job_id} className="vi-studio-src-row">
                    {uri ? (
                      <SourceThumb
                        uri={uri}
                        name={shortId(c.job_id)}
                        playing={isPreviewing}
                        onToggle={() =>
                          setPreviewId((cur) => (cur === c.job_id ? null : c.job_id))
                        }
                      />
                    ) : (
                      <div className="vi-studio-src-thumb vi-lib-placeholder" aria-hidden>
                        <span>clip</span>
                      </div>
                    )}
                    <div className="vi-studio-src-body">
                      <div className="vi-studio-src-meta" title={c.job_id}>
                        <code>{shortId(c.job_id)}</code>
                        {dims(c) ? <span>{dims(c)}</span> : null}
                        {whenText(c) ? <span>{whenText(c)}</span> : null}
                        {isCurrent ? <span>· current</span> : null}
                      </div>
                      <div className="vi-studio-src-actions">
                        <button
                          type="button"
                          className="vi-btn vi-btn-sm vi-btn-accent"
                          onClick={() => pick(c.job_id)}
                          disabled={isCurrent}
                          title="Play this clip in the viewer"
                        >
                          {isCurrent ? "▸ Playing" : "▸ Play here"}
                        </button>
                      </div>
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      )}

      {chosenLib ? (
        <div className="vi-studio-viewer">
          {chosenLib.ref.kind === "image" ? (
            <img
              key={selected ?? undefined}
              className="vi-studio-clip-video"
              src={mediaBytesUrl(chosenLib.ref.uri)}
              alt={libLabel(chosenLib)}
            />
          ) : (
            <video
              key={selected ?? undefined}
              className="vi-studio-clip-video"
              controls
              autoPlay
              loop
              playsInline
              src={mediaBytesUrl(chosenLib.ref.uri)}
            />
          )}
        </div>
      ) : chosen ? (
        <div className="vi-studio-viewer">
          <video
            key={chosen.job_id}
            className="vi-studio-clip-video"
            controls
            autoPlay
            loop
            playsInline
            src={studioClipUrl(chosen.job_id)}
          />
        </div>
      ) : (
        <div className="vi-comfy-empty">
          <p className="vi-comfy-empty-title">
            {pinned ? "Nothing playing — cleared" : "No clip yet"}
          </p>
          <p className="vi-comfy-empty-sub">
            {pinned
              ? "Pick from the library above, or turn “follow newest” back on."
              : "Generate a clip below, or pick a playable clip from the library above — it plays here."}
          </p>
        </div>
      )}
    </section>
  );
}
