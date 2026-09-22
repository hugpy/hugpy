// Shows just above the composer when a file is attached (or an upload failed):
// the file chip, the intermediary's parallel suggestion, and the category-
// specific option chips (the tools that consume THIS kind of file). Clicking an
// option runs it on the file; the result is narrated like everything else. Plain
// text + a question still works — submit routes to the same file tool with the
// question as prompt. On upload failure we surface the reason (never silent).
import type { PageSpec } from "../../../console/src/imports/pages/pageSpec";
import type { UploadedFileRef, MediaCategory } from "../utilities/fileUpload";
import { readabilityFor } from "../utilities/fileUpload";

interface AttachmentBarProps {
  files: UploadedFileRef[];
  kind: MediaCategory;
  suggestion: string;
  suggesting: boolean;
  options: PageSpec[];
  error?: string;
  onRemove: () => void;
  onRunOption: (spec: PageSpec) => void;
}

const KIND_LABEL: Record<MediaCategory, string> = {
  audio: "Audio",
  image: "Image",
  video: "Video",
  pdf: "PDF",
  document: "Document",
  text: "File",
};

export default function AttachmentBar({
  files,
  kind,
  suggestion,
  suggesting,
  options,
  error,
  onRemove,
  onRunOption,
}: AttachmentBarProps): JSX.Element | null {
  if (files.length === 0 && !error) return null;
  const name = files.length === 0 ? "" : files.length === 1 ? files[0].name : `${files.length} files`;

  // Attach-time readability caveats (operator ask 2026-08-04, k64): tell the user
  // NOW that a legacy .doc won't read / that a video is audio-only, rather than
  // after a submit that fails. Deliberately quiet — dimmed text on the chip, not
  // the red error banner: the file is still attached and still usable.
  const notes = files
    .map((f) => ({ file: f, read: readabilityFor(f) }))
    .filter((n) => Boolean(n.read.note));

  return (
    <div
      className="hugpy-attachment-bar text-token-text-primary border border-token-border-light"
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 8,
        margin: "0 2px 6px",
        padding: "10px 12px",
        borderRadius: 14,
        // Match the composer's surface (var --bg-elevated-primary is dark in the
        // chat scope). The previous --bg-elevated-secondary resolves LIGHT here,
        // leaving the filename + suggestion text light-on-light / unreadable.
        background: "var(--bg-elevated-primary)",
        fontSize: 13,
      }}
    >
      {/* upload failure — surfaced, never silent */}
      {error && (
        <div style={{ display: "flex", alignItems: "center", gap: 8, color: "var(--text-error, #e5484d)" }}>
          <span style={{ flex: 1 }}>⚠ {error}</span>
          <button
            type="button"
            aria-label="Dismiss"
            title="Dismiss"
            onClick={onRemove}
            style={{ border: "none", background: "transparent", cursor: "pointer", color: "inherit", fontSize: 15, lineHeight: 1 }}
          >
            ✕
          </button>
        </div>
      )}

      {files.length > 0 && (
        <>
          {/* file chip + remove */}
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span
              style={{
                fontSize: 11,
                fontWeight: 600,
                letterSpacing: 0.3,
                textTransform: "uppercase",
                opacity: 0.6,
              }}
            >
              {KIND_LABEL[kind]}
            </span>
            <span style={{ fontWeight: 500, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {name}
            </span>
            <button
              type="button"
              aria-label="Remove attachment"
              title="Remove attachment"
              onClick={onRemove}
              style={{
                marginLeft: "auto",
                border: "none",
                background: "transparent",
                cursor: "pointer",
                color: "inherit",
                opacity: 0.6,
                fontSize: 15,
                lineHeight: 1,
              }}
            >
              ✕
            </button>
          </div>

          {/* readability caveats — quiet, per file, never blocking */}
          {notes.map(({ file, read }) => (
            <div
              key={file.id}
              style={{
                fontSize: 12,
                opacity: read.readable ? 0.55 : 0.75,
                display: "flex",
                gap: 6,
                alignItems: "baseline",
              }}
            >
              <span aria-hidden="true">{read.readable ? "ℹ" : "⚠"}</span>
              <span>
                {files.length > 1 ? `${file.name}: ` : ""}
                {read.note}
              </span>
            </div>
          ))}

          {/* intermediary's parallel suggestion */}
          <div style={{ opacity: suggesting ? 0.6 : 0.9, fontStyle: suggesting ? "italic" : "normal" }}>
            {suggesting ? "Looking at your file…" : suggestion || "Pick an action below, or type a question."}
          </div>

          {/* category-specific options (or a graceful note when none apply) */}
          {options.length > 0 ? (
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {options.map((spec) => (
                <button
                  key={spec.key}
                  type="button"
                  onClick={() => onRunOption(spec)}
                  className="hover:bg-token-surface-hover transition-colors duration-100"
                  style={{
                    fontSize: 12,
                    fontWeight: 500,
                    padding: "3px 10px",
                    borderRadius: 999,
                    // Outlined pill — matches the tool-tray chips (transparent
                    // fill, subtle border, theme text). Background is left unset so
                    // the hover class can tint it; text inherits the bar's color.
                    border: "1px solid var(--border, rgba(127,127,127,0.35))",
                    color: "inherit",
                    cursor: "pointer",
                  }}
                >
                  {spec.title}
                </button>
              ))}
            </div>
          ) : (
            <div style={{ fontSize: 12, opacity: 0.55 }}>
              No direct analysis for this file type yet — ask a question about it and I'll help.
            </div>
          )}
        </>
      )}
    </div>
  );
}
