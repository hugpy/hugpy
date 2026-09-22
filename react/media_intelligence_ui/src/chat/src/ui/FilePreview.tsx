/*
 * FilePreview.tsx — inline preview modal for a stored upload.
 *
 *   - image / audio / video  → rendered inline from the in-session blob URL.
 *   - pdf / document         → the extracted full-text (scrollable), or a
 *                              status line while extraction runs.
 *   - everything             → a download link (the blob URL).
 *
 * Session-only: previews rely on the in-tab blob URL + extracted text held in
 * the file store, so there is no server fetch. Closes on Esc or backdrop click.
 */
import { useEffect } from "react";
import type { StoredFile } from "../utilities/fileUpload";
import { fileKindGlyph } from "../utilities/fileUpload";

interface FilePreviewProps {
  file: StoredFile | null;
  onClose: () => void;
}

export default function FilePreview({ file, onClose }: FilePreviewProps): JSX.Element | null {
  useEffect(() => {
    if (!file) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [file, onClose]);

  if (!file) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={`Preview ${file.name}`}
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 50,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 24,
        background: "rgba(0,0,0,0.6)",
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          display: "flex",
          flexDirection: "column",
          width: "min(880px, 100%)",
          maxHeight: "86vh",
          borderRadius: 16,
          border: "1px solid var(--border-light, rgba(255,255,255,0.12))",
          background: "var(--main-surface-primary, #050505)",
          color: "var(--text-primary, #f4f4f5)",
          overflow: "hidden",
        }}
      >
        {/* header */}
        <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "12px 14px", borderBottom: "1px solid var(--border-light, rgba(255,255,255,0.1))" }}>
          <span aria-hidden style={{ fontSize: 16 }}>{fileKindGlyph(file.kind)}</span>
          <span style={{ fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1 }}>
            {file.name}
          </span>
          {file.blobUrl && (
            <a
              href={file.blobUrl}
              download={file.name}
              style={{ fontSize: 13, color: "var(--link, #a7c7ff)", textDecoration: "none", padding: "2px 8px" }}
            >
              ⬇ Download
            </a>
          )}
          <button
            type="button"
            aria-label="Close preview"
            title="Close"
            onClick={onClose}
            style={{ border: "none", background: "transparent", color: "inherit", cursor: "pointer", fontSize: 18, lineHeight: 1, padding: "2px 6px" }}
          >
            ✕
          </button>
        </div>

        {/* body */}
        <div style={{ flex: 1, minHeight: 0, overflow: "auto", padding: 16 }}>
          {file.kind === "image" && file.blobUrl && (
            <img src={file.blobUrl} alt={file.name} style={{ display: "block", maxWidth: "100%", margin: "0 auto", borderRadius: 8 }} />
          )}
          {file.kind === "audio" && file.blobUrl && (
            <audio controls src={file.blobUrl} style={{ width: "100%" }} />
          )}
          {file.kind === "video" && file.blobUrl && (
            <video controls src={file.blobUrl} style={{ width: "100%", maxHeight: "70vh", borderRadius: 8 }} />
          )}
          {(file.kind === "pdf" || file.kind === "document") && (
            file.extractStatus === "extracting" ? (
              <div style={{ opacity: 0.7, fontStyle: "italic" }}>Reading document…</div>
            ) : file.extractedText ? (
              <pre style={{ whiteSpace: "pre-wrap", wordBreak: "break-word", fontSize: 13, lineHeight: 1.5, margin: 0, fontFamily: "inherit" }}>
                {file.extractedText}
              </pre>
            ) : (
              <div style={{ opacity: 0.7 }}>
                {file.extractStatus === "error"
                  ? "Couldn't read this document."
                  : "No extractable text — use Download to open it."}
              </div>
            )
          )}
          {!["image", "audio", "video", "pdf", "document"].includes(file.kind) && (
            <div style={{ opacity: 0.7 }}>No inline preview for this file type — use Download.</div>
          )}
        </div>
      </div>
    </div>
  );
}
