// Shared media-intake receptacle — the drag/drop + click-to-browse surface the
// four stations mount in place of their empty item-view. NO upload logic lives
// here: the station passes its EXISTING upload handler (onPick / onPickVideo) as
// `onFile`, and this component only surfaces the single file the user drops or
// picks, then hands it straight back. Mirrors the media arm's receptacle
// interaction — a dashed `.vi-dropzone`, a hidden file input, a drag-active
// highlight, keyboard activation — but authors it as literal `.vi-*` CSS here.
// Single file by design: every station ingests exactly one source at a time.
import { useRef, useState } from "react";
import type { ChangeEvent, DragEvent, KeyboardEvent } from "react";

export interface DropReceptacleProps {
  /** The station's existing upload handler — receives the picked/dropped file. */
  onFile: (file: File) => void;
  /** Accept filter for the hidden <input> (e.g. "image/*", "audio/*,video/*"). */
  accept: string;
  /** Disable + show a "Working…" affordance while the station uploads/ingests. */
  busy?: boolean;
  /** Primary invite line. Falls back to the stable deploy marker below. */
  title?: string;
  /** Small station-specific secondary line. */
  hint?: string;
  /** Optional extra class on the dropzone (e.g. a compact variant). */
  className?: string;
}

// Stable deploy marker — a literal that survives minification (grepped
// post-deploy as the "new receptacle marker"). Doubles as the default title.
const DROP_MARKER = "Drop a file or click to browse";

export function DropReceptacle({
  onFile,
  accept,
  busy = false,
  title,
  hint,
  className,
}: DropReceptacleProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);

  function openPicker() {
    if (busy) return;
    inputRef.current?.click();
  }

  function onKeyDown(e: KeyboardEvent<HTMLDivElement>) {
    if (busy) return;
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      openPicker();
    }
  }

  function onChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) onFile(file);
    // Reset so re-picking the SAME file re-fires onChange.
    e.target.value = "";
  }

  function onDragOver(e: DragEvent<HTMLDivElement>) {
    if (busy) return;
    e.preventDefault();
    setDragActive(true);
  }
  function onDragEnter(e: DragEvent<HTMLDivElement>) {
    if (busy) return;
    e.preventDefault();
    setDragActive(true);
  }
  function onDragLeave() {
    setDragActive(false);
  }
  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragActive(false);
    if (busy) return;
    const file = e.dataTransfer.files?.[0];
    if (file) onFile(file);
  }

  const cls =
    "vi-dropzone" +
    (dragActive ? " vi-dropzone--active" : "") +
    (busy ? " vi-dropzone--busy" : "") +
    (className ? ` ${className}` : "");

  return (
    <div
      className={cls}
      role="button"
      tabIndex={0}
      aria-disabled={busy}
      onClick={openPicker}
      onKeyDown={onKeyDown}
      onDragOver={onDragOver}
      onDragEnter={onDragEnter}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="vi-file-input"
        onChange={onChange}
        disabled={busy}
      />
      {busy ? (
        <span className="vi-dropzone-title">Working…</span>
      ) : (
        <>
          <span className="vi-dropzone-title">{title ?? DROP_MARKER}</span>
          {hint ? <span className="vi-dropzone-hint">{hint}</span> : null}
        </>
      )}
    </div>
  );
}

export default DropReceptacle;
