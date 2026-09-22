// Compact file-intake bar — the single presentation for picking a source in
// every station. NO drag-drop dropbox: it is just a "Choose …" button (a label
// wrapping a hidden file input) plus the caller-supplied status / filename /
// meta line, and an optional Clear affordance. The button still drives the
// station's existing onPick/upload path — this is presentation only.
import type { ChangeEvent, ReactNode } from "react";

export function AddFileBar({
  accept,
  onFile,
  buttonLabel,
  busy = false,
  onClear,
  children,
}: {
  accept: string;
  onFile: (file: File) => void;
  buttonLabel: string;
  busy?: boolean;
  /** Optional clear/reset affordance (shown only when provided). */
  onClear?: () => void;
  /** Status / filename / meta rendered inline after the button. */
  children?: ReactNode;
}) {
  function onChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) onFile(file);
    // Reset so re-picking the SAME file re-fires onChange.
    e.target.value = "";
  }

  return (
    <div className="vi-addbar">
      <label className="vi-btn vi-btn-accent vi-file-label">
        {buttonLabel}
        <input
          type="file"
          accept={accept}
          className="vi-file-input"
          onChange={onChange}
          disabled={busy}
        />
      </label>
      {children}
      {onClear && (
        <button
          type="button"
          className="vi-btn vi-btn-sm vi-btn-ghost"
          onClick={onClear}
          disabled={busy}
        >
          Clear
        </button>
      )}
    </div>
  );
}

export default AddFileBar;
