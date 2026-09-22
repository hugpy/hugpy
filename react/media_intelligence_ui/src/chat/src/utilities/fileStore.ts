// Client-side file library persistence — PER SESSION (sessionStorage).
// The Sidebar "Files" section — and the transcripts/extracted text saved onto
// each file — live in sessionStorage, NOT localStorage: the library is scoped to
// the tab session. It survives an in-session reload but is wiped automatically
// when the tab/session closes, so uploaded files never linger across sessions.
// `blobUrl` is an in-tab object URL (invalid after reload) so it's stripped;
// metadata + extractedText (the transcript) persist for the session only.
// Degrades silently to in-memory when storage is unavailable or over quota.
import type { StoredFile } from "./fileUpload";
import { isCanned } from "../../../demo/mode";

const KEY = "hugpy.media.files.v1";
const MAX_FILES = 100;
const MAX_TEXT = 200_000; // per-file extracted-text cap to stay under quota

// One-time purge of any library persisted by the previous localStorage-backed
// build, so files saved in an earlier session can't survive into this one.
try {
  localStorage.removeItem(KEY);
} catch {
  // storage unavailable — nothing to purge.
}

export function loadFiles(): StoredFile[] {
  try {
    const raw = sessionStorage.getItem(KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return (parsed as StoredFile[]).filter(
      (f) => f && typeof f.id === "string" && typeof f.name === "string",
    );
  } catch {
    return [];
  }
}

export function saveFiles(files: StoredFile[]): void {
  // The canned demo is in-memory only — never overwrite a real visitor's library.
  if (isCanned()) return;
  try {
    const trimmed = files.slice(0, MAX_FILES).map((f) => ({
      ...f,
      blobUrl: undefined, // in-tab object URL — invalid after reload
      extractedText:
        f.extractedText && f.extractedText.length > MAX_TEXT
          ? f.extractedText.slice(0, MAX_TEXT)
          : f.extractedText,
    }));
    sessionStorage.setItem(KEY, JSON.stringify(trimmed));
  } catch {
    // quota exceeded / storage unavailable — degrade to in-memory silently.
  }
}
