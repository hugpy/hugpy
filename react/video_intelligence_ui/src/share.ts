// k9 — video SHARE-LINK credential (client half).
//
// An operator mints a share link (`https://dev.hugpy.ai/video/?share=<key>`) and
// sends it to an outside party. On first load this module lifts the `?share=` key
// out of the URL, stashes it in localStorage (so it survives reloads and SPA
// navigations — on dev the shell + assets are served ungated by the webpack
// watcher, so a bare reload restores the key from storage), and STRIPS it from the
// address bar (nothing reloads — `history.replaceState`) so the credential isn't
// left sitting in the URL.
//
// The stored key then rides EVERY hugpy API call:
//   • XHR through the transport (`transport/client.ts`) carries it as the
//     `X-Video-Share` header — the SPA's single fetch choke point stamps it once;
//   • element-src media loads (`<img>`/`<video>` to /video/media | /video/studio/
//     clip, via `config.ts`) can't carry a header, so those URLs carry `?share=`
//     in the query (`withShareParam`).
// Server side, `video_auth._video_share_principal` accepts any of the three.
//
// A console-authenticated operator never has a stored share key (they never opened
// a share link), so nothing here affects them — their same-origin session cookie
// authenticates their calls exactly as before.

const STORAGE_KEY = "hugpy.video.share";
const PARAM = "share";

let _key: string | null = null;
let _initialized = false;

/** Lift `?share=` out of the URL (stash + strip), else restore from storage.
 *  Idempotent; safe to call before any network activity. Must run at boot. */
export function initShare(): void {
  if (_initialized) return;
  _initialized = true;
  if (typeof window === "undefined") return;
  try {
    const url = new URL(window.location.href);
    const fromUrl = url.searchParams.get(PARAM);
    if (fromUrl) {
      _key = fromUrl;
      try {
        localStorage.setItem(STORAGE_KEY, fromUrl);
      } catch {
        /* private mode — the in-memory key still drives this load */
      }
      // Strip the credential from the address bar without reloading.
      url.searchParams.delete(PARAM);
      const rest = url.search ? url.search : "";
      window.history.replaceState({}, "", `${url.pathname}${rest}${url.hash}`);
      return;
    }
    _key = localStorage.getItem(STORAGE_KEY);
  } catch {
    // storage/URL unavailable — leave whatever we have (likely null).
  }
}

/** The active share key, or null. Lazily initialises so an early caller is safe. */
export function videoShareKey(): string | null {
  if (!_initialized) initShare();
  return _key;
}

/** Append the share credential to a URL used as an element `src` (img/video),
 *  which cannot carry the `X-Video-Share` header. No-op when there's no key. */
export function withShareParam(url: string): string {
  const key = videoShareKey();
  if (!key) return url;
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}${PARAM}=${encodeURIComponent(key)}`;
}

/** True when this session is running off a share link (no console login). Used
 *  only to shape UI copy — the Share button visibility is decided by a live
 *  operator-auth probe, never by this. */
export function isSharedSession(): boolean {
  return videoShareKey() != null;
}
