// OPERATOR HISTORY HYDRATION (operator ask 2026-08-13): the superadmin gets
// COMPLETE visibility of previous renders in the Session Library — not just
// what this browser tab produced. The library itself stays a per-session
// sessionStorage working set (by design); this hook back-fills it from the
// SERVER record once per session for an operator principal:
//
//   * probe: GET keysVideoShareUrl — the same 200⇒operator / 401⇒guest probe
//     SharePanel uses (members and share guests never hydrate; their lists
//     stay owner-scoped server-side).
//   * source: GET /video/jobs?all=1 for terminal done generate_image /
//     generate_scene / generate_movie rows (the list projection strips
//     results, so each job detail is fetched for its outputs). Clips
//     (studio_i2v) already hydrate via useStudioClips poll — not repeated.
//   * sink: addToLibrary — idempotent by uri, tombstone-respecting (an item
//     the operator removed stays removed), genKind-tagged so the Generate
//     station can filter the strips by its CURRENT SECTION.
//
// Once-per-session guard in sessionStorage; canned demo never hydrates.
import { useEffect } from "react";
import { request, okValue } from "../transport/client";
import { hugpyConfig, jobStatusUrl } from "../config";
import { addToLibrary, type LibraryGenKind } from "./mediaLibrary";
import { getSessionId } from "../session";
import { isCanned } from "../demo/mode";
import type { MediaRef } from "./contract";

// generate_studio_movie = CINEMA renders — hydrated too (mapped onto the
// generate_movie library kind: same Videos shelf, no schema change).
const GEN_KINDS: readonly string[] = [
  "generate_image", "generate_scene", "generate_movie", "generate_studio_movie",
];
// v2 (2026-08-13): the once-per-session flag must change whenever the
// HYDRATED KINDS change — a tab that hydrated v1 (no cinema jobs) kept its
// flag through every reload and never saw generate_studio_movie history.
const FLAG_PREFIX = "vi.operatorHistory.v2";
const DETAIL_CAP = 150; // newest-first; enough history without hammering the API

function flagKey(): string {
  try { return `${FLAG_PREFIX}:${getSessionId()}`; } catch { return FLAG_PREFIX; }
}

export function useOperatorHistory(): void {
  useEffect(() => {
    if (isCanned()) return;
    try { if (sessionStorage.getItem(flagKey())) return; } catch { /* no-op */ }
    let cancelled = false;
    void (async () => {
      // Operator probe — non-operators (401/403) never hydrate.
      const probe = await request<unknown>(hugpyConfig.keysVideoShareUrl, {
        meta: { specKey: "media", operation: "operator.probe" },
      });
      if (cancelled || !probe.ok) return;
      // jobStatusUrl("") = `${apiBase}/video/jobs/` — trim the slash and query
      // the list route with terminal rows included.
      const list = await request<unknown>(
        `${jobStatusUrl("").replace(/\/$/, "")}?all=1&limit=300`,
        { meta: { specKey: "media", operation: "history.jobs.list" } },
      );
      if (cancelled || !list.ok) return;
      const jobs = ((okValue(list) as { jobs?: unknown[] })?.jobs ?? []) as Array<
        Record<string, unknown>
      >;
      const wanted = jobs
        .filter((j) => j?.status === "done" && GEN_KINDS.includes(String(j?.name)))
        .slice(0, DETAIL_CAP);
      for (const j of wanted) {
        if (cancelled) return;
        const id = String(j.job_id ?? j.id ?? "");
        if (!id) continue;
        const det = await request<unknown>(jobStatusUrl(id), {
          meta: { specKey: "media", operation: "history.job.detail" },
        });
        if (!det.ok) continue;
        const result = (okValue(det) as { result?: { outputs?: unknown[] } })?.result;
        for (const o of (result?.outputs ?? []) as Array<Record<string, unknown>>) {
          if (!o?.uri || !o?.asset_id) continue;
          const ref = {
            asset_id: String(o.asset_id),
            kind: (o.kind === "video" ? "video" : "image"),
            uri: String(o.uri),
            mime: (o.mime as string) ?? (o.kind === "video" ? "video/mp4" : "image/png"),
            width: (o.width as number) ?? null,
            height: (o.height as number) ?? null,
            duration_s: (o.duration_s as number) ?? null,
            fps_native: (o.fps_native as number) ?? null,
            channels: (o.channels as number) ?? null,
            sample_rate: (o.sample_rate as number) ?? null,
          } as MediaRef;
          addToLibrary(ref, "history", `${String(j.name)} · ${id.slice(0, 8)}`, {
            genKind: (j.name === "generate_studio_movie"
              ? "generate_movie"
              : j.name) as LibraryGenKind,
            // groupId = the producing job: each historical run groups as its own
            // work (like live pushes) instead of one giant shared "history" pile
            // whose identical key collided ACROSS the type sections (expanding
            // Videos/history also expanded Images/history).
            groupId: id,
          });
        }
      }
      try { sessionStorage.setItem(flagKey(), String(Date.now())); } catch { /* no-op */ }
    })();
    return () => { cancelled = true; };
  }, []);
}
