// The Settings-tab "Project" combobox source. Fetches the distinct known project
// names (GET /video/projects → { projects: [name, …] }) so the field can offer
// existing names as a datalist while still letting the user FREE-TYPE a brand-new
// one (choose-or-type). Data-not-throws, like useStudioPresets (from which the fetch
// idiom is cloned): a failed/absent list just yields [] — the field still works as a
// plain free-text input (empty = auto-named, the default). `refresh` re-pulls so a
// name coined on this enqueue can appear in the list next time.
import { useCallback, useEffect, useRef, useState } from "react";
import { z } from "zod";
import { request, okValue, errorOf, describeAppError } from "../transport/client";
import { hugpyConfig } from "../config";

const projectsResponseSchema = z.object({ projects: z.array(z.string()) });

export interface ProjectsState {
  projects: string[];
  loading: boolean;
  error: string | null;
  /** Force an immediate re-fetch of the project-name list. */
  refresh: () => void;
}

/**
 * Load the distinct known project names. `active` gates the network call; while
 * inactive the hook reports loading:false / an empty list. Stays cached after the
 * first successful pull; `refresh` re-pulls in the background.
 */
export function useProjects(active: boolean): ProjectsState {
  const [state, setState] = useState<Omit<ProjectsState, "refresh">>({
    projects: [],
    loading: false,
    error: null,
  });

  const mounted = useRef(true);
  const inFlight = useRef(false);
  const loadedRef = useRef(false);
  const ctrlRef = useRef<AbortController | null>(null);

  const load = useCallback(async (background: boolean) => {
    if (inFlight.current) return;
    inFlight.current = true;
    if (!background) setState((s) => ({ ...s, loading: true }));
    const signal = ctrlRef.current?.signal;
    try {
      const res = await request<unknown>(hugpyConfig.projectsUrl, {
        signal,
        meta: { specKey: "studio", operation: "video.projects.list" },
      });
      if (!mounted.current) return;
      if (!res.ok) {
        // A background refresh that fails must not wipe a working list.
        if (background) return;
        setState({ projects: [], loading: false, error: describeAppError(errorOf(res)) });
        return;
      }
      const parsed = projectsResponseSchema.safeParse(okValue(res));
      if (!parsed.success) {
        if (background) return;
        setState({ projects: [], loading: false, error: "Malformed projects response." });
        return;
      }
      loadedRef.current = true;
      setState({ projects: parsed.data.projects, loading: false, error: null });
    } finally {
      inFlight.current = false;
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    if (!active || loadedRef.current) return;
    const ctrl = new AbortController();
    ctrlRef.current = ctrl;
    void load(false);
    return () => {
      mounted.current = false;
      ctrl.abort();
    };
  }, [active, load]);

  const refresh = useCallback(() => {
    void load(loadedRef.current);
  }, [load]);

  return { ...state, refresh };
}
