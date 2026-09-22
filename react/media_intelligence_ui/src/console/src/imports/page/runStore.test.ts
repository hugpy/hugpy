import { describe, it, expect } from "vitest";
import {
  beginRun,
  setRunSteps,
  completeRun,
  failRun,
  cancelRun,
  getLatestForSpec,
} from "./runStore";

describe("runStore (Phase 8 — run lifecycle as state)", () => {
  it("begins a run and exposes it as the latest for its spec", () => {
    const id = beginRun("k/a", new AbortController());
    const run = getLatestForSpec("k/a");
    expect(run?.id).toBe(id);
    expect(run?.status).toBe("running");
  });

  it("completes a run with a result (survives as latest)", () => {
    const id = beginRun("k/b", new AbortController());
    completeRun(id, { answer: 42 });
    const run = getLatestForSpec("k/b");
    expect(run?.status).toBe("ok");
    expect(run?.result).toEqual({ answer: 42 });
    expect(run?.endedAt).toBeTruthy();
  });

  it("fails a run with an error message", () => {
    const id = beginRun("k/c", new AbortController());
    failRun(id, "boom");
    expect(getLatestForSpec("k/c")?.status).toBe("error");
    expect(getLatestForSpec("k/c")?.error).toBe("boom");
  });

  it("cancel aborts the controller and sets cancelled", () => {
    const ctrl = new AbortController();
    const id = beginRun("k/d", ctrl);
    cancelRun(id);
    expect(ctrl.signal.aborted).toBe(true);
    expect(getLatestForSpec("k/d")?.status).toBe("cancelled");
  });

  it("a late completeRun does NOT override a cancelled run", () => {
    const id = beginRun("k/e", new AbortController());
    cancelRun(id);
    completeRun(id, { sneaky: true }); // arrives after cancel
    expect(getLatestForSpec("k/e")?.status).toBe("cancelled");
    expect(getLatestForSpec("k/e")?.result).toBeUndefined();
  });

  it("records live chain steps", () => {
    const id = beginRun("k/f", new AbortController());
    setRunSteps(id, [{ pageKey: "p1", ok: true, data: 1 }]);
    expect(getLatestForSpec("k/f")?.steps?.length).toBe(1);
    setRunSteps(id, [
      { pageKey: "p1", ok: true, data: 1 },
      { pageKey: "p2", ok: false, data: null, error: "x" },
    ]);
    expect(getLatestForSpec("k/f")?.steps?.length).toBe(2);
  });

  // k65 — a batch writes one last snapshot after Cancel so the file that was
  // mid-flight stops reading "running"; a FINISHED run is still never rewritten.
  it("steps may still be recorded on a cancelled run, but not on a finished one", () => {
    const cancelled = beginRun("k/h", new AbortController());
    setRunSteps(cancelled, [{ pageKey: "a.pdf", ok: true, data: 1 }]);
    cancelRun(cancelled);
    setRunSteps(cancelled, [
      { pageKey: "a.pdf", ok: true, data: 1 },
      { pageKey: "b.pdf", ok: false, data: null },
    ]);
    expect(getLatestForSpec("k/h")?.status).toBe("cancelled");
    expect(getLatestForSpec("k/h")?.steps?.length).toBe(2);

    const done = beginRun("k/i", new AbortController());
    completeRun(done, "final", [{ pageKey: "a.pdf", ok: true, data: 1 }]);
    setRunSteps(done, []); // late write after the run ended
    expect(getLatestForSpec("k/i")?.steps?.length).toBe(1);
  });

  it("starting a new run replaces the latest for that spec", () => {
    const first = beginRun("k/g", new AbortController());
    completeRun(first, "one");
    const second = beginRun("k/g", new AbortController());
    expect(getLatestForSpec("k/g")?.id).toBe(second);
    expect(getLatestForSpec("k/g")?.status).toBe("running");
  });
});
