import { describe, it, expect, vi, afterEach } from "vitest";
import { request, describeAppError, errorOf, okValue } from "./client";

function res(
  status: number,
  body: unknown,
  headers: Record<string, string> = {},
) {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get: (k: string) => headers[k.toLowerCase()] ?? null },
    text: async () => (typeof body === "string" ? body : JSON.stringify(body)),
  } as unknown as Response;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("transport client — Result<AppError> mapping", () => {
  it("maps a 2xx JSON body to ok", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => res(200, { hello: "world" })));
    const r = await request("/x", { method: "POST" });
    expect(r.ok).toBe(true);
    expect(okValue(r)).toEqual({ hello: "world" });
  });

  it("falls back to text when body isn't JSON", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => res(200, "plain text")));
    const r = await request("/x", { method: "POST", expect: "text" });
    expect(okValue(r)).toBe("plain text");
  });

  it("maps 401/403 to unauthorized", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => res(401, { error: "nope" })));
    const r = await request("/x", { method: "POST" });
    expect(r.ok).toBe(false);
    expect(errorOf(r).kind).toBe("unauthorized");
  });

  it("maps 5xx to server with the body's error message + traceId header", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => res(500, { error: "boom" }, { "x-trace-id": "t-123" })),
    );
    const r = await request("/x", { method: "POST" });
    expect(r.ok).toBe(false);
    const e = errorOf(r);
    expect(e.kind).toBe("server");
    if (e.kind === "server") {
      expect(e.message).toBe("boom");
      expect(e.traceId).toBe("t-123");
    }
  });

  it("maps a thrown fetch to network", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new Error("connection refused");
      }),
    );
    const r = await request("/x", { method: "POST" });
    expect(errorOf(r).kind).toBe("network");
  });

  it("does NOT retry a POST, but DOES retry an idempotent GET on 5xx", async () => {
    const post = vi.fn(async () => res(500, { error: "x" }));
    vi.stubGlobal("fetch", post);
    await request("/x", { method: "POST" });
    expect(post).toHaveBeenCalledTimes(1);

    let n = 0;
    const get = vi.fn(async () => (++n < 2 ? res(500, {}) : res(200, { ok: 1 })));
    vi.stubGlobal("fetch", get);
    const r = await request("/x", { method: "GET", retries: 1 });
    expect(r.ok).toBe(true);
    expect(get).toHaveBeenCalledTimes(2);
  });

  it("returns aborted when the caller signal is already aborted", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new DOMException("aborted", "AbortError");
      }),
    );
    const ctrl = new AbortController();
    ctrl.abort();
    const r = await request("/x", { method: "POST", signal: ctrl.signal });
    expect(errorOf(r).kind).toBe("aborted");
  });

  it("does NOT attach X-Request-Id by default (keeps cross-origin requests simple)", async () => {
    const f = vi.fn(async () => res(200, {}));
    vi.stubGlobal("fetch", f);
    await request("/x", { method: "POST" });
    const init = (f.mock.calls[0] as unknown[])[1] as RequestInit;
    expect((init.headers as Record<string, string>)["X-Request-Id"]).toBeUndefined();
  });
});

describe("describeAppError", () => {
  it("is kind-aware", () => {
    expect(describeAppError({ kind: "unauthorized", status: 401, requestId: "r" })).toMatch(
      /sign in/i,
    );
    expect(describeAppError({ kind: "timeout", ms: 5000, requestId: "r" })).toMatch(/timed out/i);
    expect(describeAppError({ kind: "aborted", requestId: "r" })).toBe("Cancelled.");
    expect(
      describeAppError({ kind: "server", status: 500, message: "boom", traceId: "t1", requestId: "r" }),
    ).toMatch(/boom.*t1/);
  });
});
