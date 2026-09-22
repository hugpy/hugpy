// A thread turn for a tool run (ChatEntry.kind === "tool"). Renders the tool's
// result inline via the console's ExecutionOutput (which dispatches by operation
// to the Transcription/Keyword/Generic renderers), matching the assistant side
// (full width, no bubble). Reuses the registry + output tree as a library.
import ExecutionOutput from "../../../console/src/imports/page/UtilityPage/output/ExecutionOutput";
import { getPage } from "../../../console/src/imports/pages/pagesRegistry";
import type { PageSpec } from "../../../console/src/imports/pages/pageSpec";
import type { ChatEntry } from "../imports";

export default function ToolRunTurn({ chat }: { chat: ChatEntry }): JSX.Element {
  const title = chat.title ?? chat.specKey ?? "Tool";
  const running =
    chat.status === "thinking" ||
    chat.status === "streaming" ||
    chat.status === "queued";

  let spec: PageSpec | null = null;
  try {
    spec = chat.specKey ? getPage(chat.specKey) : null;
  } catch {
    spec = null;
  }
  // The bridge's media-intelligence turn isn't a registered tool (we don't want it
  // in the tray/router catalog). Synthesize a minimal spec so ExecutionOutput can
  // dispatch on the turn's explicit `operation`.
  if (!spec && chat.operation) {
    spec = {
      key: chat.specKey ?? "media/intelligence",
      title,
      category: "media",
      path: "",
      fields: [],
      accepts: [],
      produces: chat.operation as PageSpec["produces"],
    } as PageSpec;
  }

  return (
    <div className="hugpy-tool-turn" style={{ width: "100%", padding: "10px 0" }}>
      <div
        className="hugpy-tool-chip"
        style={{ fontSize: 12, opacity: 0.75, marginBottom: 6 }}
      >
        ▶ {title}
        {running ? " · running…" : chat.status === "error" ? " · error" : ""}
      </div>

      {chat.status === "complete" && spec && chat.result != null ? (
        <ExecutionOutput
          result={chat.result}
          spec={spec}
          operation={(chat.operation as never) ?? "any"}
        />
      ) : chat.status === "complete" ? (
        // Restored from history (heavy result stripped) — or an empty result.
        <div style={{ opacity: 0.55, fontSize: 12 }}>
          {chat.restored
            ? "Result not stored in history — re-run to view."
            : "No output."}
        </div>
      ) : chat.status === "error" ? (
        <div style={{ color: "var(--text-error, #c0392b)", fontSize: 13 }}>
          {chat.error ?? "Tool failed."}
        </div>
      ) : (
        <div style={{ opacity: 0.6, fontSize: 13 }}>…</div>
      )}
    </div>
  );
}
