import React from "react";
import styles from "../UtilityPage.module.css";

// Copy + Download for ANY output value. Strings copy/download as text; anything
// else as pretty JSON. Self-contained (no external clipboard util) so it can drop
// into every renderer — transcripts, extracted text, keywords, embeddings, the
// full intelligence record, chain results, etc.

function toText(v: unknown): string {
  if (typeof v === "string") return v;
  try {
    return JSON.stringify(v, null, 2);
  } catch {
    return String(v);
  }
}

export default function ResultActions({
  value,
  label,
}: {
  value: unknown;
  label?: string;
}) {
  const [copied, setCopied] = React.useState(false);
  const text = toText(value);
  const isJson = typeof value !== "string";

  if (value == null || text === "") return null;

  async function copy() {
    try {
      await navigator.clipboard?.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1200);
    } catch {
      /* clipboard blocked (insecure context / permissions) — ignore */
    }
  }

  function download() {
    try {
      const blob = new Blob([text], {
        type: isJson ? "application/json" : "text/plain",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${label || "output"}.${isJson ? "json" : "txt"}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 0);
    } catch {
      /* ignore */
    }
  }

  return (
    <div className={styles.resultActions}>
      <button
        type="button"
        className={styles.resultActionBtn}
        onClick={copy}
        title="Copy this output to the clipboard"
      >
        {copied ? "✓ Copied" : "Copy"}
      </button>
      <button
        type="button"
        className={styles.resultActionBtn}
        onClick={download}
        title="Download this output as a file"
      >
        Download
      </button>
    </div>
  );
}
