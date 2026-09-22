import React from "react";
import styles from "../UtilityPage.module.css";
import { isRecord, unwrapResult } from "./ResultHelpers";

// Renders /ml/embed (feature-extraction) and the similarity matrix that
// /ml/similarity adds. Returns null when the value isn't embedding-shaped, so
// the registry wrapper can fall back to the generic renderer.

export default function EmbeddingOutput({ value }: { value: unknown }) {
  const u = unwrapResult(value);
  const embeddings =
    isRecord(u) && Array.isArray(u.embeddings) ? (u.embeddings as unknown[]) : null;
  if (!embeddings) return null;

  const first = Array.isArray(embeddings[0]) ? (embeddings[0] as unknown[]) : [];
  const dims = first.length;
  const preview = first
    .slice(0, 8)
    .map((n) => (typeof n === "number" ? n.toFixed(4) : String(n)));

  const similarities =
    isRecord(u) && Array.isArray(u.similarities) ? (u.similarities as unknown[][]) : null;

  return (
    <div className={styles.prettyResult}>
      <div className={styles.resultBlock}>
        <h3>Embeddings</h3>
        <div
          className={styles.inlineMeta}
          style={{ display: "flex", gap: 12, flexWrap: "wrap" }}
        >
          <span>
            {embeddings.length} vector{embeddings.length === 1 ? "" : "s"}
          </span>
          {dims ? <span>{dims} dimensions</span> : null}
        </div>
        {preview.length > 0 && (
          <pre className={styles.resultText}>
            [{preview.join(", ")}
            {dims > preview.length ? ", …" : ""}]
          </pre>
        )}
      </div>

      {similarities && (
        <div className={styles.resultBlock}>
          <h3>Similarity matrix</h3>
          <table className={styles.resultTable}>
            <tbody>
              {similarities.map((row, r) => (
                <tr key={r}>
                  {row.map((cell, c) => (
                    <td key={c}>
                      {typeof cell === "number" ? cell.toFixed(3) : String(cell)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
