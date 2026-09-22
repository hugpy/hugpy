import React, { type ReactNode } from "react";
import styles from "../UtilityPage.module.css";
import KeywordOutput from "./KeywordOutput";
import TranscriptionOutput from "./TranscriptionOutput";
import { isRecord, unwrapResult } from "./ResultHelpers";

// Renders a media_intelligence DocumentIntelligence record (the output of the
// chat bridge / the package's MediaPipeline): a pipeline-stage strip, summary,
// keywords, transcript (a-v), caption (image), and the extracted document text.
// Tolerant by design — every section is optional and only shown when present.

interface DocStage {
  stage: string;
  status: "ok" | "skipped" | "error";
  detail?: string;
}

function StageStrip({ stages }: { stages: DocStage[] }): ReactNode {
  if (!stages.length) return null;
  const dot = (s: DocStage["status"]) =>
    s === "ok" ? "●" : s === "skipped" ? "○" : "✕";
  const color = (s: DocStage["status"]) =>
    s === "ok" ? "var(--text-success, #2e7d32)"
    : s === "error" ? "var(--text-error, #c0392b)"
    : "var(--text-tertiary, #9aa0a6)";
  return (
    <div className={styles.inlineMeta} style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 8 }}>
      {stages.map((st, i) => (
        <span key={i} title={st.detail ?? st.status} style={{ color: color(st.status) }}>
          {dot(st.status)} {st.stage}
        </span>
      ))}
    </div>
  );
}

export default function DocumentIntelligenceOutput({ value }: { value: unknown }): ReactNode {
  const di = unwrapResult(value);
  if (!isRecord(di)) {
    return <pre className={styles.resultText}>{String(di)}</pre>;
  }

  const source = typeof di.source === "string" ? di.source : "";
  const kind = typeof di.kind === "string" ? di.kind : "";
  const summary = typeof di.summary === "string" ? di.summary : "";
  const caption = typeof di.caption === "string" ? di.caption : "";
  const text = typeof di.text === "string" ? di.text : "";
  const pages = Array.isArray(di.pages) ? di.pages : [];
  const transcript = isRecord(di.transcript) ? di.transcript : null;
  const keywords = di.keywords;
  const stages = (Array.isArray(di.stages) ? di.stages : []) as DocStage[];

  // k65 — a PARTIAL read reports itself. The extractor returns per-page warnings
  // and page counts when a document only half read; showing them here is what
  // stops a 3-of-5-page extract from looking like the whole document. Read in
  // both spellings so the client bridge (camelCase) and a direct server payload
  // (snake_case) both surface.
  const num = (...vals: unknown[]) => vals.find((v) => typeof v === "number") as number | undefined;
  const pagesTotal = num(di.pagesTotal, di.pages_total);
  const pagesExtracted = num(di.pagesExtracted, di.pages_extracted);
  const warnings = (Array.isArray(di.warnings) ? di.warnings : []).filter(
    (w): w is string => typeof w === "string",
  );
  const partial =
    typeof pagesTotal === "number" &&
    typeof pagesExtracted === "number" &&
    pagesTotal > 0 &&
    pagesExtracted < pagesTotal;

  return (
    <div className={styles.prettyResult}>
      {(source || kind) && (
        <div className={styles.inlineMeta}>
          {source}{kind ? ` · ${kind}` : ""}
        </div>
      )}

      <StageStrip stages={stages} />

      {(partial || warnings.length > 0) && (
        <div className={styles.resultBlock}>
          <div className={styles.inlineMeta} style={{ color: "var(--text-error, #c0392b)" }}>
            {partial && `Extracted ${pagesExtracted}/${pagesTotal} pages`}
            {partial && warnings.length > 0 && " · "}
            {warnings.join(" · ")}
          </div>
        </div>
      )}

      {summary && (
        <div className={styles.resultBlock}>
          <h3>Summary</h3>
          <div className={styles.summaryBox}>{summary}</div>
        </div>
      )}

      {caption && !summary && (
        <div className={styles.resultBlock}>
          <h3>Description</h3>
          <div className={styles.summaryBox}>{caption}</div>
        </div>
      )}

      {keywords != null && (
        <div className={styles.resultBlock}>
          <h3>Keywords</h3>
          <KeywordOutput value={keywords} />
        </div>
      )}

      {transcript && (
        <div className={styles.resultBlock}>
          <h3>Transcript</h3>
          <TranscriptionOutput value={transcript} />
        </div>
      )}

      {!transcript && pages.length > 0 && (
        <details className={styles.rawResultDetails}>
          <summary>Document text · {pages.length} page{pages.length === 1 ? "" : "s"}</summary>
          {pages.map((p, i) => {
            const pageText = isRecord(p) && typeof p.text === "string" ? p.text : "";
            const idx = isRecord(p) && typeof p.index === "number" ? p.index : i;
            return (
              <div key={i} className={styles.resultBlock}>
                <div className={styles.inlineMeta}>Page {idx}</div>
                <pre className={styles.resultText}>{pageText}</pre>
              </div>
            );
          })}
        </details>
      )}

      {!transcript && pages.length === 0 && text && (
        <details className={styles.rawResultDetails}>
          <summary>Extracted text</summary>
          <pre className={styles.resultText}>{text}</pre>
        </details>
      )}
    </div>
  );
}
