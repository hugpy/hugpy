import React, { type ReactNode } from "react";
import styles from "../UtilityPage.module.css";
import TranscriptionOutput from "./TranscriptionOutput";
import KeywordOutput from "./KeywordOutput";
import PdfReportOutput from "./PdfReportOutput";
import PdfTextOutput from "./PdfTextOutput";
import DocumentIntelligenceOutput from "./DocumentIntelligenceOutput";
import DocumentTextOutput from "./DocumentTextOutput";
import ImageGenOutput from "./ImageGenOutput";
import EmbeddingOutput from "./EmbeddingOutput";
import ResultActions from "./ResultActions";
import {
  formatUnknown,
  getNestedRecord,
  hasKeywordShape,
  isRecord,
  unwrapResult,
} from "./ResultHelpers";
import {
  TranscriptionResultSchema,
  KeywordResultSchema,
  PdfReportSchema,
  PdfTextSchema,
} from "../../../schemas";
import type { Operation } from "../../../pages/pageSpec";

// Phase 7 — output rendering dispatches through a registry keyed by the spec's DECLARED
// operation, not by sniffing the response shape. Each renderer validates its candidate
// shape(s) with a Phase-4 schema and shows a TYPED error on mismatch (never a silent
// JSON dump). Shape disambiguation that's genuinely op-local (e.g. `transcribe` returns
// a transcription shape for audio but a pdf-text shape for pdf/text) lives inside that
// op's renderer — scoped, not a global chain.

function RawDetails({ raw }: { raw: unknown }): ReactNode {
  // Default-open: the full target value is shown uncollapsed (the <summary> still
  // lets you fold it away). Renderers surface a friendly view above this.
  return (
    <details className={styles.rawResultDetails} open>
      <summary>Raw response</summary>
      <pre className={styles.resultText}>{formatUnknown(raw)}</pre>
    </details>
  );
}

function pretty(node: ReactNode, raw: unknown, label?: string): ReactNode {
  return (
    <div className={styles.prettyResult}>
      <ResultActions value={raw} label={label} />
      {node}
      <RawDetails raw={raw} />
    </div>
  );
}

function ShapeError({ op, raw }: { op: string; raw: unknown }): ReactNode {
  return (
    <div className={styles.prettyResult}>
      <pre className={`${styles.resultText} ${styles.resultError}`}>
        The server returned an unexpected shape for “{op}”. Showing the raw response.
      </pre>
      <RawDetails raw={raw} />
    </div>
  );
}

/** The long-tail renderer: string passthrough + summary/scope/keywords/text blocks. */
export function GenericOutput({ result }: { result: unknown }): ReactNode {
  const unwrapped = unwrapResult(result);

  if (typeof unwrapped === "string") {
    return (
      <div className={styles.prettyResult}>
        <ResultActions value={result} label="output" />
        <pre className={styles.resultText}>{unwrapped}</pre>
      </div>
    );
  }

  if (!isRecord(unwrapped)) {
    return (
      <div className={styles.prettyResult}>
        <ResultActions value={result} label="output" />
        <pre className={styles.resultText}>{formatUnknown(unwrapped)}</pre>
      </div>
    );
  }

  const summary = typeof unwrapped.summary === "string" ? unwrapped.summary : "";
  const text = typeof unwrapped.text === "string" ? unwrapped.text : "";
  const scope = typeof unwrapped.scope === "string" ? unwrapped.scope : "";
  const nestedKeywords = getNestedRecord(unwrapped, "keywords");

  return (
    <div className={styles.prettyResult}>
      <ResultActions value={result} label="output" />
      {summary && (
        <div className={styles.resultBlock}>
          <h3>Summary</h3>
          <div className={styles.summaryBox}>{summary}</div>
        </div>
      )}
      {scope && (
        <div className={styles.resultBlock}>
          <h3>Scope</h3>
          <div className={styles.inlineMeta}>{scope}</div>
        </div>
      )}
      {nestedKeywords && (
        <div className={styles.resultBlock}>
          <h3>Keywords</h3>
          <KeywordOutput value={nestedKeywords} />
        </div>
      )}
      {hasKeywordShape(unwrapped) && <KeywordOutput value={unwrapped} />}
      {text && (
        <details className={styles.rawResultDetails}>
          <summary>Input text</summary>
          <pre className={styles.resultText}>{text}</pre>
        </details>
      )}
      <RawDetails raw={result} />
    </div>
  );
}

function renderTranscribe(value: unknown): ReactNode {
  const u = unwrapResult(value);
  if (TranscriptionResultSchema.safeParse(u).success) {
    return pretty(<TranscriptionOutput value={u} />, value, "transcript");
  }
  if (PdfTextSchema.safeParse(u).success) {
    return pretty(<PdfTextOutput value={u} />, value, "transcript");
  }
  return <ShapeError op="transcribe" raw={value} />;
}

function renderKeywords(value: unknown): ReactNode {
  const u = unwrapResult(value);
  if (KeywordResultSchema.safeParse(u).success) {
    return pretty(<KeywordOutput value={u} />, value, "keywords");
  }
  return <ShapeError op="keywords" raw={value} />;
}

function renderSummarize(value: unknown): ReactNode {
  const u = unwrapResult(value);
  // pdf/summarize can return a page-by-page report; otherwise fall to the generic block.
  if (PdfReportSchema.safeParse(u).success) {
    return pretty(<PdfReportOutput value={u} />, value, "summary");
  }
  return <GenericOutput result={value} />;
}

function renderIntelligence(value: unknown): ReactNode {
  // The bridge's DocumentIntelligence record is always best-effort and tolerant,
  // so its renderer handles missing sections itself — no ShapeError gate.
  return pretty(<DocumentIntelligenceOutput value={value} />, value, "intelligence");
}

function renderExtract(value: unknown): ReactNode {
  // Document/URL text extraction (/ml/extract, /ml/fetch). Tolerant by design.
  return pretty(<DocumentTextOutput value={value} />, value, "extract");
}

function renderImagegen(value: unknown): ReactNode {
  // Text-to-image. No raw-response <details> — the base64 PNG would be megabytes
  // of noise; ImageGenOutput shows the picture + a Download link instead.
  return <ImageGenOutput value={value} />;
}

function renderMetadata(value: unknown): ReactNode {
  // `metadata` is produced by /ml/embed; render vectors when embedding-shaped,
  // else fall through to the generic block.
  const u = unwrapResult(value);
  if (isRecord(u) && Array.isArray(u.embeddings)) {
    return pretty(<EmbeddingOutput value={value} />, value, "embeddings");
  }
  return <GenericOutput result={value} />;
}

/** Registry: declared op → renderer. Ops absent here use GenericOutput. */
export const outputRenderers = new Map<
  Operation,
  (value: unknown) => ReactNode
>([
  ["transcribe", renderTranscribe],
  ["keywords", renderKeywords],
  ["summarize", renderSummarize],
  ["intelligence", renderIntelligence],
  ["extract", renderExtract],
  ["imagegen", renderImagegen],
  ["metadata", renderMetadata],
]);
