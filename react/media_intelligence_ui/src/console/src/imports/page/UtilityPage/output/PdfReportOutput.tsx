import React, { useMemo, useState } from "react";
import styles from "../UtilityPage.module.css";
import KeywordOutput from "./KeywordOutput";
import { isRecord } from "./ResultHelpers";

type PdfPageReport = {
  scope?: string;
  summary?: string;
  text?: string;
  keywords?: unknown;
};

interface PdfReportOutputProps {
  value: unknown;
}

function asPages(value: unknown): PdfPageReport[] {
  if (!isRecord(value)) return [];

  const pages = value.pages;

  if (!Array.isArray(pages)) return [];

  return pages.filter(isRecord).map((page) => ({
    scope: typeof page.scope === "string" ? page.scope : "",
    summary: typeof page.summary === "string" ? page.summary : "",
    text: typeof page.text === "string" ? page.text : "",
    keywords: page.keywords,
  }));
}

function pageLabel(scope: string, index: number): string {
  const match = scope.match(/^page:(\d+)$/);

  if (!match) return `Page ${index + 1}`;

  return `Page ${Number(match[1]) + 1}`;
}

function wordCount(text: string): number {
  return text.trim() ? text.trim().split(/\s+/).length : 0;
}

export default function PdfReportOutput({ value }: PdfReportOutputProps) {
  const pages = asPages(value);
  const [activeIndex, setActiveIndex] = useState(0);

  const activePage = pages[activeIndex];

  const combinedSummary = useMemo(() => {
    return pages
      .map((page, index) => {
        if (!page.summary) return "";
        return `${pageLabel(page.scope ?? "", index)}: ${page.summary}`;
      })
      .filter(Boolean)
      .join("\n\n");
  }, [pages]);

  if (!pages.length) return null;

  return (
    <div className={styles.pdfReportResult}>
      <div className={styles.pdfReportOverview}>
        <span className={styles.inlineMeta}>Pages: {pages.length}</span>
        <span className={styles.inlineMeta}>
          Words: {pages.reduce((total, page) => total + wordCount(page.text ?? ""), 0)}
        </span>
      </div>

      {combinedSummary && (
        <div className={styles.resultBlock}>
          <h3>Document Summary</h3>
          <div className={styles.summaryBox}>{combinedSummary}</div>
        </div>
      )}

      <div className={styles.pdfPageTabs}>
        {pages.map((page, index) => (
          <button
            key={page.scope || index}
            type="button"
            className={`${styles.pdfPageTab} ${
              activeIndex === index ? styles.pdfPageTabActive : ""
            }`}
            onClick={() => setActiveIndex(index)}
          >
            {pageLabel(page.scope ?? "", index)}
          </button>
        ))}
      </div>

      {activePage && (
        <div className={styles.pdfPagePanel}>
          <div className={styles.resultBlock}>
            <h3>{pageLabel(activePage.scope ?? "", activeIndex)} Summary</h3>
            <div className={styles.summaryBox}>
              {activePage.summary || "No summary returned for this page."}
            </div>
          </div>

          {activePage.keywords && (
            <div className={styles.resultBlock}>
              <h3>Page Keywords</h3>
              <KeywordOutput value={activePage.keywords} />
            </div>
          )}

          {activePage.text && (
            <details className={styles.rawResultDetails}>
              <summary>Extracted page text</summary>
              <pre className={styles.resultText}>{activePage.text}</pre>
            </details>
          )}
        </div>
      )}
    </div>
  );
}
