import React, { useMemo, useState } from "react";
import styles from "../UtilityPage.module.css";
import { isRecord } from "./ResultHelpers";

type PdfTextPage = {
  page_num: number;
  text: string;
};

interface PdfTextOutputProps {
  value: unknown;
}

function asPdfTextPages(value: unknown): PdfTextPage[] {
  if (!Array.isArray(value)) return [];

  return value
    .filter(isRecord)
    .map((page) => ({
      page_num: typeof page.page_num === "number" ? page.page_num : 0,
      text: typeof page.text === "string" ? page.text : "",
    }))
    .filter((page) => page.text.trim());
}

function wordCount(text: string): number {
  return text.trim() ? text.trim().split(/\s+/).length : 0;
}

function pageLabel(pageNum: number): string {
  return `Page ${pageNum + 1}`;
}

export default function PdfTextOutput({ value }: PdfTextOutputProps) {
  const pages = asPdfTextPages(value);
  const [activeIndex, setActiveIndex] = useState(0);

  const activePage = pages[activeIndex];

  const combinedText = useMemo(() => {
    return pages
      .map((page) => `${pageLabel(page.page_num)}\n\n${page.text}`)
      .join("\n\n---\n\n");
  }, [pages]);

  const totalWords = useMemo(() => {
    return pages.reduce((total, page) => total + wordCount(page.text), 0);
  }, [pages]);

  if (!pages.length) return null;

  return (
    <div className={styles.pdfTextResult}>
      <div className={styles.pdfReportOverview}>
        <span className={styles.inlineMeta}>Pages: {pages.length}</span>
        <span className={styles.inlineMeta}>Words: {totalWords}</span>
      </div>

      <details className={styles.rawResultDetails}>
        <summary>Combined extracted text</summary>
        <pre className={styles.resultText}>{combinedText}</pre>
      </details>

      <div className={styles.pdfPageTabs}>
        {pages.map((page, index) => (
          <button
            key={page.page_num}
            type="button"
            className={`${styles.pdfPageTab} ${
              activeIndex === index ? styles.pdfPageTabActive : ""
            }`}
            onClick={() => setActiveIndex(index)}
          >
            {pageLabel(page.page_num)}
          </button>
        ))}
      </div>

      {activePage && (
        <div className={styles.pdfPagePanel}>
          <div className={styles.resultBlock}>
            <h3>{pageLabel(activePage.page_num)} Text</h3>
            <div className={styles.pdfTextBox}>{activePage.text}</div>
          </div>
        </div>
      )}
    </div>
  );
}
