import React, { useState } from "react";
import styles from "../UtilityPage.module.css";
import { isRecord, unwrapResult } from "./ResultHelpers";

// Renders the `extract` op: text read from an uploaded document (/ml/extract,
// per-page for PDFs) OR from a fetched webpage (/ml/fetch: title + url + text).
// One shared renderer because both produce the same {ok, text, pages?, ...} shape.

interface DocPage {
  index: number;
  text: string;
}

function asPages(u: Record<string, unknown>): DocPage[] {
  const arr = Array.isArray(u.pages) ? u.pages : [];
  return arr
    .filter(isRecord)
    .map((p, i) => ({
      index:
        typeof p.index === "number"
          ? p.index
          : typeof p.page_num === "number"
            ? p.page_num
            : i,
      text: typeof p.text === "string" ? p.text : "",
    }))
    .filter((p) => p.text.trim());
}

export default function DocumentTextOutput({ value }: { value: unknown }) {
  const u = unwrapResult(value);
  const [active, setActive] = useState(0);

  if (!isRecord(u)) {
    return <pre className={styles.resultText}>{String(u)}</pre>;
  }

  const text = typeof u.text === "string" ? u.text : "";
  const title = typeof u.title === "string" ? u.title : "";
  const url = typeof u.url === "string" ? u.url : "";
  const name = typeof u.name === "string" ? u.name : "";
  const kind = typeof u.kind === "string" ? u.kind : "";
  const pages = asPages(u);
  const words = text.trim() ? text.trim().split(/\s+/).length : 0;

  // Webpage-assessment extras (from /ml/fetch's assess_url): present only when
  // abstract_webtools produced a structured assessment — absent for /ml/extract.
  const description = typeof u.description === "string" ? u.description : "";
  const render = typeof u.render === "string" ? u.render : "";
  const truncated = u.truncated === true;
  const links = Array.isArray(u.links)
    ? (u.links.filter((l) => typeof l === "string") as string[])
    : [];
  // "selenium-forced" means the cheap fetch was JS-walled and a real browser render
  // was used — surface it so the user knows the read was the heavier path.
  const renderedViaBrowser = render === "selenium-forced";

  if (!text && !pages.length) {
    const err = typeof u.error === "string" ? u.error : "No text could be extracted.";
    return <pre className={`${styles.resultText} ${styles.resultError}`}>{err}</pre>;
  }

  const activePage = pages[active] ?? pages[0];

  return (
    <div className={styles.prettyResult}>
      <div
        className={styles.inlineMeta}
        style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "baseline" }}
      >
        {title && <strong>{title}</strong>}
        {name && <span>{name}</span>}
        {kind && <span>{kind}</span>}
        {url && (
          <a href={url} target="_blank" rel="noreferrer">
            {url}
          </a>
        )}
        {pages.length ? (
          <span>
            {pages.length} page{pages.length === 1 ? "" : "s"}
          </span>
        ) : null}
        {words ? <span>{words} words</span> : null}
        {renderedViaBrowser ? (
          <span title="The cheap fetch was empty/JS-walled, so a full browser render was used.">
            🖥 rendered
          </span>
        ) : null}
        {truncated ? <span title="Body text was capped to a token budget.">✂ truncated</span> : null}
      </div>

      {description ? (
        <p style={{ margin: "0 0 4px", opacity: 0.85, fontStyle: "italic" }}>{description}</p>
      ) : null}

      {pages.length > 0 ? (
        <>
          <div className={styles.pdfPageTabs}>
            {pages.map((p, i) => (
              <button
                key={i}
                type="button"
                className={`${styles.pdfPageTab} ${active === i ? styles.pdfPageTabActive : ""}`}
                onClick={() => setActive(i)}
              >
                Page {p.index + 1}
              </button>
            ))}
          </div>
          {activePage && (
            <div className={styles.pdfPagePanel}>
              <div className={styles.resultBlock}>
                <div className={styles.pdfTextBox} style={{ whiteSpace: "pre-wrap" }}>
                  {activePage.text}
                </div>
              </div>
            </div>
          )}
        </>
      ) : (
        <div className={styles.resultBlock}>
          <div className={styles.pdfTextBox} style={{ whiteSpace: "pre-wrap" }}>
            {text}
          </div>
        </div>
      )}

      {links.length > 0 ? (
        <details style={{ marginTop: 8, fontSize: 13 }}>
          <summary style={{ cursor: "pointer", opacity: 0.8 }}>
            {links.length} link{links.length === 1 ? "" : "s"} on this page
          </summary>
          <ul style={{ margin: "6px 0 0", paddingLeft: 18, display: "grid", gap: 2 }}>
            {links.map((l, i) => (
              <li key={i} style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                <a href={l} target="_blank" rel="noreferrer">
                  {l}
                </a>
              </li>
            ))}
          </ul>
        </details>
      ) : null}
    </div>
  );
}
