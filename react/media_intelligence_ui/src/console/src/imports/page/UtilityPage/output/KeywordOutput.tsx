import React from "react";
import styles from "../UtilityPage.module.css";
import {
  asDensityEntries,
  asStringArray,
  isRecord,
} from "./ResultHelpers";

interface KeywordOutputProps {
  value: unknown;
}

export default function KeywordOutput({ value }: KeywordOutputProps) {
  if (!isRecord(value)) return null;

  const primary = asStringArray(value.primary);
  const secondary = asStringArray(value.secondary);
  const dropped = asStringArray(value.dropped);
  const hashtags = asStringArray(value.hashtags);
  const slugCandidates = asStringArray(value.slug_candidates);

  const metaKeywords =
    typeof value.meta_keywords === "string" ? value.meta_keywords : "";

  const density = asDensityEntries(value.density);

  const densityFlags = isRecord(value.density_flags)
    ? value.density_flags
    : {};

  const hasAnything =
    primary.length ||
    secondary.length ||
    dropped.length ||
    hashtags.length ||
    slugCandidates.length ||
    metaKeywords ||
    density.length;

  if (!hasAnything) return null;

  return (
    <>
      {primary.length > 0 && (
        <div className={styles.resultBlock}>
          <h3>Primary Keywords</h3>
          <div className={styles.chipRow}>
            {primary.map((item) => (
              <span key={item} className={styles.chip}>
                {item}
              </span>
            ))}
          </div>
        </div>
      )}

      {secondary.length > 0 && (
        <div className={styles.resultBlock}>
          <h3>Secondary Keywords</h3>
          <div className={styles.chipRow}>
            {secondary.map((item) => (
              <span key={item} className={styles.chip}>
                {item}
              </span>
            ))}
          </div>
        </div>
      )}

      {metaKeywords && (
        <div className={styles.resultBlock}>
          <h3>Meta Keywords</h3>
          <pre className={styles.resultText}>{metaKeywords}</pre>
        </div>
      )}

      {hashtags.length > 0 && (
        <div className={styles.resultBlock}>
          <h3>Hashtags</h3>
          <div className={styles.chipRow}>
            {hashtags.map((item) => (
              <span key={item} className={styles.chip}>
                {item}
              </span>
            ))}
          </div>
        </div>
      )}

      {slugCandidates.length > 0 && (
        <div className={styles.resultBlock}>
          <h3>Slug Candidates</h3>
          <div className={styles.chipRow}>
            {slugCandidates.map((item) => (
              <span key={item} className={styles.chip}>
                {item}
              </span>
            ))}
          </div>
        </div>
      )}

      {density.length > 0 && (
        <div className={styles.resultBlock}>
          <h3>Keyword Density</h3>
          <table className={styles.resultTable}>
            <thead>
              <tr>
                <th>Keyword</th>
                <th>Density</th>
                <th>Flag</th>
              </tr>
            </thead>
            <tbody>
              {density.map(([key, densityValue]) => (
                <tr key={key}>
                  <td>{key}</td>
                  <td>{String(densityValue)}%</td>
                  <td>{String(densityFlags[key] ?? "")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {dropped.length > 0 && (
        <div className={styles.resultBlock}>
          <h3>Dropped</h3>
          <div className={styles.chipRow}>
            {dropped.map((item) => (
              <span key={item} className={styles.chipMuted}>
                {item}
              </span>
            ))}
          </div>
        </div>
      )}
    </>
  );
}
