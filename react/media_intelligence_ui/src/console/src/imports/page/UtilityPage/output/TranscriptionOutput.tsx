import React from "react";
import styles from "../UtilityPage.module.css";
import { isRecord } from "./ResultHelpers";

type Segment = {
  id?: number;
  start?: number;
  end?: number;
  text?: string;
  avg_logprob?: number;
  no_speech_prob?: number;
  compression_ratio?: number;
};

interface TranscriptionOutputProps {
  value: unknown;
}

function formatTime(seconds: unknown): string {
  if (typeof seconds !== "number" || !Number.isFinite(seconds)) return "--:--";

  const total = Math.max(0, Math.floor(seconds));
  const mins = Math.floor(total / 60);
  const secs = total % 60;

  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

function asSegments(value: unknown): Segment[] {
  if (!Array.isArray(value)) return [];

  return value
    .filter(isRecord)
    .map((item) => ({
      id: typeof item.id === "number" ? item.id : undefined,
      start: typeof item.start === "number" ? item.start : undefined,
      end: typeof item.end === "number" ? item.end : undefined,
      text: typeof item.text === "string" ? item.text.trim() : "",
      avg_logprob:
        typeof item.avg_logprob === "number" ? item.avg_logprob : undefined,
      no_speech_prob:
        typeof item.no_speech_prob === "number" ? item.no_speech_prob : undefined,
      compression_ratio:
        typeof item.compression_ratio === "number"
          ? item.compression_ratio
          : undefined,
    }))
    .filter((segment) => segment.text);
}

export default function TranscriptionOutput({
  value,
}: TranscriptionOutputProps) {
  if (!isRecord(value)) return null;

  const language =
    typeof value.language === "string" ? value.language : "unknown";

  const text = typeof value.text === "string" ? value.text.trim() : "";

  const segments = asSegments(value.segments);

  const duration =
    segments.length > 0
      ? segments.reduce(
          (max, segment) =>
            typeof segment.end === "number" ? Math.max(max, segment.end) : max,
          0,
        )
      : null;

  return (
    <div className={styles.transcriptResult}>
      <div className={styles.transcriptMetaRow}>
        <span className={styles.inlineMeta}>Language: {language}</span>

        {duration != null && (
          <span className={styles.inlineMeta}>
            Duration: {formatTime(duration)}
          </span>
        )}

        <span className={styles.inlineMeta}>
          Segments: {segments.length}
        </span>
      </div>

      {text && (
        <div className={styles.resultBlock}>
          <h3>Transcript</h3>
          <div className={styles.transcriptBox}>{text}</div>
        </div>
      )}

      {segments.length > 0 && (
        <details className={styles.segmentDetails}>
          <summary>Timed segments</summary>

          <div className={styles.segmentList}>
            {segments.map((segment, index) => (
              <div
                key={segment.id ?? `${segment.start}-${index}`}
                className={styles.segmentItem}
              >
                <div className={styles.segmentTime}>
                  {formatTime(segment.start)} → {formatTime(segment.end)}
                </div>

                <div className={styles.segmentText}>
                  {segment.text}
                </div>
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}
