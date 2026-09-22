import React from "react";
import styles from "../UtilityPage.module.css";
import { isRecord, unwrapResult } from "./ResultHelpers";

// Renders the result of /ml/imagine (text-to-image): one or more generated
// images carried inline as base64 PNG (GeneratedImage.b64). When b64 is absent
// (a caller asked for paths only) we note the server-side path instead of a
// broken <img>.

interface GenImage {
  b64?: string;
  path?: string;
  width?: number;
  height?: number;
  seed?: number;
}

function asImages(value: unknown): GenImage[] {
  const u = unwrapResult(value);
  const arr = isRecord(u) && Array.isArray(u.images)
    ? u.images
    : Array.isArray(u)
      ? u
      : [];
  return arr.filter(isRecord).map((im) => ({
    b64: typeof im.b64 === "string" ? im.b64 : undefined,
    path: typeof im.path === "string" ? im.path : undefined,
    width: typeof im.width === "number" ? im.width : undefined,
    height: typeof im.height === "number" ? im.height : undefined,
    seed: typeof im.seed === "number" ? im.seed : undefined,
  }));
}

export default function ImageGenOutput({ value }: { value: unknown }) {
  const images = asImages(value);
  const u = unwrapResult(value);
  const err = isRecord(u) && typeof u.error === "string" ? u.error : "";

  if (!images.length) {
    return (
      <pre className={`${styles.resultText} ${err ? styles.resultError : ""}`}>
        {err || "No image was returned."}
      </pre>
    );
  }

  return (
    <div className={styles.prettyResult}>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 12 }}>
        {images.map((im, i) => {
          const src = im.b64 ? `data:image/png;base64,${im.b64}` : undefined;
          if (!src) {
            return (
              <div key={i} className={styles.inlineMeta}>
                Image generated server-side{im.path ? `: ${im.path}` : ""} (no inline preview)
              </div>
            );
          }
          return (
            <figure key={i} style={{ margin: 0, maxWidth: 360 }}>
              <img
                src={src}
                alt={`Generated image ${i + 1}`}
                style={{
                  width: "100%",
                  height: "auto",
                  display: "block",
                  borderRadius: 12,
                  border: "1px solid rgba(127,127,127,0.25)",
                }}
              />
              <figcaption
                className={styles.inlineMeta}
                style={{ display: "flex", gap: 10, marginTop: 6, flexWrap: "wrap" }}
              >
                {im.width && im.height ? <span>{im.width}×{im.height}</span> : null}
                {im.seed != null ? <span>seed {im.seed}</span> : null}
                <a href={src} download={`generated-${i + 1}.png`}>Download</a>
              </figcaption>
            </figure>
          );
        })}
      </div>
    </div>
  );
}
