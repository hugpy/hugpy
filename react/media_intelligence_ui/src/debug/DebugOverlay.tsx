import React, { useMemo, useState } from "react";
import styles from "./DebugOverlay.module.css";

type DebugValueMap = Record<string, unknown>;

interface DebugOverlayProps {
  title?: string;
  values: DebugValueMap;
  defaultOpen?: boolean;
  enabled?: boolean;
}

function safeStringify(value: unknown): string {
  try {
    return JSON.stringify(
      value,
      (_key, val) => {
        if (val instanceof File) {
          return {
            __type: "File",
            name: val.name,
            size: val.size,
            type: val.type,
            lastModified: val.lastModified,
          };
        }

        if (val instanceof FileList) {
          return Array.from(val).map((file) => ({
            __type: "File",
            name: file.name,
            size: file.size,
            type: file.type,
            lastModified: file.lastModified,
          }));
        }

        if (val instanceof Error) {
          return {
            __type: "Error",
            name: val.name,
            message: val.message,
            stack: val.stack,
          };
        }

        return val;
      },
      2,
    );
  } catch (error) {
    return String(value);
  }
}

export default function DebugOverlay({
  title = "Debug",
  values,
  defaultOpen = false,
  enabled = import.meta.env.DEV,
}: DebugOverlayProps) {
  const [open, setOpen] = useState(defaultOpen);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);

  const keys = Object.keys(values);

  const activeKey = selectedKey && selectedKey in values ? selectedKey : keys[0];

  const renderedValue = useMemo(() => {
    if (!activeKey) return "";
    return safeStringify(values[activeKey]);
  }, [activeKey, values]);

  if (!enabled) return null;

  return (
    <div className={styles.debugRoot}>
      <button
        type="button"
        className={styles.toggleButton}
        onClick={() => setOpen((next) => !next)}
      >
        {open ? "Hide" : "Debug"}
      </button>

      {open && (
        <aside className={styles.panel}>
          <div className={styles.header}>
            <strong>{title}</strong>
            <button type="button" onClick={() => setOpen(false)}>
              ×
            </button>
          </div>

          <div className={styles.body}>
            <nav className={styles.keyList}>
              {keys.map((key) => (
                <button
                  key={key}
                  type="button"
                  className={key === activeKey ? styles.activeKey : ""}
                  onClick={() => setSelectedKey(key)}
                >
                  {key}
                </button>
              ))}
            </nav>

            <pre className={styles.valueBox}>{renderedValue}</pre>
          </div>
        </aside>
      )}
    </div>
  );
}
