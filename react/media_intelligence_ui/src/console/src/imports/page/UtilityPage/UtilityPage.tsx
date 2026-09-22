// src/utilities/imports/utilities/page/UtilityPage.tsx
import React, { useMemo, useState, useSyncExternalStore } from "react";
import styles from "./UtilityPage.module.css";
import {
  API_BASE_HUGPY,
  type FieldSpec,
  type FieldSource,
  type FieldResolution,
  type MediaInputValue,
  type MediaKind,
  type Operation,
  type PageSpec,
  type UploadedFileRef,
} from "./../../pages/pageSpec";
import { getPage } from "./../../pages/pagesRegistry";
import { runChain, type ChainStepResult } from "./../../chain/chainRuntime";
import { asFileArray, OperationSchema } from "./../../schemas";
import { describeAppError, errorOf } from "./../../../../../transport/client";
import {
  subscribe,
  getLatestForSpec,
  beginRun,
  setRunSteps,
  completeRun,
  failRun,
  cancelRun,
  type BatchFileResult,
} from "./../runStore";
import {
  submitPage,
  resolveSourcedField,
  isFieldVisible,
  selectedUploadedFiles,
  batchFilesFor,
  pinInputToFile,
  type PageValues,
  buildSourcePayload,
} from "./../submitPage";
import ExecutionOutput from "./output/ExecutionOutput";
interface Props {
  spec: PageSpec;
  input: MediaInputValue;
  media: MediaKind;
  operation: Operation | "any";
  onJumpToInput: () => void;
  // The tool's option values are owned one level up (HugpyConsole), alongside the
  // selected spec + operation, and rendered in the operation dropdown row. This
  // page only READS them to run the tool — it no longer holds form state itself.
  values: PageValues;
}
type Values = PageValues;

// ---------- helpers (module scope) ----------

export function initialValues(spec: PageSpec): Values {
  const v: Values = {};
  for (const f of spec.fields) {
    if (f.source) continue;                     // sourced fields never live in local state
    if (f.kind === "file" || f.kind === "files") v[f.name] = [];
    else if (f.kind === "checkbox") v[f.name] = !!f.default;
    else v[f.name] = f.default ?? "";
  }
  return v;
}


// MIME -> MediaKind so isCompatible can compare apples to apples.
function mimeToMediaKind(mime?: string): MediaKind | null {
  if (!mime) return null;
  if (mime.startsWith("image/")) return "image";
  if (mime.startsWith("audio/")) return "audio";
  if (mime.startsWith("video/")) return "video";
  if (mime.startsWith("text/")) return "text";
  return null;
}
function hasLocalFiles(input: MediaInputValue): boolean {
  return (
    input.inputMode === "file" &&
    Array.isArray(input.files) &&
    input.files.length > 0
  );
}

function hasUploadedSelectedFiles(input: MediaInputValue): boolean {
  return (
    input.inputMode === "file" &&
    selectedUploadedFiles(input).length > 0
  );
}

function hasReceptacleFiles(input: MediaInputValue): boolean {
  return hasLocalFiles(input) || hasUploadedSelectedFiles(input);
}

function fileFieldCanUseReceptacle(
  field: FieldSpec,
  input: MediaInputValue,
): boolean {
  return (
    input.inputMode === "file" &&
    !field.source &&
    (field.kind === "file" || field.kind === "files") &&
    hasReceptacleFiles(input)
  );
}

function asReceptacleSourcedField(field: FieldSpec): FieldSpec {
  return {
    ...field,
    source: "selectedIds",
  };
}

function localFileResolution(input: MediaInputValue): FieldResolution {
  const files = input.files ?? [];
  const names = files.map((file) => file.name);

  return {
    status: "ok",
    value: names,
    summary:
      names.length === 1
        ? names[0]
        : `${names.length} files — ${names.join(", ")}`,
  };
}

// ---------- subcomponents (module scope) ----------

function InheritedField({
  field,
  resolution,
  onJumpToInput,
}: {
  field: FieldSpec;
  resolution: FieldResolution;
  onJumpToInput: () => void;
}) {
  const sourceLabel: Record<FieldSource, string> = {
    text: "input text",
    url: "input URL",
    selectedIds: "uploaded files",
  };

  return (
    <div className={`${styles.inherited} ${styles[`inherited_${resolution.status}`]}`}>
      <div className={styles.inheritedHeader}>
        <label>{field.label}{field.required && " *"}</label>
        <span className={styles.inheritedBadge}>from {sourceLabel[field.source!]}</span>
      </div>

      {resolution.status === "ok" && (
        <div className={styles.inheritedSummary}>{resolution.summary}</div>
      )}

      {resolution.status !== "ok" && (
        <div className={styles.inheritedEmpty}>
          <span>{resolution.reason}</span>
          <button type="button" className={styles.linkBtn} onClick={onJumpToInput}>
            Go to input panel
          </button>
        </div>
      )}
    </div>
  );
}

function FieldInput({
  field, value, onChange,
}: {
  field: FieldSpec;
  value: Values[string];
  onChange: (name: string, v: Values[string]) => void;
}) {
  const id = `f_${field.name}`;
  const common = { id, name: field.name };

  return (
    <div className={styles.field}>
      <label htmlFor={id}>{field.label}{field.required && " *"}</label>
      {field.kind === "textarea" && (
        <textarea {...common} rows={6} value={String(value ?? "")}
          onChange={e => onChange(field.name, e.target.value)} />
      )}
      {field.kind === "text" && (
        <input {...common} type="text" value={String(value ?? "")}
          onChange={e => onChange(field.name, e.target.value)} />
      )}
      {field.kind === "number" && (
        <input {...common} type="number" value={String(value ?? "")}
          onChange={e => onChange(field.name, Number(e.target.value))} />
      )}
      {field.kind === "checkbox" && (
        <input {...common} type="checkbox" checked={!!value}
          onChange={e => onChange(field.name, e.target.checked)} />
      )}
      {field.kind === "select" && (
        <select {...common} value={String(value ?? "")}
          onChange={e => onChange(field.name, e.target.value)}>
          {field.choices?.map(c => <option key={c} value={c}>{c}</option>)}
        </select>
      )}
      {(field.kind === "file" || field.kind === "files") && (
        <input {...common} type="file"
          multiple={field.kind === "files"}
          accept={field.accept}
          onChange={e => onChange(field.name, Array.from(e.target.files ?? []))} />
      )}
      {field.help && <small>{field.help}</small>}
    </div>
  );
}



// ---------- component ----------

export default function UtilityPage({ spec, input, media, operation, onJumpToInput, values }: Props) {
  const [formError, setFormError] = useState("");

  // Run lifecycle lives in a module-level store (Phase 8) so results survive this page's
  // unmount — it is keyed by spec.key and remounts on every tool switch. The submit
  // writes to the store even after unmount, so switching away and back shows the run.
  const run = useSyncExternalStore(subscribe, () => getLatestForSpec(spec.key));
  const loading = run?.status === "running";

const fileMetadata = useMemo(() => {
  if (input.inputMode !== "file") return [];

  const uploaded = input.uploadedFiles ?? [];
  const local = input.files ?? [];

  return [
    ...uploaded.map((file) => ({
      name: file.name,
      type: file.type,
    })),
    ...local.map((file) => ({
      name: file.name,
      type: file.type,
    })),
  ];
}, [input]);

const isCompatible = useMemo(() => {
  if (input.inputMode !== "file") return true;
  if (fileMetadata.length === 0) return true;

  return fileMetadata.every((file) => {
    const kind = mimeToMediaKind(file.type);
    if (!kind) return true;
    return spec.accepts.includes(kind);
  });
}, [input.inputMode, fileMetadata, spec.accepts]);

  async function runTool() {
    setFormError("");

    // 1. Validate required fields (sourced + user-controlled). Validation errors are
    //    input-level — shown on the form, not modeled as a run.
    for (const f of spec.fields) {
      if (!f.required) continue;
      if (f.source) {
        const r = resolveSourcedField(f, input);
        if (r.status !== "ok") { setFormError(`${f.label}: ${r.reason}`); return; }
        continue;
      }
      if (!isFieldVisible(f, values)) continue;
if (fileFieldCanUseReceptacle(f, input)) {
  if (hasUploadedSelectedFiles(input)) {
    const r = resolveSourcedField(asReceptacleSourcedField(f), input);

    if (r.status !== "ok") {
      setFormError(`${f.label}: ${r.reason}`);
      return;
    }
  }

  // Local input.files is valid too.
  // submitPage will fall back to input.files for multipart upload.
  continue;
}

const v = values[f.name];

const empty =
  f.kind === "files" || f.kind === "file"
    ? !asFileArray(v).length
    : v === "" || v == null;

if (empty) {
  setFormError(`${f.label} is required.`);
  return;
}
    }

    // 2. Begin a run in the store (survives unmount) and execute.
    const controller = new AbortController();
    const runId = beginRun(spec.key, controller);

    // A scalar file tool pointed at SEVERAL selected files is a batch: every file
    // is processed, one after another (k65). 0–1 files → [] → the untouched
    // single-call paths below.
    const batchFiles = batchFilesFor(spec, input);
    const isChain = spec.key.startsWith("chain:");

    try {
      if (batchFiles.length > 1) {
        await runBatch(runId, controller, batchFiles, isChain);
      } else if (isChain) {
        const data = await runChainFor(
          input,
          controller,
          (steps) => setRunSteps(runId, steps), // live per-step status
        );

        if (controller.signal.aborted) cancelRun(runId);
        else completeRun(runId, data, data);
      } else {
        const r = await submitPage(
          spec,
          input,
          media,
          operation,
          values,
          {},
          { signal: controller.signal },
        );

        if (r.ok) {
          completeRun(runId, r.value);
        } else {
          const err = errorOf(r);
          if (err.kind === "aborted") cancelRun(runId);
          else failRun(runId, describeAppError(err));
        }
      }
    } catch (err) {
      failRun(runId, err instanceof Error ? err.message : String(err));
    }
  }

  // One chain execution over `stepInput` — shared by the plain chain run and by
  // each file of a BATCHED chain, so the two can't drift apart.
  async function runChainFor(
    stepInput: MediaInputValue,
    controller: AbortController,
    onProgress?: (steps: ChainStepResult[]) => void,
  ): Promise<ChainStepResult[]> {
    return runChain(
      spec.key.slice("chain:".length),
      stepInput,
      async (pageKey, si, overrides) => {
        const page = getPage(pageKey);

        const opOverride = OperationSchema.safeParse(overrides.__op);
        const stepOperation: Operation | "any" = opOverride.success
          ? opOverride.data
          : operation;

        const r = await submitPage(
          page,
          si,
          page.accepts[0] ?? media,
          stepOperation,
          values,
          overrides,
          { signal: controller.signal },
        );
        // runChain records this per-step; surface a kind-aware message.
        if (!r.ok) throw new Error(describeAppError(errorOf(r)));
        return r.value;
      },
      onProgress,
    );
  }

  /**
   * Run the tool once per selected file (operator ask 2026-08-04, k65 — the
   * console used to submit ONE file and silently ignore the rest).
   *
   * Sequential on purpose: the /ml amenities are one-file-per-call and the GPU
   * ones serialize anyway. Each file is pinned by narrowing the media input to
   * that single id, so every derived value (the resolved field, the `source`
   * payload) matches the file actually being processed. A file that fails records
   * its error in its own row and the batch CONTINUES; Cancel aborts the rest via
   * the run's own AbortController.
   */
  async function runBatch(
    runId: string,
    controller: AbortController,
    files: UploadedFileRef[],
    isChain: boolean,
  ): Promise<void> {
    const rows: BatchFileResult[] = files.map((f) => ({
      pageKey: spec.key,
      batchFile: f.name,
      status: "pending",
      ok: false,
      data: null,
    }));
    setRunSteps(runId, rows); // every file is on screen before the first one starts

    let failures = 0;

    for (let i = 0; i < files.length; i += 1) {
      if (controller.signal.aborted) break;

      rows[i] = { ...rows[i], status: "running" };
      setRunSteps(runId, [...rows]);

      const one = pinInputToFile(input, files[i]);

      try {
        if (isChain) {
          const steps = await runChainFor(one, controller);
          const failed = steps.find((s) => !s.ok);
          if (failed) throw new Error(failed.error ?? "a chain step failed");
          rows[i] = { ...rows[i], status: "ok", ok: true, data: steps };
        } else {
          const r = await submitPage(
            spec,
            one,
            media,
            operation,
            values,
            {},
            { signal: controller.signal },
          );
          if (!r.ok) throw new Error(describeAppError(errorOf(r)));
          rows[i] = { ...rows[i], status: "ok", ok: true, data: r.value };
        }
      } catch (err) {
        // A cancel lands here as the in-flight request's abort — that's not this
        // file's failure, so leave the row unstarted and stop.
        if (controller.signal.aborted) {
          rows[i] = { ...rows[i], status: "pending" };
          break;
        }
        failures += 1;
        rows[i] = {
          ...rows[i],
          status: "error",
          ok: false,
          data: null,
          error: err instanceof Error ? err.message : String(err),
        };
      }

      setRunSteps(runId, [...rows]);
    }

    setRunSteps(runId, rows); // final snapshot (also lands on an already-cancelled run)

    if (controller.signal.aborted) cancelRun(runId);
    else if (failures === rows.length) {
      failRun(runId, `All ${rows.length} files failed — each file's reason is below.`);
    } else completeRun(runId, rows, rows);
  }

  return (
    <main className={styles.page}>
      {/* Title block on the left, the fixed-place Run button opposite it on the
          right of the SAME row. The tool's options live one row up, extending the
          operation dropdown row (rendered by HugpyConsole / MediaInputDropdowns). */}
      <header className={styles.header}>
        <div className={styles.headerText}>
          <p className={styles.eyebrow}>{spec.category}</p>
          <h1>{spec.title}</h1>
          {spec.description && <p className={styles.subtitle}>{spec.description}</p>}
        </div>

        <div className={styles.headerActions}>
          <button
            type="button"
            className={styles.button}
            onClick={runTool}
            disabled={loading || !isCompatible}
          >
            {loading ? "Running..." : spec.submitLabel ?? "Run"}
          </button>

          {loading && run && (
            <button
              type="button"
              className={styles.buttonGhost}
              onClick={() => cancelRun(run.id)}
            >
              Cancel
            </button>
          )}
        </div>
      </header>

      {!isCompatible && (
        <div className={styles.error}>
          This tool accepts: {spec.accepts.join(", ")}. You uploaded:{" "}
          {fileMetadata
            .map((m) => mimeToMediaKind(m.type) ?? m.type ?? "?")
            .join(", ")}
          .
        </div>
      )}

      {formError && (
        <div className={styles.error} role="alert">
          <span>{formError}</span>
          <button type="button" className={styles.linkBtn} onClick={onJumpToInput}>
            Go to input
          </button>
        </div>
      )}

      <section className={styles.resultPanel} aria-live="polite">
        <h2>Execution Output</h2>

        {!formError && !run && <p className={styles.muted}>Output appears here.</p>}

        {/* Running with no steps yet (non-chain, or chain pre-first-step). */}
        {!formError && run?.status === "running" && !run.steps && (
          <p className={styles.muted}>Working...</p>
        )}

        {!formError && run?.status === "cancelled" && (
          <p className={styles.muted}>Cancelled.</p>
        )}

        {!formError && run?.status === "error" && (
          <pre className={`${styles.resultText} ${styles.resultError}`} role="alert">
            {run.error}
          </pre>
        )}

        {/* Completed non-chain run, or any run with chain steps (live or final). */}
        {!formError && (run?.steps || run?.status === "ok") && (
          <ExecutionOutput
            result={run.steps ?? run.result}
            spec={spec}
            operation={operation}
          />
        )}
      </section>
    </main>
  );
}