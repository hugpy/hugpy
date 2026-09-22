import type { Dispatch, SetStateAction } from "react";
import type {
  FieldSpec,
  MediaKind,
  Operation,
  MediaInputValue,
  PageSpec,
} from "./../imports/pages/pageSpec";
import { MediaKindSchema, OperationSchema } from "./../imports/schemas";
import { isFieldVisible, type PageValues } from "./../imports/page/submitPage";
import {styles} from "./../imports";

type MediaInputDropdownsProps = {
  source: MediaInputValue;
  media: MediaKind;
  setMedia: Dispatch<SetStateAction<MediaKind>>;
  operation: Operation | "any";
  setOperation: Dispatch<SetStateAction<Operation | "any">>;
  allowedMedia: MediaKind[];
  operationsForMedia: Operation[];
  // The selected tool's own options extend this same row (see OptionField). The
  // media itself comes from the receptacle, so source-backed and file fields are
  // not shown here — only the tool's configurable knobs.
  selectedSpec?: PageSpec;
  values: PageValues;
  onChangeField: (name: string, value: PageValues[string]) => void;
};

// One tool option, rendered in the SAME visual language as the operation select
// so the options read as a continuation of that dropdown row.
function OptionField({
  field,
  value,
  onChange,
}: {
  field: FieldSpec;
  value: PageValues[string];
  onChange: (name: string, value: PageValues[string]) => void;
}) {
  if (field.kind === "checkbox") {
    return (
      <label className={styles.optionCheckbox}>
        <input
          type="checkbox"
          checked={!!value}
          onChange={(e) => onChange(field.name, e.target.checked)}
        />
        {field.label}
      </label>
    );
  }

  return (
    <label>
      {field.label}
      {field.kind === "select" ? (
        <select
          value={String(value ?? "")}
          onChange={(e) => onChange(field.name, e.target.value)}
        >
          {field.choices?.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      ) : field.kind === "textarea" ? (
        <textarea
          rows={2}
          value={String(value ?? "")}
          onChange={(e) => onChange(field.name, e.target.value)}
        />
      ) : (
        <input
          type={field.kind === "number" ? "number" : "text"}
          value={String(value ?? "")}
          onChange={(e) =>
            onChange(
              field.name,
              field.kind === "number" ? Number(e.target.value) : e.target.value,
            )
          }
        />
      )}
    </label>
  );
}

export function MediaInputDropdowns({
  source,
  media,
  setMedia,
  operation,
  setOperation,
  allowedMedia,
  operationsForMedia,
  selectedSpec,
  values,
  onChangeField,
}: MediaInputDropdownsProps) {
  // The tool's configurable options: non-sourced, non-file fields (the media
  // itself is supplied by the receptacle), visible under the current values.
  const optionFields = (selectedSpec?.fields ?? []).filter(
    (f) =>
      !f.source &&
      f.kind !== "file" &&
      f.kind !== "files" &&
      isFieldVisible(f, values),
  );

  return (
    <section className={styles.filterPanel}>
      {source.inputMode === "file" && (
        <label>
          Media type
          <select
            value={media}
            onChange={(event) => {
              const parsed = MediaKindSchema.safeParse(event.target.value);
              if (!parsed.success) return; // ignore unknown values
              setMedia(parsed.data);
              setOperation("any");
            }}
          >
            {allowedMedia.map((kind) => (
              <option key={kind} value={kind}>
                {kind}
              </option>
            ))}
          </select>
        </label>
      )}

      <label>
        Operation
        <select
          value={operation}
          onChange={(event) => {
            const v = event.target.value;
            if (v === "any") {
              setOperation("any");
              return;
            }
            const parsed = OperationSchema.safeParse(v);
            if (parsed.success) setOperation(parsed.data); // ignore unknowns
          }}
        >
          <option value="any">Any operation</option>
          {operationsForMedia.map((op) => (
            <option key={op} value={op}>
              {op}
            </option>
          ))}
        </select>
      </label>

      {/* Selected tool's options, extending the row. */}
      {optionFields.map((f) => (
        <OptionField
          key={f.name}
          field={f}
          value={values[f.name]}
          onChange={onChangeField}
        />
      ))}
    </section>
  );
}
