// Phase 4 — runtime validation at the trust boundaries (parse, don't assert).
// One place that mirrors the hand-written contracts in pageSpec.ts as zod schemas,
// plus small guards used to replace `as` casts on network/DOM-origin values.
import { z } from "zod";

// ---- enums (mirror pageSpec.ts unions) ----
export const MediaKindSchema = z.enum([
  "text",
  "audio",
  "image",
  "video",
  "url",
  "pdf",
  "document",
]);

export const OperationSchema = z.enum([
  "summarize",
  "keywords",
  "metadata",
  "chat",
  "transcribe",
  "ocr",
  "analyze",
  "download",
  "extract_audio",
  "extract",
  "intelligence",
  "imagegen",
]);

export const InputModeSchema = z.enum(["text", "url", "file"]);

const FieldKindSchema = z.enum([
  "text",
  "textarea",
  "number",
  "checkbox",
  "select",
  "file",
  "files",
]);

const FieldSourceSchema = z.enum(["text", "url", "selectedIds"]);

// ---- spec schemas (mirror FieldSpec / PageSpec) ----
export const FieldSpecSchema = z.object({
  name: z.string(),
  label: z.string(),
  kind: FieldKindSchema,
  required: z.boolean().optional(),
  default: z.union([z.string(), z.number(), z.boolean()]).optional(),
  choices: z.array(z.string()).optional(),
  help: z.string().optional(),
  accept: z.string().optional(),
  visibleWhen: z
    .object({
      field: z.string(),
      equals: z.union([z.string(), z.number(), z.boolean()]),
    })
    .optional(),
  source: FieldSourceSchema.optional(),
});

export const PageSpecSchema = z.object({
  key: z.string(),
  title: z.string(),
  category: z.string(),
  path: z.string(),
  method: z.enum(["POST", "GET"]).optional(),
  isUpload: z.boolean().optional(),
  fields: z.array(FieldSpecSchema),
  submitLabel: z.string().optional(),
  description: z.string().optional(),
  resultKind: z.enum(["json", "text", "image-analysis"]).optional(),
  timeoutMs: z.number().optional(),
  accepts: z.array(MediaKindSchema),
  produces: z.union([z.array(OperationSchema), OperationSchema]),
});

export type PageSpecParsed = z.infer<typeof PageSpecSchema>;

// ---- response shapes (guards for ExecutionOutput; Phase 7 dispatches on these) ----
export const TranscriptionResultSchema = z
  .object({ text: z.string(), segments: z.array(z.unknown()) })
  .passthrough();

export const PdfReportSchema = z
  .object({ pages: z.array(z.unknown()) })
  .passthrough();

export const PdfTextSchema = z.array(
  z.object({ page_num: z.number(), text: z.string() }).passthrough(),
);

const KEYWORD_FIELDS = [
  "primary",
  "secondary",
  "density",
  "dropped",
  "hashtags",
  "slug_candidates",
  "meta_keywords",
] as const;

export const KeywordResultSchema = z
  .object({
    primary: z.unknown().optional(),
    secondary: z.unknown().optional(),
    density: z.unknown().optional(),
    dropped: z.unknown().optional(),
    hashtags: z.unknown().optional(),
    slug_candidates: z.unknown().optional(),
    meta_keywords: z.unknown().optional(),
  })
  .passthrough()
  // A real guard: at least one keyword field must be present (mirrors hasKeywordShape),
  // so an unrelated object surfaces a typed error instead of an empty keyword render.
  .refine((v) => KEYWORD_FIELDS.some((k) => k in v), {
    message: "no keyword fields present",
  });

// Upload responses come in many shapes; this lens validates "it's an object with
// these optional fields" without trusting any of them blindly.
const UploadNodeSchema = z
  .object({
    id: z.unknown().optional(),
    file_id: z.unknown().optional(),
    path: z.unknown().optional(),
    url: z.unknown().optional(),
    file_url: z.unknown().optional(),
    filename: z.unknown().optional(),
    name: z.unknown().optional(),
    type: z.unknown().optional(),
    size: z.unknown().optional(),
  })
  .passthrough();

export const UploadResponseSchema = UploadNodeSchema.extend({
  ok: z.boolean().optional(),
  file: UploadNodeSchema.optional(),
  data: UploadNodeSchema.optional(),
  result: UploadNodeSchema.optional(),
});
export type UploadResponseParsed = z.infer<typeof UploadResponseSchema>;

// ---- chain results (runChain output → ChainOutput renderer) ----
export const ChainStepResultSchema = z
  .object({
    pageKey: z.string(),
    ok: z.boolean(),
    data: z.unknown(),
    error: z.string().optional(),
  })
  .passthrough();
export const ChainStepResultArraySchema = z.array(ChainStepResultSchema);

// ---- batch results (one row per selected file → BatchOutput renderer) ----
// A batch row is a chain step plus the file it ran for, so `batchFile` is what
// tells the two apart when a run's steps come back as an opaque unknown
// (operator ask 2026-08-04, k65).
export const BatchFileResultSchema = ChainStepResultSchema.extend({
  batchFile: z.string(),
  status: z.enum(["pending", "running", "ok", "error"]),
});
export const BatchFileResultArraySchema = z.array(BatchFileResultSchema);

// ---- small guards (replace `as` on network/DOM-origin values) ----
export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Narrow an unknown form-state value to File[] without asserting. */
export function asFileArray(value: unknown): File[] {
  if (!Array.isArray(value)) return [];
  return value.filter((f): f is File => f instanceof File);
}

/** Parse an unknown upload response into the optional-fields lens; {} if not an object. */
export function parseUploadResponse(value: unknown): UploadResponseParsed {
  const parsed = UploadResponseSchema.safeParse(value);
  return parsed.success ? parsed.data : {};
}
