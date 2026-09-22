export type AnyRecord = Record<string, unknown>;

export function isRecord(value: unknown): value is AnyRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function unwrapResult(value: unknown): unknown {
  if (isRecord(value) && "result" in value) {
    return value.result;
  }

  return value;
}

export function asStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map(String).filter(Boolean);
}

export function asDensityEntries(value: unknown): Array<[string, unknown]> {
  if (!isRecord(value)) return [];
  return Object.entries(value);
}

export function getNestedRecord(value: unknown, key: string): AnyRecord | null {
  if (!isRecord(value)) return null;

  const nested = value[key];

  return isRecord(nested) ? nested : null;
}
export function hasPdfReportShape(value: unknown): boolean {
  if (!isRecord(value)) return false;

  return Array.isArray(value.pages);
}
export function hasPdfTextShape(value: unknown): boolean {
  if (!Array.isArray(value)) return false;

  return value.some((item) => {
    return (
      isRecord(item) &&
      typeof item.page_num === "number" &&
      typeof item.text === "string"
    );
  });
}
export function hasTranscriptionShape(value: unknown): boolean {
  if (!isRecord(value)) return false;

  return (
    typeof value.text === "string" &&
    Array.isArray(value.segments)
  );
}
export function hasKeywordShape(value: unknown): boolean {
  if (!isRecord(value)) return false;

  return (
    "primary" in value ||
    "secondary" in value ||
    "density" in value ||
    "dropped" in value ||
    "hashtags" in value ||
    "slug_candidates" in value ||
    "meta_keywords" in value
  );
}

export function formatUnknown(value: unknown): string {
  if (value == null) return "";

  if (typeof value === "string") return value;

  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}
