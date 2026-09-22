import type { InputMode, UploadedFileRef } from "./../../../receptacle";
import type { MediaKind } from "./../imports/pages/pageSpec";

export const MEDIA_BY_INPUT_MODE: Record<InputMode, MediaKind[]> = {
  text: ["text"],
  url: ["url"],
  file: ["audio", "image", "video", "pdf", "document"],
};
export function smart_split(s: string): string[] {
  const out: string[] = [];
  let depth = 0;
  let buf = "";

  for (let i = 0; i < s.length; i++) {
    const c = s[i];

    if (c === "[") {
      depth++;
      buf += c;
      continue;
    }

    if (c === "]") {
      depth--;
      buf += c;
      continue;
    }

    // Split ONLY when depth === 0
    if (c === "," && depth === 0) {
      out.push(buf);
      buf = "";
      continue;
    }

    buf += c;
  }

  if (buf.length > 0) out.push(buf);
  return out;
}
export function split_outside_brackets(s: string): string[] {
  let depth = 0;
  let start = 0;
  const out: string[] = [];

  for (let i = 0; i < s.length; i++) {
    const c = s[i];

    if (c === "[") depth++;
    else if (c === "]") depth--;

    // split only when NOT inside brackets
    if (c === "," && depth === 0) {
      out.push(s.slice(start, i));
      start = i + 1;
    }
  }

  out.push(s.slice(start));
  return out;
}

export function smart_split_preserving_lists(s: string): string[] {
  let depth = 0;
  let start = 0;
  const parts: string[] = [];

  for (let i = 0; i < s.length; i++) {
    const c = s[i];

    if (c === "[") depth++;
    else if (c === "]") depth--;

    // Split ONLY when depth == 0 AND a comma appears
    if (c === "," && depth === 0) {
      parts.push(s.slice(start, i));
      start = i + 1;
    }
  }

  parts.push(s.slice(start));

  return parts;
}


export function parse_nested_list(s: string): any[] {
  // strip outer brackets
  const inner = s.slice(1, -1).trim();

  const parts = split_outside_brackets(inner);
  const result: any[] = [];

  for (let p of parts) {
    p = p.trim();

    if (p.startsWith("[") && p.endsWith("]")) {
      // recursively parse nested list
      result.push(parse_nested_list(p));
    } else {
      result.push(p);
    }
  }

  return result;
}

export function asArray(obj: any): any[] {
  if (obj == null) return [];

  if (Array.isArray(obj)) {
    return obj.flatMap(item => asArray(item));
  }

  const s = String(obj).trim();

  if (s.startsWith("[") && s.endsWith("]")) {
    return parse_nested_list(s);
  }

  return split_outside_brackets(s).map(x => x.trim());
}


export function inferFileMedia(
  files: File[],
  uploaded: UploadedFileRef[],
): MediaKind | null {
  const sources: Array<{ type?: string; name?: string }> = [
    ...uploaded,
    ...files,
  ];

  for (const source of sources) {
    const type = source.type ?? "";
    const name = source.name?.toLowerCase() ?? "";

    if (type === "application/pdf" || name.endsWith(".pdf")) return "pdf";
    if (
      /\.(docx?|xlsx?|csv|txt|md)$/.test(name) ||
      type.includes("officedocument") ||
      type.includes("ms-excel") ||
      type.startsWith("text/")
    )
      return "document";
    if (type.startsWith("audio/")) return "audio";
    if (type.startsWith("video/")) return "video";
    if (type.startsWith("image/")) return "image";
  }

  return null;
}
