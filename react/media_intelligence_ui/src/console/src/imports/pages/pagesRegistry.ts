// src/utilities/pagesRegistry.ts
import type { PageSpec } from "./pageSpec";

const _pages = new Map<string, PageSpec>();

export function registerPage(spec: PageSpec): void {
  if (_pages.has(spec.key)) {
    throw new Error(`Page ${spec.key} already registered`);
  }
  _pages.set(spec.key, spec);
}

export function getPage(key: string): PageSpec {
  const p = _pages.get(key);
  if (!p) throw new Error(`Unknown page ${key}`);
  return p;
}

export function listPages(): PageSpec[] {
  return Array.from(_pages.values());
}

export function pagesByCategory(): Record<string, PageSpec[]> {
  const out: Record<string, PageSpec[]> = {};
  for (const p of listPages()) {
    (out[p.category] ??= []).push(p);
  }
  for (const cat of Object.keys(out)) {
    out[cat].sort((a, b) => a.title.localeCompare(b.title));
  }
  return out;
}
