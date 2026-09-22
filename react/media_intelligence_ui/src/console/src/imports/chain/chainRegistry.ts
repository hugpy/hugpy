// chainRegistry.ts
import type { ChainSpec, ChainStep } from "./chainSpec";
import { getPage, registerPage } from "./../pages/pagesRegistry";
import { OPERATION_OUTPUT,FieldSpec, type PageSpec, type Operation } from "./../pages/pageSpec";

const _chains = new Map<string, ChainSpec>();

function asArray<T>(v: T | T[]): T[] { return Array.isArray(v) ? v : [v]; }
function chainAsPage(spec: ChainSpec): PageSpec {
  const first = getPage(spec.steps[0].pageKey);
  const last = getPage(spec.steps[spec.steps.length - 1].pageKey);
  const lastOps = asArray(last.produces);
  const finalOp = stepProduces(spec.steps[spec.steps.length - 1]);

  // Start with first step's fields. Drop any that the chain pins via overrides.
  const firstOverrides = spec.steps[0].overrides ?? {};
  const fields: FieldSpec[] = first.fields
    .filter(f => !(f.name in firstOverrides))
    .map(f => ({ ...f }));

  // Optionally surface later-step tunables. Keep this opt-in; default is hidden.
  // (Pattern: prefix later-step field names with `step{N}_` to avoid collisions.)

  return {
    key: `chain:${spec.key}`,
    title: spec.title,
    category: spec.category,
    description: spec.description,
    path: `/__chain/${spec.key}`,            // ChainRuntime intercepts this
    isUpload: first.isUpload,
    accepts: first.accepts,
    produces: finalOp,
    fields,
  };
}
function stepProduces(step: ChainStep): Operation {
  const page = getPage(step.pageKey);
  // If a page produces multiple ops, the chain author must pin one.
  // We require it via override convention: overrides.__op
  const ops = asArray(page.produces);
  if (ops.length === 1) return ops[0];
  const pinned = step.overrides?.__op;
  if (!pinned || !ops.includes(pinned as Operation)) {
    throw new Error(
      `Step ${step.pageKey} produces multiple ops (${ops.join(", ")}); ` +
      `set overrides.__op to one of them.`
    );
  }
  return pinned as Operation;
}

export function validateChain(spec: ChainSpec): void {
  if (!spec.steps.length) throw new Error(`Chain ${spec.key} has no steps`);

  for (let i = 1; i < spec.steps.length; i++) {
    const prev = spec.steps[i - 1];
    const next = spec.steps[i];
    const prevOp = stepProduces(prev);
    const prevKind = OPERATION_OUTPUT[prevOp];
    const nextPage = getPage(next.pageKey);
    if (!nextPage.accepts.includes(prevKind)) {
      throw new Error(
        `Chain ${spec.key} step ${i}: ${next.pageKey} accepts ` +
        `[${nextPage.accepts.join(", ")}] but ${prev.pageKey} produces ${prevKind}.`
      );
    }
  }
}

export function registerChain(spec: ChainSpec): void {
  if (_chains.has(spec.key)) throw new Error(`Chain ${spec.key} already registered`);
  validateChain(spec);
  _chains.set(spec.key, spec);

  // Project the chain into the page registry as a synthetic PageSpec
  registerPage(chainAsPage(spec));
}

export function getChain(key: string): ChainSpec {
  const c = _chains.get(key);
  if (!c) throw new Error(`Unknown chain ${key}`);
  return c;
}

export function listChains(): ChainSpec[] {
  return Array.from(_chains.values());
}
