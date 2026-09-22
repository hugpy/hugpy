/**
 * or-k18 / k114 — wire types for the oracle surfaces, as the backend ACTUALLY
 * returns them (routes/oracle_routes.py; oracle/steward.py HealthReport;
 * oracle/selection.py SelectionDecision; oracle/recipes/video_performance.py
 * VisualResult → the run manifest's `dag` block; oracle/contracts.py Scorecard).
 *
 * Everything optional is optional because the producing side can omit it (an
 * older server, a disabled selector, a run that never reached the DAG). Nothing
 * here is normalized into a friendlier shape — the UI reads the truth and says
 * when a field is absent.
 */

// ---------------------------------------------------------------------------
// GET /api/oracle/steward (+ POST = apply bounded rebalancing)
// ---------------------------------------------------------------------------

export type FindingSeverity = "info" | "warn" | "alarm";

export interface StewardFinding {
  kind: string; // calibration | streak | starvation | gap_rate | matrix_stale | cache | ok
  severity: FindingSeverity | string;
  capability: string | null;
  message: string;
  evidence: Record<string, unknown>;
  action: string;
}

export interface StewardReport {
  ok: boolean;
  at?: string;
  summary?: string;
  findings?: StewardFinding[];
  policy_changed?: boolean;
  policy_after?: Record<string, unknown> | null;
  applied?: boolean;
  ledger_rows?: number;
  selection_policy?: Record<string, unknown>;
  /** 503 body: selection disabled / ledger unavailable. */
  error?: string;
  ledger_path?: string;
}

// ---------------------------------------------------------------------------
// POST /api/oracle/selection
// ---------------------------------------------------------------------------

export interface CandidateVerdict {
  model_id: string;
  selected: boolean;
  score: number;
  reasons: string[];
  rejected_at: string | null;
  evidence: Record<string, unknown>;
}

export interface SelectionDecision {
  ok: boolean;
  capability: string;
  operation: string;
  model_id: string | null;
  fallback?: string | null;
  rationale: string;
  ranked: CandidateVerdict[];
  rejected: CandidateVerdict[];
  steps: string[];
  candidate_index: number;
  spread: boolean;
  gap: boolean;
  explored: boolean;
  score: number | null;
  error?: string;
}

export interface SelectionQuery {
  capability: string;
  quality?: "preview" | "balanced" | "best";
  max_seconds?: number | null;
  max_vram_gib?: number | null;
  candidate_index?: number;
  candidates?: number;
  exclude?: string[];
}

// ---------------------------------------------------------------------------
// GET /api/oracle/producers?ref=…  and  GET /api/oracle/ledger/summary
// ---------------------------------------------------------------------------

export interface ProducerRow {
  ref: string;
  capability: string;
  model_id: string;
  ts?: string | number | null;
  worker?: string | null;
}

export interface ProducerLookup {
  ok: boolean;
  ref?: string;
  producer?: ProducerRow | null;
  error?: string;
}

/** Shape is the ledger's own `summary()` dict plus ok/remote; kept open. */
export interface LedgerSummary {
  ok: boolean;
  remote?: string | null;
  error?: string;
  ledger_path?: string;
  [key: string]: unknown;
}

// ---------------------------------------------------------------------------
// GET /api/oracle/capabilities
// ---------------------------------------------------------------------------

export interface CapabilityView {
  name: string;
  model_ids?: string[];
  registry_version?: string | null;
  [key: string]: unknown;
}

export interface CapabilitiesBody {
  ok: boolean;
  count?: number;
  capabilities?: CapabilityView[];
  registry_version?: string | null;
  error?: string;
}

// ---------------------------------------------------------------------------
// The run manifest — GET /video/jobs/<id> → result.movie (performance_relay.py
// puts PerformanceResult.to_dict() there and adds `dag` = VisualResult.to_dict()
// when the visual stages ran on the durable DAG).
// ---------------------------------------------------------------------------

export interface Scorecard {
  hard_pass: boolean;
  checks?: Array<{
    name: string;
    kind: string;
    value: unknown;
    threshold: unknown;
    passed: boolean;
    detail?: unknown;
  }>;
  judge_results?: Array<{
    judge: string;
    verdict: string;
    score: number | null;
    rationale: string;
  }>;
  confidence?: number;
  disagreements?: string[];
  diagnosis?: string | null;
  repair_code?: string | null;
  recommended_repair?: string | null;
}

export interface RepairPlan {
  run_id: string;
  failed_node: string;
  code: string;
  root: string | null;
  path: string[];
  strategy: string;
  param_changes: Record<string, unknown>;
  rationale: string;
  repairable: boolean;
  budget_left: number;
}

/** A shot as the assemble node emits it (SeamExecutor._clip → "shot"). */
export interface DagShot {
  segment_id: string;
  clip_ref: string | null;
  seconds: number | null;
  keyframe_ref?: string | null;
  scorecard?: Scorecard | null;
  considered?: number;
  [key: string]: unknown;
}

export interface DagRunRecord {
  run_id: string;
  graph_id: string;
  goal_digest: string;
  revision: number;
  state: string; // running | paused | awaiting_approval | completed | failed | cancelled
  repair_budget: number;
  revisions_used: number;
  created_at: string;
  updated_at: string;
  note: string | null;
}

export interface DagBlock {
  run: DagRunRecord;
  ok: boolean;
  video_ref: string | null;
  shots: DagShot[];
  repairs: RepairPlan[];
  failed_nodes: string[];
  /** node_id → state (pending | awaiting_approval | leased | running | succeeded | failed | rejected | cancelled). */
  nodes: Record<string, string>;
  limitations: string[];
  steward: StewardReport | null;
  validation: Record<string, unknown> | null;
}

/** PerformanceResult.shots[] — the linear journal's per-shot story. */
export interface ManifestShot {
  segment_id: string;
  index: number;
  accepted: boolean;
  keyframe_ref: string | null;
  keyframe_seed: number | null;
  keyframe_candidates: number;
  keyframe_repaired: boolean;
  keyframe_scorecard: Scorecard | null;
  clip_ref: string | null;
  clip_seconds: number | null;
  clip_candidates: number;
  clip_repaired: boolean;
  scorecard: Scorecard | null;
  repair_codes: string[];
  diagnosis: string | null;
}

export interface ManifestReceipt {
  request?: Record<string, unknown>;
  capability: string;
  model_id: string;
  worker: string | null;
  started_at: string;
  ended_at: string;
  duration_s?: number;
  retries?: number;
  failure?: string | null;
  artifacts?: Array<Record<string, unknown>>;
  warnings?: string[];
  log_excerpt?: string[];
  registry_version?: string | null;
}

export interface RunManifest {
  run_id: string;
  ok: boolean;
  goal_digest?: string;
  stages?: Array<Record<string, unknown>>;
  gap?: Record<string, unknown> | null;
  stopped_after?: string | null;
  registry_version?: string | null;
  snapshot_digest?: string | null;
  audio_master_digest?: string | null;
  lock_digest?: string | null;
  segment_digests?: string[];
  validation?: Record<string, unknown> | null;
  shots?: ManifestShot[];
  video_ref?: string | null;
  scorecard?: Scorecard | null;
  receipts?: ManifestReceipt[];
  artifact_refs?: string[];
  limitations?: string[];
  warnings?: string[];
  state_path?: string;
  /** Present only when the visual stages ran on the durable DAG. */
  dag?: DagBlock | null;
  [key: string]: unknown;
}

/** GET /video/jobs/<id> — media_bus.get's read-only view. */
export interface JobView {
  job_id: string;
  name: string | null;
  status: string | null;
  result: {
    ok?: boolean;
    error?: { code: string; message: string; retryable?: boolean } | null;
    outputs?: unknown[];
    movie?: RunManifest | null;
    [key: string]: unknown;
  } | null;
  progress: Record<string, unknown> | null;
}

/** One derived row of the DAG state list. Every field says where it came from. */
export interface DagNodeRow {
  nodeId: string;
  /** "spatial" | "keyframe" | "clip" | "assemble" | "lock" | other prefix. */
  kind: string;
  segmentId: string | null;
  state: string;
  /** The ref this node produced, when the manifest carries one (shots[]). */
  producedRef: string | null;
  /** Producer model from the manifest's receipts/shots, else null (then the ledger is asked). */
  producerModel: string | null;
  producerSource: "receipt" | "ledger" | null;
  /** The journal's lease fields are NOT exported in the manifest; always null here. */
  lease: string | null;
  scorecard: Scorecard | null;
  repair: RepairPlan | null;
  failed: boolean;
}
