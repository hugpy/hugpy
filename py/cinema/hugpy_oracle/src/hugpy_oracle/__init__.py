"""hugpy_oracle: decisions and plans over Hugpy's execution packages.

The oracle owns creative planning, model selection, screenplay/storyboard,
the plan DAG runtime, repair/evaluation, benchmarks, the reliability ledger
and scorecards. It produces IMMUTABLE plans/specs and submits them to
``hugpy_video`` through video's public job API; it never writes video's
state files. It imports ``hugpy_platform``, ``hugpy_control``,
``hugpy_engine``, ``hugpy_media`` and ``hugpy_video`` only; the fleet, the
curation store and the HTTP server reach it through the Protocols in
:mod:`hugpy_oracle.providers` (fleet/curation) and ``hugpy_video.hooks``
(video), installed by the composition root.

Import is cheap by construction: every public name below is resolved LAZILY
on first attribute access (PEP 562), so ``import hugpy_oracle`` pulls no
registry, no catalog and no third-party stack. ``from hugpy_oracle import X``
keeps working for every name in ``__all__``.

Module map (historical wave labels kept for the record): contracts (k90a),
router/runtime/scorecard (k90b), evaluation/repair (k90c), authority (k97),
speech (k98), catalog + probes + schema_export (k101), audio_master (k102),
plan + validator (k103), production + segments (k104), dag_runtime /
repair_controller / selection / prompt_compiler / steward / spatial /
recipes (k111-k113), relay (k106 / k121).
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.0"

# public name -> (submodule, attribute). One entry per exported name.
_LAZY: dict[str, tuple[str, str]] = {
    # audio_master
    'AudioBuildResult': ('audio_master', 'AudioBuildResult'),
    'AudioGap': ('audio_master', 'AudioGap'),
    'AudioMaster': ('audio_master', 'AudioMaster'),
    'DialogueTimeline': ('audio_master', 'DialogueTimeline'),
    'Line': ('audio_master', 'Line'),
    'LineTiming': ('audio_master', 'LineTiming'),
    'SpeechCandidate': ('audio_master', 'SpeechCandidate'),
    'SpeechPolicy': ('audio_master', 'SpeechPolicy'),
    'VoiceKind': ('audio_master', 'VoiceKind'),
    'VoiceProfile': ('audio_master', 'VoiceProfile'),
    'WordTiming': ('audio_master', 'WordTiming'),
    'build_audio_master': ('audio_master', 'build_audio_master'),
    'candidate_seed': ('audio_master', 'candidate_seed'),
    # authority
    'AuthorityDecision': ('authority', 'AuthorityDecision'),
    'CAPABILITY_ACCESS': ('authority', 'CAPABILITY_ACCESS'),
    'IDENTITY_CONDITIONED': ('authority', 'IDENTITY_CONDITIONED'),
    'VOICE_CONDITIONED': ('authority', 'VOICE_CONDITIONED'),
    'check_authority': ('authority', 'check'),
    'refusal_receipt': ('authority', 'refusal_receipt'),
    'refusal_scorecard': ('authority', 'refusal_scorecard'),
    'required_authorities': ('authority', 'required_authorities'),
    # catalog
    'LEGACY_TASK_CAPABILITY': ('catalog', 'LEGACY_TASK_CAPABILITY'),
    'LEGACY_TASK_EXCLUDED': ('catalog', 'LEGACY_TASK_EXCLUDED'),
    'STUDIO_CAPABILITY_EXCLUDED': ('catalog', 'STUDIO_CAPABILITY_EXCLUDED'),
    'STUDIO_CAPABILITY_NAME': ('catalog', 'STUDIO_CAPABILITY_NAME'),
    'capability_version': ('catalog', 'capability_version'),
    'get_capability': ('catalog', 'get_capability'),
    'list_capabilities': ('catalog', 'list_capabilities'),
    'registry_snapshot': ('catalog', 'registry_snapshot'),
    'registry_version': ('catalog', 'registry_version'),
    'resolve_owners': ('catalog', 'resolve_owners'),
    'unmapped_tasks': ('catalog', 'unmapped_tasks'),
    # contracts
    'AccessKind': ('contracts', 'AccessKind'),
    'ArtifactKind': ('contracts', 'ArtifactKind'),
    'ArtifactRef': ('contracts', 'ArtifactRef'),
    'AuthorityKind': ('contracts', 'AuthorityKind'),
    'Authorization': ('contracts', 'Authorization'),
    'BudgetHints': ('contracts', 'BudgetHints'),
    'CapabilityView': ('contracts', 'CapabilityView'),
    'Check': ('contracts', 'Check'),
    'CheckKind': ('contracts', 'CheckKind'),
    'DEFAULT_CAPABILITY_VERSION': ('contracts', 'DEFAULT_CAPABILITY_VERSION'),
    'Eligibility': ('contracts', 'Eligibility'),
    'ExecutionReceipt': ('contracts', 'ExecutionReceipt'),
    'FailureClass': ('contracts', 'FailureClass'),
    'FrozenMap': ('contracts', 'FrozenMap'),
    'GoalSpec': ('contracts', 'GoalSpec'),
    'InputKind': ('contracts', 'InputKind'),
    'InputRef': ('contracts', 'InputRef'),
    'JudgeResult': ('contracts', 'JudgeResult'),
    'PlannerMode': ('contracts', 'PlannerMode'),
    'ProbeCheck': ('contracts', 'ProbeCheck'),
    'ProbeResult': ('contracts', 'ProbeResult'),
    'ProbeStatus': ('contracts', 'ProbeStatus'),
    'Provenance': ('contracts', 'Provenance'),
    'QualityProfile': ('contracts', 'QualityProfile'),
    'RepairCode': ('contracts', 'RepairCode'),
    'ResourceHints': ('contracts', 'ResourceHints'),
    'RightsManifest': ('contracts', 'RightsManifest'),
    'Scorecard': ('contracts', 'Scorecard'),
    'SourceRegistry': ('contracts', 'SourceRegistry'),
    'canonical_json': ('contracts', 'canonical_json'),
    'coerce_artifact_kind': ('contracts', 'coerce_artifact_kind'),
    # dag_runtime
    'DagRuntime': ('dag_runtime', 'DagRuntime'),
    'JournalError': ('dag_runtime', 'JournalError'),
    'NodeContext': ('dag_runtime', 'NodeContext'),
    'NodeRecord': ('dag_runtime', 'NodeRecord'),
    'NodeResult': ('dag_runtime', 'NodeResult'),
    'NodeState': ('dag_runtime', 'NodeState'),
    'RepairBudgetExceeded': ('dag_runtime', 'RepairBudgetExceeded'),
    'ResourceBroker': ('dag_runtime', 'ResourceBroker'),
    'RunJournal': ('dag_runtime', 'RunJournal'),
    'RunRecord': ('dag_runtime', 'RunRecord'),
    'RunState': ('dag_runtime', 'RunState'),
    'StepReport': ('dag_runtime', 'StepReport'),
    'derive_cache_key': ('dag_runtime', 'derive_cache_key'),
    # evaluation
    'DEFAULT_EVALUATED': ('evaluation', 'DEFAULT_EVALUATED'),
    'RUBRICS': ('evaluation', 'RUBRICS'),
    'THRESHOLDS': ('evaluation', 'THRESHOLDS'),
    'evaluate': ('evaluation', 'evaluate'),
    'parse_judge_verdict': ('evaluation', 'parse_judge_verdict'),
    # plan
    'AcceptanceTest': ('plan', 'AcceptanceTest'),
    'CycleError': ('plan', 'CycleError'),
    'Edge': ('plan', 'Edge'),
    'FrozenParams': ('plan', 'FrozenParams'),
    'NodeKind': ('plan', 'NodeKind'),
    'PlanGraph': ('plan', 'PlanGraph'),
    'PlanNode': ('plan', 'PlanNode'),
    'Port': ('plan', 'Port'),
    'ResourceRequest': ('plan', 'ResourceRequest'),
    'RetryPolicy': ('plan', 'RetryPolicy'),
    'goal_digest': ('plan', 'goal_digest'),
    'sibling_check': ('plan', 'sibling_check'),
    'sibling_violations': ('plan', 'sibling_violations'),
    # probes
    'PROBE_BUDGET_S': ('probes', 'PROBE_BUDGET_S'),
    'PROBE_SPECS': ('probes', 'PROBE_SPECS'),
    'ProbeSpec': ('probes', 'ProbeSpec'),
    'probe_capability': ('probes', 'probe_capability'),
    'register_probe': ('probes', 'register_probe'),
    'run_probe': ('probes', 'run_probe'),
    # production
    'CAMERA_KEYS': ('production', 'CAMERA_KEYS'),
    'CAMERA_MOVES': ('production', 'CAMERA_MOVES'),
    'CAMERA_VIEWS': ('production', 'CAMERA_VIEWS'),
    'ContinuityBible': ('production', 'ContinuityBible'),
    'ContinuityState': ('production', 'ContinuityState'),
    'GenerationSnapshot': ('production', 'GenerationSnapshot'),
    'LockRefused': ('production', 'LockRefused'),
    'ProductionError': ('production', 'ProductionError'),
    'ProductionLock': ('production', 'ProductionLock'),
    'RunPromptLedger': ('production', 'RunPromptLedger'),
    'RunPromptRefused': ('production', 'RunPromptRefused'),
    'SHOT_SIZES': ('production', 'SHOT_SIZES'),
    'ShotPlan': ('production', 'ShotPlan'),
    'ShotPlanEntry': ('production', 'ShotPlanEntry'),
    'prompt_digest': ('production', 'prompt_digest'),
    # prompt_compiler
    'ContextPlan': ('prompt_compiler', 'ContextPlan'),
    'compile_context': ('prompt_compiler', 'compile_context'),
    'render_prompt': ('prompt_compiler', 'render_prompt'),
    # recipes.video_performance
    'VisualResult': ('recipes.video_performance', 'VisualResult'),
    'build_visual_graph': ('recipes.video_performance', 'build_visual_graph'),
    'resume_visual_stages': ('recipes.video_performance', 'resume_visual_stages'),
    'run_performance_on_dag': ('recipes.video_performance', 'run_performance_on_dag'),
    'run_visual_stages': ('recipes.video_performance', 'run_visual_stages'),
    # repair
    'RepairDecision': ('repair', 'RepairDecision'),
    'attempt_repair': ('repair', 'attempt_repair'),
    'execute_repair': ('repair', 'execute_repair'),
    # repair_controller
    'RepairController': ('repair_controller', 'RepairController'),
    'RepairPlan': ('repair_controller', 'RepairPlan'),
    'RepairPolicy': ('repair_controller', 'RepairPolicy'),
    # router
    'CAPABILITY_TASK': ('router', 'CAPABILITY_TASK'),
    'RouteDecision': ('router', 'RouteDecision'),
    'RouteRefusal': ('router', 'RouteRefusal'),
    'infer_capability': ('router', 'infer_capability'),
    'resolve_route': ('router', 'resolve_route'),
    # runtime
    'DispatchTimeout': ('runtime', 'DispatchTimeout'),
    'GoalShapeError': ('runtime', 'GoalShapeError'),
    'SYNC_DEADLINE_ENV': ('runtime', 'SYNC_DEADLINE_ENV'),
    'execute_route': ('runtime', 'execute_route'),
    'run_bounded': ('runtime', 'run_bounded'),
    'sync_deadline_s': ('runtime', 'sync_deadline_s'),
    # schema_export
    'export_all': ('schema_export', 'export_all'),
    'json_schema_for': ('schema_export', 'json_schema_for'),
    # scorecard
    'build_deferred_scorecard': ('scorecard', 'build_deferred_scorecard'),
    'build_gap_scorecard': ('scorecard', 'build_gap_scorecard'),
    'build_technical_scorecard': ('scorecard', 'build_technical_scorecard'),
    # segments
    'CompileRefused': ('segments', 'CompileRefused'),
    'JOINT_MODES': ('segments', 'JOINT_MODES'),
    'LockedContext': ('segments', 'LockedContext'),
    'LockedSegmentBrief': ('segments', 'LockedSegmentBrief'),
    'SEGMENT_CAPABILITY': ('segments', 'SEGMENT_CAPABILITY'),
    'SegmentSpec': ('segments', 'SegmentSpec'),
    'SiblingViolation': ('segments', 'SiblingViolation'),
    'assert_siblings': ('segments', 'assert_siblings'),
    'build_locked_context': ('segments', 'build_locked_context'),
    'compile_segments': ('segments', 'compile_segments'),
    'default_prompt_writer': ('segments', 'default_prompt_writer'),
    'execution_order': ('segments', 'execution_order'),
    'render_dependencies': ('segments', 'render_dependencies'),
    'segment_seed': ('segments', 'segment_seed'),
    'shot_plan_from_windows': ('segments', 'shot_plan_from_windows'),
    'shot_windows_from_audio': ('segments', 'shot_windows_from_audio'),
    'to_plan_graph': ('segments', 'to_plan_graph'),
    # selection
    'ReliabilityLedger': ('selection', 'ReliabilityLedger'),
    'SelectionDecision': ('selection', 'SelectionDecision'),
    'SelectionPolicy': ('selection', 'SelectionPolicy'),
    'Selector': ('selection', 'Selector'),
    # spatial
    'CANONICAL': ('spatial', 'CANONICAL'),
    'ConditioningRequest': ('spatial', 'ConditioningRequest'),
    'SpatialSceneManifest': ('spatial', 'SpatialSceneManifest'),
    'SpatialValidation': ('spatial', 'SpatialValidation'),
    'TierFallback': ('spatial', 'TierFallback'),
    'TierProfile': ('spatial', 'TierProfile'),
    'ToneProfile': ('spatial', 'ToneProfile'),
    'convert_points': ('spatial', 'convert_points'),
    'frame_alignment_report': ('spatial', 'frame_alignment_report'),
    'tone_profile': ('spatial', 'tone_profile'),
    'validate_manifest': ('spatial', 'validate_manifest'),
    # steward
    'HealthReport': ('steward', 'HealthReport'),
    'Steward': ('steward', 'Steward'),
    'StewardPolicy': ('steward', 'StewardPolicy'),
    # tone_scale
    'tone_to_operator': ('tone_scale', 'to_operator'),
    'tone_to_unit': ('tone_scale', 'to_unit'),
    # validator
    'ErrorCode': ('validator', 'ErrorCode'),
    'ValidationError': ('validator', 'ValidationError'),
    'ValidationReport': ('validator', 'ValidationReport'),
    'validate': ('validator', 'validate'),
    'validate_plan': ('validator', 'validate'),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'hugpy_oracle' has no attribute {name!r}") from None
    module = importlib.import_module(f"hugpy_oracle.{module_name}")
    value = getattr(module, attr)
    globals()[name] = value          # cache: later lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


def install_hooks(video_hooks: Any = None, video_jobs: Any = None) -> dict[str, Any]:
    """Install the oracle's implementations into video's seams: the prompt
    coordinator (``hugpy_video.hooks``) and the ``video_performance`` bus job
    (``hugpy_video.jobs``). Returns a report of what was wired; a video
    without those seams wires nothing."""
    from hugpy_oracle.relay.hooks import install_hooks as _install
    return _install(video_hooks, video_jobs)


__all__ = [
    "__version__", "install_hooks",
    'DagRuntime', 'JournalError', 'NodeContext', 'NodeRecord', 'NodeResult', 'NodeState',
    'RepairBudgetExceeded', 'ResourceBroker', 'RunJournal', 'RunRecord', 'RunState', 'StepReport',
    'derive_cache_key', 'RepairController', 'RepairPlan', 'RepairPolicy', 'ReliabilityLedger', 'SelectionDecision',
    'SelectionPolicy', 'Selector', 'ContextPlan', 'compile_context', 'render_prompt', 'HealthReport',
    'Steward', 'StewardPolicy', 'VisualResult', 'build_visual_graph', 'resume_visual_stages', 'run_performance_on_dag',
    'run_visual_stages', 'tone_to_operator', 'tone_to_unit', 'CANONICAL', 'ConditioningRequest', 'SpatialSceneManifest',
    'SpatialValidation', 'TierFallback', 'TierProfile', 'ToneProfile', 'convert_points', 'frame_alignment_report',
    'tone_profile', 'validate_manifest', 'ArtifactKind', 'ArtifactRef', 'Authorization', 'AuthorityKind',
    'BudgetHints', 'CapabilityView', 'Check', 'CheckKind', 'Eligibility', 'ExecutionReceipt',
    'FailureClass', 'GoalSpec', 'InputKind', 'InputRef', 'JudgeResult', 'PlannerMode',
    'QualityProfile', 'RepairCode', 'ResourceHints', 'RightsManifest', 'Scorecard', 'SourceRegistry',
    'AccessKind', 'DEFAULT_CAPABILITY_VERSION', 'FrozenMap', 'ProbeCheck', 'ProbeResult', 'ProbeStatus',
    'Provenance', 'canonical_json', 'coerce_artifact_kind', 'AuthorityDecision', 'CAPABILITY_ACCESS', 'IDENTITY_CONDITIONED',
    'VOICE_CONDITIONED', 'check_authority', 'refusal_receipt', 'refusal_scorecard', 'required_authorities', 'LEGACY_TASK_CAPABILITY',
    'LEGACY_TASK_EXCLUDED', 'STUDIO_CAPABILITY_NAME', 'STUDIO_CAPABILITY_EXCLUDED', 'get_capability', 'list_capabilities', 'resolve_owners',
    'unmapped_tasks', 'PROBE_BUDGET_S', 'PROBE_SPECS', 'ProbeSpec', 'capability_version', 'export_all',
    'json_schema_for', 'probe_capability', 'register_probe', 'registry_snapshot', 'registry_version', 'run_probe',
    'AudioBuildResult', 'AudioGap', 'AudioMaster', 'DialogueTimeline', 'Line', 'LineTiming',
    'SpeechCandidate', 'SpeechPolicy', 'VoiceKind', 'VoiceProfile', 'WordTiming', 'build_audio_master',
    'candidate_seed', 'AcceptanceTest', 'CycleError', 'Edge', 'FrozenParams', 'NodeKind',
    'PlanGraph', 'PlanNode', 'Port', 'ResourceRequest', 'RetryPolicy', 'goal_digest',
    'sibling_check', 'sibling_violations', 'ErrorCode', 'ValidationError', 'ValidationReport', 'validate',
    'validate_plan', 'CAMERA_KEYS', 'CAMERA_MOVES', 'CAMERA_VIEWS', 'SHOT_SIZES', 'ContinuityBible',
    'ContinuityState', 'GenerationSnapshot', 'LockRefused', 'ProductionError', 'ProductionLock', 'RunPromptLedger',
    'RunPromptRefused', 'ShotPlan', 'ShotPlanEntry', 'prompt_digest', 'JOINT_MODES', 'SEGMENT_CAPABILITY',
    'CompileRefused', 'LockedContext', 'LockedSegmentBrief', 'SegmentSpec', 'SiblingViolation', 'assert_siblings',
    'build_locked_context', 'compile_segments', 'default_prompt_writer', 'execution_order', 'render_dependencies', 'segment_seed',
    'shot_plan_from_windows', 'shot_windows_from_audio', 'to_plan_graph', 'CAPABILITY_TASK', 'RouteDecision', 'RouteRefusal',
    'infer_capability', 'resolve_route', 'GoalShapeError', 'execute_route', 'SYNC_DEADLINE_ENV', 'DispatchTimeout',
    'run_bounded', 'sync_deadline_s', 'build_technical_scorecard', 'build_gap_scorecard', 'build_deferred_scorecard', 'DEFAULT_EVALUATED',
    'RUBRICS', 'THRESHOLDS', 'evaluate', 'parse_judge_verdict', 'RepairDecision', 'attempt_repair',
    'execute_repair',
]
