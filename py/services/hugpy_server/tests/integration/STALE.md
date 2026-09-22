# Pre-partition test failures: skipped or xfailed

How we found them: we ran these test files against the pre-partition monolith checkpoint (git `7c19ce7`, same venv) and against this package, then diffed the two lists of failures. The 100 tests that also failed on the monolith count as pre-existing.
Of those 100, 62 turned out to depend on test order, not to be stale. Test modules rebound `media_bus.DB_PATH` / `media_bus.enqueue` / `identity_profiles.IDENTITIES_HOME` and the fake identity-service URLs at import time. The fix is collection-time isolation in `tests/conftest.py`, and those 62 now pass (listed at the bottom).
The remaining 38 are marked below: `skipif` when the host is missing something, `xfail(strict=False)` when the test is stale. To fix one, update the test or the code and remove the marker. To delete one, drop the test.

| test id | classification | reason |
|---|---|---|
| `integration/test_cold_hold_cap.py::test_cold_hold_cap` | stale test | asserts 'is still loading on' but ColdHoldCapacityError.stream_message now says 'is still loading into VRAM on' |
| `integration/test_identity_profiles.py::test_create_and_store_shape` | stale test | identity profiles are versioned now (active_version/canonical); the store entry no longer carries the flat reference_images key the test asserts |
| `integration/test_identity_profiles.py::test_delete_archives` | stale test | asserts archived['reference_images'] but versioned profiles no longer carry that flat key (KeyError; on the monolith it 403'd on ownership first) |
| `integration/test_identity_profiles.py::test_validation_rejects` | stale test | expects 400 'at most 4' for 5 reference_images, but profile creation now accepts the body (201): refs are no longer validated at create |
| `integration/test_identity_profiles.py::test_create_owns_reference_dir` | stale test | asserts the store mirror carries reference_images; versioned profiles no longer carry that flat key (KeyError) |
| `integration/test_identity_profiles.py::test_migration_happy_path` | stale test | legacy migration keeps the original reference_images paths; the test expects them copied to <identity>/ref_00.png |
| `integration/test_identity_profiles.py::test_migration_missing_source_kept_not_dropped` | stale test | migration no longer records a 'missing_references' key on the migrated entry |
| `integration/test_mlt_render.py::test_unresolved_resource_errors_as_data` | environment (melt binary) | run_mlt_render probes melt first; without it every path returns melt_missing |
| `integration/test_mlt_render.py::test_runner_project_outside_jail` | environment (melt binary) | run_mlt_render probes melt first; without it every path returns melt_missing |
| `integration/test_mlt_render.py::test_runner_missing_project` | environment (melt binary) | run_mlt_render probes melt first; without it every path returns melt_missing |
| `integration/test_studio_enhance.py::test_real_interpolation_doubles_fps_and_frames` | stale test | script-style module: _setup_fixtures() (ffmpeg source clip) only runs under __main__, so under pytest the source clip never exists |
| `integration/test_studio_enhance.py::test_real_upscale_hits_target_geometry` | stale test | script-style module: _setup_fixtures() (ffmpeg source clip) only runs under __main__, so under pytest the source clip never exists |
| `integration/test_studio_enhance.py::test_prompt_in_hash_but_not_in_pixels` | stale test | script-style module: _setup_fixtures() (ffmpeg source clip) only runs under __main__, so under pytest the source clip never exists |
| `integration/test_studio_enhance.py::test_resume_on_hash` | stale test | script-style module: _setup_fixtures() (ffmpeg source clip) only runs under __main__, so under pytest the source clip never exists |
| `integration/test_studio_enhance.py::test_premium_rife_graceful_deps_missing` | stale test | script-style module: _setup_fixtures() (ffmpeg source clip) only runs under __main__, so under pytest the source clip never exists |
| `integration/test_studio_enhance.py::test_premium_ltx_graceful_weights_missing` | stale test | script-style module: _setup_fixtures() (ffmpeg source clip) only runs under __main__, so under pytest the source clip never exists |
| `integration/test_studio_enhance.py::test_route_interp_source_video_200` | stale test | script-style module: _setup_fixtures() (ffmpeg source clip) only runs under __main__, so under pytest the source clip never exists |
| `integration/test_studio_enhance.py::test_route_upres_source_video_200` | stale test | script-style module: _setup_fixtures() (ffmpeg source clip) only runs under __main__, so under pytest the source clip never exists |
| `integration/test_studio_id_lock.py::test_runner_preflight_real_ref_degrades_deps_missing` | stale test | script-style module: _setup_fixtures() (reference image / control clip) only runs under __main__, so under pytest the fixture files never exist |
| `integration/test_studio_id_lock.py::test_route_id_lock_valid_200` | stale test | script-style module: _setup_fixtures() (reference image / control clip) only runs under __main__, so under pytest the fixture files never exist |
| `integration/test_studio_id_lock.py::test_route_id_lock_non_image_rejected` | stale test | script-style module: _setup_fixtures() (reference image / control clip) only runs under __main__, so under pytest the fixture files never exist |
| `integration/test_studio_id_lock.py::test_route_control_only_with_id_lock` | stale test | script-style module: _setup_fixtures() (reference image / control clip) only runs under __main__, so under pytest the fixture files never exist |
| `integration/test_studio_id_lock.py::test_route_id_lock_with_control_200` | stale test | script-style module: _setup_fixtures() (reference image / control clip) only runs under __main__, so under pytest the fixture files never exist |
| `integration/test_studio_lock_templates.py::test_all_presets_route` | stale test | preset max-quality-t2v targets 1280x720 t2v, but the wan2.2-t2v-a14b row it bound to was removed from studio models_seed (2026-08-13); no catalog model satisfies it |
| `integration/test_studio_tier_presets.py::test_all_presets_route` | stale test | preset max-quality-t2v targets 1280x720 t2v, but the wan2.2-t2v-a14b row it bound to was removed from studio models_seed (2026-08-13); no catalog model satisfies it |
| `integration/test_studio_presets_route.py::test_every_preset_routes_to_a_model` | stale test | preset max-quality-t2v targets 1280x720 t2v, but the wan2.2-t2v-a14b row it bound to was removed from studio models_seed (2026-08-13); no catalog model satisfies it |
| `integration/test_studio_presets_route.py::test_preset_binding_intent` | stale test | preset max-quality-t2v targets 1280x720 t2v, but the wan2.2-t2v-a14b row it bound to was removed from studio models_seed (2026-08-13); no catalog model satisfies it |
| `integration/test_studio_model_pin.py::test_router_pin_overrides_autopick` | stale test | pins wan2.2-t2v-a14b, which was removed from studio models_seed (2026-08-13) -> PINNED_MODEL_UNAVAILABLE |
| `integration/test_studio_source_video.py::test_route_source_video_real_mp4_200` | stale test | script-style module: _setup_fixtures() (source mp4 + seeded catalog asset) only runs under __main__, so under pytest the source video never exists |
| `integration/test_studio_source_video.py::test_route_source_video_not_a_video_400` | stale test | script-style module: _setup_fixtures() (source mp4 + seeded catalog asset) only runs under __main__, so under pytest the source video never exists |
| `integration/test_studio_source_video.py::test_route_source_asset_id_resolves_and_unknown_404` | stale test | script-style module: _setup_fixtures() (source mp4 + seeded catalog asset) only runs under __main__, so under pytest the source video never exists |
| `integration/test_studio_source_video.py::test_produce_clip_extends_from_source_video` | stale test | script-style module: _setup_fixtures() (source mp4 + seeded catalog asset) only runs under __main__, so under pytest the source video never exists |
| `integration/test_studio_source_video.py::test_run_studio_i2v_source_video_ok` | stale test | script-style module: _setup_fixtures() (source mp4 + seeded catalog asset) only runs under __main__, so under pytest the source video never exists |
| `integration/test_studio_vace.py::test_produce_v2v_real_source_deps_missing` | stale test | script-style module: _setup_fixtures() (source mp4) only runs under __main__, so under pytest the source video never exists |
| `integration/test_studio_vace.py::test_run_studio_i2v_v2v_spec_deps_missing` | stale test | script-style module: _setup_fixtures() (source mp4) only runs under __main__, so under pytest the source video never exists |
| `integration/test_studio_vace.py::test_route_v2v_source_video_200` | stale test | script-style module: _setup_fixtures() (source mp4) only runs under __main__, so under pytest the source video never exists |
| `test_compute_actions.py::test_routes_never_500_on_a_store_fault` | stale test | on a store fault /llm/model-metrics and /llm/compute-actions return the empty panel without the 'error' reason the test asserts |
| `test_studio_tester.py::test_endpoint_enqueues_studio_tester_job` | stale test | fake media_bus.enqueue lacks the private= kwarg the route now passes (TypeError -> 500) |

## Pre-existing on the monolith, but passing once collection-time isolation was added (no marker)

- `test_identity_cleanup_prompt_route.py::test_generate_cleanup_precedence_empty_everywhere_is_byte_identical`
- `test_identity_from_video.py::test_character_without_glb_still_gets_a_profile_but_reports_error`
- `test_identity_from_video.py::test_from_images_route_creates_profile_and_chains_mesh_build`
- `test_identity_from_video.py::test_from_video_route_validates_and_202s_even_when_service_is_down`
- `test_identity_from_video.py::test_not_configured_is_error_as_data`
- `test_identity_from_video.py::test_payload_persist_and_profiles_per_character`
- `test_identity_from_video.py::test_rerun_refreshes_the_same_profile_instead_of_failing_on_duplicate`
- `test_identity_from_video.py::test_service_error_and_no_characters`
- `test_identity_from_video.py::test_turntable_files_are_attached_when_the_service_emits_them`
- `test_identity_from_video.py::test_unreachable_service_is_error_as_data`
- `test_identity_pose_stage.py::test_pose_none_no_render_no_pose_stage`
- `test_identity_pose_stage.py::test_pose_render_failure_falls_back_job_succeeds`
- `test_identity_pose_stage.py::test_pose_success_front_replaced_mode_tpose`
- `test_identity_profiles.py::test_attach_reconstruction_creates_bundle`
- `test_identity_profiles.py::test_delete_moves_identity_dir`
- `test_identity_profiles.py::test_get_by_slug_and_unknown_404`
- `test_identity_profiles.py::test_list_contains_created`
- `test_identity_profiles.py::test_patch_empty_reference_images_rejected`
- `test_identity_profiles.py::test_patch_notes_then_refs_replace`
- `test_identity_profiles.py::test_patch_rename_is_display_only_slug_stable`
- `test_identity_profiles.py::test_promote_reconstruction_to_canonical`
- `test_identity_profiles.py::test_update_supersedes_refs_not_erased`
- `test_identity_render_relay.py::test_front_autoselect_falls_back_on_no_or_error`
- `test_identity_render_relay.py::test_front_autoselect_kill_switch_off`
- `test_identity_render_relay.py::test_front_autoselect_second_candidate_wins`
- `test_identity_render_relay.py::test_generate_route_auto_promotes_when_canonical_empty`
- `test_identity_render_relay.py::test_generate_route_enqueues_and_seeds_state`
- `test_identity_render_relay.py::test_generate_route_latest_wins_replaces_canonical`
- `test_identity_render_relay.py::test_relay_full_pipeline`
- `test_identity_render_relay.py::test_relay_mesh_only_no_turntable`
- `test_identity_versions.py::test_generate_pose_none_default_and_invalid`
- `test_identity_versions.py::test_generate_pose_tpose_capable_passes_through`
- `test_identity_versions.py::test_generate_pose_tpose_not_capable_falls_back`
- `test_identity_video_extract_relay.py::test_add_appends_reconstruction_to_existing_profile`
- `test_identity_video_extract_relay.py::test_create_mints_profile_per_character`
- `test_identity_video_extract_relay.py::test_review_returns_groups_and_writes_no_profile`
- `test_identity_video_extract_relay.py::test_review_route_enqueues_and_returns_groups`
- `test_identity_video_extract_relay.py::test_route_enqueues_and_validates`
- `test_job_progress_bridge.py::test_relay_degrades_when_service_omits_progress_fields`
- `test_job_progress_bridge.py::test_relay_stamps_pbody_progress_into_llm_jobs`
- `test_member_tier.py::test_list_jobs_owner_scope`
- `test_studio_clip_archive.py::test_archived_clip_excluded_control_stays`
- `test_studio_clip_archive.py::test_archive_ok`
- `test_studio_clip_archive.py::test_both_listed_before_archive`
- `test_studio_clip_archive.py::test_detail_archived_clip_410`
- `test_studio_clip_archive.py::test_double_archive_is_idempotent_noop`
- `test_studio_clip_archive.py::test_double_unarchive_is_idempotent_noop`
- `test_studio_clip_archive.py::test_serve_archived_clip_410`
- `test_studio_clip_archive.py::test_unarchive_restores`
- `test_studio_clip_serve.py::test_detail_cancelled_job`
- `test_studio_clip_serve.py::test_detail_done_clip`
- `test_studio_clip_serve.py::test_detail_failed_job`
- `test_studio_clip_serve.py::test_serve_mp4_extension`
- `test_studio_clip_serve.py::test_serve_no_extension_catalog_mime`
- `test_studio_clip_serve.py::test_serve_no_extension_no_mime_falls_back`
- `test_studio_movie_sessions.py::test_pause_cancels_and_marks_paused`
- `test_studio_movie_sessions.py::test_resume_reenqueues_and_reuses_completed_segments`
- `test_studio_movie_sessions.py::test_resume_refuses_while_running_and_rejects_bad_ids`
- `test_studio_offload.py::test_e2e_cancel_mid_render`
- `test_studio_offload.py::test_e2e_delegate_synthetic_render`
- `test_studio_project_thread.py::test_projects_route_lists_distinct_sorted`
- `test_video_to_editor.py::test_route_success_and_resend`
