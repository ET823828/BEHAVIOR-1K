"""Optional EmbodiedPerf integration for BEHAVIOR challenge evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any


COLD_START_EPISODES = 1


def create_behavior_trace_session(
    *,
    output_dir: str | Path,
    model_key: str,
    checkpoint: str,
    gpu_ids: Sequence[int],
    task_name: str,
    policy_name: str,
    host: str,
    port: int,
    power_interval_s: float = 0.05,
) -> Any:
    """Create one fail-closed TraceSession without making EmbodiedPerf a hard dependency."""

    for value, field_name in (
        (model_key, "model_key"),
        (checkpoint, "checkpoint"),
        (task_name, "task_name"),
        (policy_name, "policy_name"),
        (host, "host"),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65_535:
        raise ValueError("port must be an integer between 1 and 65535")
    if not gpu_ids:
        raise ValueError("gpu_ids must name every local GPU included in power and memory accounting")
    if any(isinstance(gpu_id, bool) or not isinstance(gpu_id, int) or gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError("gpu_ids must contain non-negative integers")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("gpu_ids must not contain duplicates")
    if (
        isinstance(power_interval_s, bool)
        or not isinstance(power_interval_s, (int, float))
        or not math.isfinite(power_interval_s)
        or power_interval_s <= 0
    ):
        raise ValueError("power_interval_s must be a finite positive number")

    profile_dir = Path(output_dir).expanduser().resolve()
    if profile_dir.exists() and not profile_dir.is_dir():
        raise NotADirectoryError(f"profiler output path is not a directory: {profile_dir}")
    if profile_dir.exists() and any(profile_dir.iterdir()):
        raise FileExistsError(f"profiler output directory must be empty: {profile_dir}")
    trace_path = profile_dir / "traces.jsonl"
    summary_path = profile_dir / "summary.json"
    for path in (trace_path, summary_path):
        if path.exists():
            raise FileExistsError(f"refusing to mix profiler runs in existing artifact: {path}")
    profile_dir.mkdir(parents=True, exist_ok=True)

    try:
        from embodiedperf.tracing import BenchmarkTraceRecipe, TraceSession
    except ImportError as exc:
        raise RuntimeError(
            "EmbodiedPerf profiling was requested, but embodiedperf is not installed in the evaluator environment"
        ) from exc

    recipe = BenchmarkTraceRecipe(
        benchmark="BEHAVIOR-1K/2026-challenge",
        benchmark_family="BEHAVIOR-1K",
        paradigm_default="reactive_vla",
        runtime_mode_default="websocket_closed_loop",
        measurement_method="client_observed_black_box_with_local_system_telemetry",
        language_conditioned=True,
        required_gpu_evidence=True,
        default_boundary_label="websocket_policy_round_trip",
    )
    return TraceSession(
        recipe=recipe,
        output_path=trace_path,
        model_key=model_key,
        benchmark="BEHAVIOR-1K/2026-challenge",
        runtime_mode="websocket_closed_loop",
        gpu_ids=tuple(gpu_ids),
        power_interval_s=power_interval_s,
        cuda_synchronize=False,
        runtime_timeline=True,
        semantic_stage_markers=True,
        # The evaluator cannot place CUDA events on a policy server's process/stream.
        semantic_stage_deferred_cuda=False,
        semantic_stage_applicable_kinds=(
            "cache",
            "communication_wait",
            "environment_step",
            "observation_preprocess",
        ),
        cpu_profile={
            "enabled": True,
            "interval_s": 0.05,
            "raw_sample_cap": 50_000,
            "energy": {"enabled": False},
        },
        run_metadata={
            "benchmark_protocol": "BEHAVIOR-1K 2026 challenge evaluator",
            "task_name": task_name,
            "policy_name": policy_name,
            "checkpoint": checkpoint,
            "websocket_endpoint": f"{host}:{port}",
            "gpu_scope": list(gpu_ids),
            "cold_start_definition": "first_trace_record_per_eval_process",
            "cold_start_episodes_excluded_from_statistics": COLD_START_EPISODES,
            "cpu_scope": "evaluator_process_only",
            "server_internal_l2_scope": "unavailable_without_server_side_instrumentation",
        },
    )


def _finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    if number < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return number


def _optional_values(rows: Sequence[Mapping[str, Any]], field_name: str) -> list[float]:
    values = []
    for index, row in enumerate(rows):
        value = row.get(field_name)
        if value is not None:
            values.append(_finite_float(value, f"traces[{index}].{field_name}"))
    return values


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _load_traces(trace_path: Path) -> list[dict[str, Any]]:
    rows = []
    seen_run_ids = set()
    with trace_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank trace record at line {line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"trace line {line_number} must be a JSON object")
            run_id = row.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                raise ValueError(f"trace line {line_number} has no run_id")
            if run_id in seen_run_ids:
                raise ValueError(f"duplicate trace run_id: {run_id}")
            if not isinstance(row.get("success"), bool):
                raise TypeError(f"trace line {line_number} success must be boolean")
            seen_run_ids.add(run_id)
            rows.append(row)
    return rows


def _semantic_view_functions():
    from embodiedperf.tracing.semantic_stage_views import (
        build_semantic_stage_perfetto,
        derive_semantic_stage_analysis,
        render_semantic_stage_html,
        serialize_semantic_stage_perfetto,
        validate_semantic_stage_views,
    )

    return (
        derive_semantic_stage_analysis,
        render_semantic_stage_html,
        build_semantic_stage_perfetto,
        serialize_semantic_stage_perfetto,
        validate_semantic_stage_views,
    )


def _write_json_artifact(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def materialize_cold_start_free_semantic_views(
    trace_path: str | Path,
    *,
    output_dir: str | Path,
    cold_start_episodes: int = COLD_START_EPISODES,
) -> dict[str, dict[str, str]]:
    """Render canonical-derived semantic HTML and Perfetto for warm episodes only."""

    if isinstance(cold_start_episodes, bool) or not isinstance(cold_start_episodes, int):
        raise TypeError("cold_start_episodes must be an integer")
    if cold_start_episodes < 1:
        raise ValueError("at least one cold-start episode must be excluded")
    rows = _load_traces(Path(trace_path).expanduser().resolve())
    if len(rows) <= cold_start_episodes:
        raise ValueError(
            f"profile has {len(rows)} raw episode(s), so no warm episode remains after excluding {cold_start_episodes}"
        )

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite semantic timeline directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.tmp")
    if staging.exists():
        raise FileExistsError(f"stale semantic timeline staging directory exists: {staging}")
    (
        derive_analysis,
        render_html,
        build_perfetto,
        serialize_perfetto,
        validate_views,
    ) = _semantic_view_functions()
    staging.mkdir()
    references = {}
    try:
        for raw_index, row in enumerate(rows[cold_start_episodes:], start=cold_start_episodes + 1):
            metadata = row.get("metadata")
            collection = metadata.get("semantic_stage_v1") if isinstance(metadata, Mapping) else None
            if not isinstance(collection, Mapping):
                raise ValueError(f"warm trace {row['run_id']} has no semantic_stage_v1 collection")
            expected = {"expected_run_id": row["run_id"]}
            analysis = derive_analysis(collection, **expected)
            html = render_html(collection, **expected)
            perfetto = build_perfetto(collection, **expected)
            validation = validate_views(collection, html, perfetto, **expected)
            if not isinstance(validation, Mapping) or validation.get("status") != "pass":
                raise ValueError(f"semantic view validation did not pass for {row['run_id']}")

            episode_name = f"episode_{raw_index:06d}"
            episode_dir = staging / episode_name
            episode_dir.mkdir()
            _write_json_artifact(episode_dir / "semantic_stage_collection.json", collection)
            _write_json_artifact(episode_dir / "semantic_stage_analysis.json", analysis)
            with (episode_dir / "semantic_stage.html").open("x", encoding="utf-8") as stream:
                stream.write(html)
            with (episode_dir / "semantic_stage.perfetto.json").open("x", encoding="utf-8") as stream:
                stream.write(serialize_perfetto(perfetto))
                stream.write("\n")
            _write_json_artifact(episode_dir / "cross_view_validation.json", validation)
            references[row["run_id"]] = {
                "html": f"{destination.name}/{episode_name}/semantic_stage.html",
                "perfetto": f"{destination.name}/{episode_name}/semantic_stage.perfetto.json",
                "analysis": f"{destination.name}/{episode_name}/semantic_stage_analysis.json",
                "validation": f"{destination.name}/{episode_name}/cross_view_validation.json",
            }
        os.replace(staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return references


def _l0_values(rows: Sequence[Mapping[str, Any]]) -> tuple[list[float], int]:
    values = []
    unavailable = 0
    for row_index, row in enumerate(rows):
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            raise TypeError(f"traces[{row_index}].metadata must be an object")
        readiness = metadata.get("action_readiness_v1")
        if not isinstance(readiness, Mapping) or not isinstance(readiness.get("records"), list):
            raise ValueError(f"traces[{row_index}] has no action_readiness_v1 records")
        for record_index, record in enumerate(readiness["records"]):
            if not isinstance(record, Mapping):
                raise TypeError(f"traces[{row_index}] readiness record {record_index} must be an object")
            if record.get("status") != "available":
                unavailable += 1
                continue
            values.append(
                _finite_float(
                    record.get("observation_to_action_ready_ms"),
                    f"traces[{row_index}].action_readiness_v1.records[{record_index}]",
                )
            )
    return values, unavailable


def _server_infer_values(rows: Sequence[Mapping[str, Any]]) -> tuple[list[float], list[float], int, int]:
    values = []
    current_observation_values = []
    cached_actions = 0
    unknown_provenance = 0
    for row_index, row in enumerate(rows):
        metadata = row.get("metadata")
        step_records = metadata.get("step_records", []) if isinstance(metadata, Mapping) else []
        if not isinstance(step_records, (list, tuple)):
            raise TypeError(f"traces[{row_index}].metadata.step_records must be a sequence")
        for step_index, step in enumerate(step_records):
            if not isinstance(step, Mapping):
                raise TypeError(f"traces[{row_index}] step record {step_index} must be an object")
            step_metadata = step.get("metadata")
            timing = step_metadata.get("server_timing") if isinstance(step_metadata, Mapping) else None
            provenance = step_metadata.get("action_provenance") if isinstance(step_metadata, Mapping) else None
            provenance_status = provenance.get("status") if isinstance(provenance, Mapping) else None
            if provenance_status == "current_observation_not_used":
                cached_actions += 1
            elif provenance_status != "current_observation_used":
                unknown_provenance += 1
            if isinstance(timing, Mapping) and timing.get("infer_ms") is not None:
                infer_ms = _finite_float(
                    timing["infer_ms"],
                    f"traces[{row_index}].step_records[{step_index}].server_timing.infer_ms",
                )
                values.append(infer_ms)
                if provenance_status == "current_observation_used":
                    current_observation_values.append(infer_ms)
    return values, current_observation_values, cached_actions, unknown_provenance


def summarize_cold_start_free_profile(
    trace_path: str | Path,
    *,
    output_path: str | Path,
    cold_start_episodes: int = COLD_START_EPISODES,
    semantic_view_references: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Publish aggregate statistics after excluding the first episode(s) in trace order."""

    if isinstance(cold_start_episodes, bool) or not isinstance(cold_start_episodes, int):
        raise TypeError("cold_start_episodes must be an integer")
    if cold_start_episodes < 1:
        raise ValueError("at least one cold-start episode must be excluded")
    source = Path(trace_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing profile summary: {destination}")
    rows = _load_traces(source)
    if len(rows) <= cold_start_episodes:
        raise ValueError(
            f"profile has {len(rows)} raw episode(s), so no warm episode remains after excluding {cold_start_episodes}"
        )

    excluded = rows[:cold_start_episodes]
    warm = rows[cold_start_episodes:]
    if semantic_view_references is not None:
        warm_run_ids = {row["run_id"] for row in warm}
        if set(semantic_view_references) != warm_run_ids:
            raise ValueError("semantic view references must exactly match the warm trace run ids")
        required_reference_fields = {"html", "perfetto", "analysis", "validation"}
        for run_id, reference in semantic_view_references.items():
            if not isinstance(reference, Mapping) or set(reference) != required_reference_fields:
                raise ValueError(f"semantic view reference fields are invalid for {run_id}")
            if any(not isinstance(value, str) or not value.strip() for value in reference.values()):
                raise ValueError(f"semantic view reference paths are invalid for {run_id}")
    successful = [row for row in warm if row["success"]]
    l0_values, unavailable_l0 = _l0_values(warm)
    if not l0_values:
        raise ValueError("warm profile has no available observation-to-action-ready measurements")
    (
        server_infer_values,
        server_current_observation_values,
        server_cached_actions,
        server_unknown_provenance,
    ) = _server_infer_values(warm)
    successful_latency = _optional_values(successful, "episode_time_ms")
    successful_energy = _optional_values(successful, "compute_energy_j")
    average_power = _optional_values(warm, "average_power_w")
    memory = _optional_values(warm, "memory_footprint_mb")

    summary = {
        "schema": "behavior_1k_embodiedperf_summary_v1",
        "sourceTrace": str(source),
        "measurementPolicy": {
            "coldStartDefinition": "first_trace_record_per_eval_process",
            "coldStartEpisodesExcluded": cold_start_episodes,
            "appliesTo": "all_statistics_and_published_episode_references",
            "episodeLatencyScope": "active_rollout_after_reset_until_termination",
            "successfulEpisodeMetrics": "successful_warm_episodes_only",
            "averagePowerScope": "all_warm_attempts",
            "cpuScope": "evaluator_process_only",
            "gpuScope": "explicit_gpu_ids_sampled_by_evaluator_process",
            "serverWrapperActScope": "server-reported policy_wrapper.act duration when available",
            "serverCurrentObservationActScope": (
                "server-reported policy_wrapper.act duration only when validated server provenance declares "
                "that the current observation was consumed"
            ),
        },
        "coverage": {
            "rawEpisodes": len(rows),
            "coldStartEpisodesExcluded": len(excluded),
            "warmEpisodes": len(warm),
            "warmSuccessfulEpisodes": len(successful),
            "availableL0Records": len(l0_values),
            "unavailableL0Records": unavailable_l0,
            "serverWrapperActRecords": len(server_infer_values),
            "serverCurrentObservationActRecords": len(server_current_observation_values),
            "serverCachedActionRecords": server_cached_actions,
            "serverUnknownProvenanceRecords": server_unknown_provenance,
        },
        "excludedRunIds": [row["run_id"] for row in excluded],
        "metrics": {
            "successRate": len(successful) / len(warm),
            "l0ObservationToActionReadyMs": {
                "mean": _mean(l0_values),
                "p95": _percentile(l0_values, 0.95),
            },
            "successfulEpisodeLatencyMs": {
                "count": len(successful_latency),
                "mean": _mean(successful_latency),
                "p95": _percentile(successful_latency, 0.95),
            },
            "successfulEpisodeEnergyJ": {
                "count": len(successful_energy),
                "mean": _mean(successful_energy),
            },
            "averagePowerW": {"count": len(average_power), "mean": _mean(average_power)},
            "peakMemoryMb": max(memory) if memory else None,
            "serverWrapperActMs": {
                "count": len(server_infer_values),
                "mean": _mean(server_infer_values),
                "p95": _percentile(server_infer_values, 0.95),
            },
            "serverCurrentObservationActMs": {
                "count": len(server_current_observation_values),
                "mean": _mean(server_current_observation_values),
                "p95": _percentile(server_current_observation_values, 0.95),
            },
        },
        "warmEpisodes": [
            {
                "runId": row["run_id"],
                "success": row["success"],
                "timeline": row.get("metadata", {}).get("episode_timeline_artifacts_v1"),
                "semanticTimeline": (
                    dict(semantic_view_references[row["run_id"]]) if semantic_view_references is not None else None
                ),
            }
            for row in warm
        ],
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return summary


__all__ = [
    "COLD_START_EPISODES",
    "create_behavior_trace_session",
    "materialize_cold_start_free_semantic_views",
    "summarize_cold_start_free_profile",
]
