"""Optional, dependency-light bridge to the installed EmbodiedPerf package."""

from pathlib import Path
from typing import Any


def _embodiedperf_api():
    try:
        from embodiedperf.benchmarks.behavior1k import (
            create_behavior_trace_session as create_session,
            finalize_behavior_profile as finalize_profile,
        )
    except ImportError as exc:
        raise RuntimeError(
            "EmbodiedPerf profiling was requested, but embodiedperf[behavior1k] is not installed "
            "in the evaluator environment"
        ) from exc
    return create_session, finalize_profile


def create_behavior_trace_session(
    *,
    output_dir: str | Path,
    model_key: str,
    checkpoint: str,
    gpu_ids: list[int],
    task_name: str,
    policy_name: str,
    host: str,
    port: int,
    power_interval_s: float = 0.05,
) -> Any:
    """Create the package-owned session only when profiling is explicitly enabled."""

    create_session, _ = _embodiedperf_api()
    return create_session(
        output_dir=output_dir,
        model_key=model_key,
        checkpoint=checkpoint,
        gpu_ids=gpu_ids,
        task_name=task_name,
        policy_name=policy_name,
        host=host,
        port=port,
        sample_interval_s=power_interval_s,
    )


def finalize_behavior_profile(trace_path: str | Path, *, output_dir: str | Path) -> dict[str, Any]:
    """Delegate validation, aggregation, timeline, and HTML rendering to EmbodiedPerf."""

    _, finalize_profile = _embodiedperf_api()
    return finalize_profile(trace_path, output_dir=output_dir)
