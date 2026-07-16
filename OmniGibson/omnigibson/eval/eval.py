"""Websocket evaluation runner for the BEHAVIOR-1K challenge.

Drives the OmniGibson ``Evaluator`` against a policy served over a websocket
(e.g. the openpi or GR00T ``scripts/b1k/serve_b1k.py`` server). For each test instance of a
task it runs a rollout and writes a per-rollout result JSON compatible with
``omnigibson/eval/utils/score_utils.py`` (``q_score``, ``time``,
``agent_distance`` / ``normalized_agent_distance``).

Example:
    python -m omnigibson.eval.eval \
        --task-name turning_on_radio \
        --robot-config omnigibson/eval/r1pro.yaml \
        --mode public_test \
        --host 127.0.0.1 --port 8000 \
        --instance-indices 0 --max-steps 500 \
        --output-dir outputs/b1k_eval --write-video
"""

import argparse
from contextlib import nullcontext
import json
import logging
import math
import os
from pathlib import Path

from omegaconf import OmegaConf

from omnigibson.eval.evaluator import Evaluator, resolve_instance_ids
from omnigibson.eval.utils.eval_utils import DEFAULT_EVAL_SEED, seed_everything
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import create_module_logger


logger = create_module_logger(module_name=__name__)
logger.setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", required=True, help="BEHAVIOR task name, e.g. turning_on_radio.")
    parser.add_argument("--host", default="127.0.0.1", help="Policy websocket server host.")
    parser.add_argument("--port", type=int, default=8000, help="Policy websocket server port.")
    parser.add_argument(
        "--robot-config",
        type=str,
        default=None,
        help=(
            "Optional path to YAML/JSON file containing one complete robot config dictionary with canonical "
            "'model' and 'name' fields. Add eval.camera_sensor_names to configure eval camera roles."
        ),
    )
    parser.add_argument(
        "--instance-indices",
        type=int,
        nargs="+",
        default=[0],
        help=(
            "Instance indices for the selected mode. For train these are direct train instance IDs; "
            "for public_test / hidden_test these index into that 20-instance split."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("train", "public_test", "hidden_test"),
        default="public_test",
        help="Instance split to evaluate. Default: public_test.",
    )
    parser.add_argument("--num-rollouts", type=int, default=1, help="Rollouts per instance.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Episode timeout in steps. Default (None) = 1.5x mean human-demo length.",
    )
    parser.add_argument(
        "--env-wrapper",
        default="omnigibson.eval.wrappers.DefaultWrapper",
        help="Target path of the EnvironmentWrapper to apply.",
    )
    parser.add_argument(
        "--policy",
        choices=("websocket", "local"),
        default="websocket",
        help="Policy backend to use. local emits zero actions and is intended for eval smoke tests.",
    )
    parser.add_argument("--output-dir", default="/tmp/b1k_eval", help="Where to write result JSONs.")
    parser.add_argument(
        "--write-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save an MP4 rollout video (head + wrist cameras) per rollout under <output-dir>/videos.",
    )
    parser.add_argument("--video-fps", type=int, default=30, help="Frame rate for saved rollout videos.")
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run OmniGibson headless (default: True).",
    )
    parser.add_argument(
        "--embodiedperf",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable optional EmbodiedPerf system profiling (default: False).",
    )
    parser.add_argument(
        "--embodiedperf-output-dir",
        default=None,
        help="Profiler artifact directory. Default: <output-dir>/embodiedperf.",
    )
    parser.add_argument(
        "--embodiedperf-model-key",
        default=None,
        help="Stable model identifier recorded in profiler artifacts (required with --embodiedperf).",
    )
    parser.add_argument(
        "--embodiedperf-checkpoint",
        default=None,
        help="Checkpoint path or immutable identifier recorded as provenance (required with --embodiedperf).",
    )
    parser.add_argument(
        "--embodiedperf-instruction",
        default=None,
        help=(
            "Exact language instruction configured on the policy server and recorded in every profile trace "
            "for provenance (required with --embodiedperf)."
        ),
    )
    parser.add_argument(
        "--embodiedperf-gpu-ids",
        type=int,
        nargs="+",
        default=None,
        help="Physical GPU ids included in local power/memory accounting (required with --embodiedperf).",
    )
    parser.add_argument(
        "--embodiedperf-power-interval-s",
        type=float,
        default=0.05,
        help="NVML sampling interval in seconds (default: 0.05).",
    )
    parser.add_argument(
        "--embodiedperf-warmup-instance-index",
        type=int,
        default=None,
        help=(
            "Optional split-relative instance index run once as the retained cold-start trace before all reported "
            "rollouts. It is excluded from result JSONs and every profiler aggregate."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    gm.HEADLESS = args.headless

    seed = seed_everything(DEFAULT_EVAL_SEED)
    logger.info(f"Seeded Python, NumPy, and Torch with seed={seed}")

    instance_ids = resolve_instance_ids(args.task_name, args.instance_indices, mode=args.mode)
    logger.info(f"Resolved {args.mode} instance ids for {args.task_name}: {instance_ids}")

    if args.embodiedperf_warmup_instance_index is not None and not args.embodiedperf:
        raise ValueError("--embodiedperf-warmup-instance-index requires --embodiedperf")
    warmup_instance_id = None
    if args.embodiedperf_warmup_instance_index is not None:
        warmup_instance_id = resolve_instance_ids(
            args.task_name,
            [args.embodiedperf_warmup_instance_index],
            mode=args.mode,
        )[0]

    profiler = None
    profile_dir = None
    if args.embodiedperf:
        if args.num_rollouts < 1:
            raise ValueError("--num-rollouts must be at least 1 when profiling is enabled")
        if args.policy != "websocket":
            raise ValueError("--embodiedperf currently requires --policy websocket")
        if not args.embodiedperf_model_key:
            raise ValueError("--embodiedperf-model-key is required with --embodiedperf")
        if not args.embodiedperf_checkpoint:
            raise ValueError("--embodiedperf-checkpoint is required with --embodiedperf")
        if not isinstance(args.embodiedperf_instruction, str) or not args.embodiedperf_instruction.strip():
            raise ValueError("--embodiedperf-instruction is required with --embodiedperf")
        if not args.embodiedperf_gpu_ids:
            raise ValueError("--embodiedperf-gpu-ids is required with --embodiedperf")
        raw_episode_count = len(instance_ids) * args.num_rollouts
        if args.embodiedperf_warmup_instance_index is not None:
            raw_episode_count += 1
        if raw_episode_count <= 1:
            raise ValueError(
                "EmbodiedPerf excludes the first episode as cold start; select at least two total rollouts"
            )
        from omnigibson.eval.profiling import create_behavior_trace_session

        profile_dir = Path(args.embodiedperf_output_dir or Path(args.output_dir) / "embodiedperf")
        profiler = create_behavior_trace_session(
            output_dir=profile_dir,
            model_key=args.embodiedperf_model_key,
            checkpoint=args.embodiedperf_checkpoint,
            gpu_ids=args.embodiedperf_gpu_ids,
            task_name=args.task_name,
            policy_name=args.policy,
            host=args.host,
            port=args.port,
            power_interval_s=args.embodiedperf_power_interval_s,
        )

    robot_config = None
    if args.robot_config is not None:
        robot_config_path = Path(args.robot_config).expanduser()
        robot_config = OmegaConf.load(str(robot_config_path))
        logger.info(f"Loaded robot config from {robot_config_path}")

    if args.policy == "websocket":
        model_cfg = {
            "_target_": "omnigibson.eval.policies.WebsocketPolicy",
            "host": args.host,
            "port": args.port,
        }
    else:
        model_cfg = {"_target_": "omnigibson.eval.policies.LocalPolicy", "action_dim": None}

    cfg = OmegaConf.create(
        {
            "env_wrapper": {"_target_": args.env_wrapper},
            "policy_name": args.policy,
            "model": model_cfg,
            "headless": args.headless,
            "partial_scene_load": True,
            "max_steps": args.max_steps,
            "write_video": args.write_video,
            "mode": args.mode,
            "seed": seed,
            "task": {"name": args.task_name},
            "robot": robot_config,
        }
    )

    json_dir = os.path.join(os.path.expanduser(args.output_dir), "json")
    os.makedirs(json_dir, exist_ok=True)
    video_dir = os.path.join(os.path.expanduser(args.output_dir), "videos")
    if args.write_video:
        os.makedirs(video_dir, exist_ok=True)

    evaluated_instances = [(instance_id, False) for instance_id in instance_ids]
    if profiler is not None and warmup_instance_id is not None:
        evaluated_instances.insert(0, (warmup_instance_id, True))
        logger.info(
            "EmbodiedPerf cold-start warmup: %s index %s -> instance %s",
            args.mode,
            args.embodiedperf_warmup_instance_index,
            warmup_instance_id,
        )

    results = []
    with Evaluator(cfg) as evaluator:
        for instance_id, is_profiler_warmup in evaluated_instances:
            try:
                evaluator.reset()
                evaluator.load_task_instance(int(instance_id))
            except Exception:
                logger.exception(f"Failed to load task instance {instance_id}.")
                raise
            rollout_ids = (-1,) if is_profiler_warmup else range(args.num_rollouts)
            for rollout_id in rollout_ids:
                video_path = os.path.join(video_dir, f"{args.task_name}_{instance_id}_{rollout_id}.mp4")
                record_video = args.write_video and not is_profiler_warmup
                try:
                    evaluator.reset()
                    if record_video:
                        evaluator.start_recording(video_path, rate=args.video_fps)
                    episode_context = (
                        profiler.episode(
                            task=f"{args.task_name}/instance-{instance_id}",
                            seed=seed,
                            instruction=args.embodiedperf_instruction,
                            init_id=int(instance_id),
                            task_config=f"{args.mode}/instance-{instance_id}",
                            metadata={
                                "instance_id": int(instance_id),
                                "rollout_id": rollout_id,
                                "mode": args.mode,
                                "profiler_warmup": is_profiler_warmup,
                            },
                        )
                        if profiler is not None
                        else nullcontext()
                    )
                    with episode_context:
                        terminated = truncated = False
                        steps = 0
                        while not (terminated or truncated):
                            terminated, truncated = evaluator.step(profiler=profiler, step_index=steps)
                            steps += 1

                        if profiler is not None:
                            profiler.finish_measurement()
                        success = bool(evaluator.env.task.success)
                        metrics = {}
                        for metric in evaluator.metrics:
                            metrics.update(metric.aggregate(evaluator.env))
                        if profiler is not None:
                            q_score_final = metrics.get("q_score", {}).get("final")
                            if q_score_final is not None:
                                q_score_final = float(q_score_final)
                                if not math.isfinite(q_score_final):
                                    raise ValueError("q_score.final must be finite when profiling is enabled")
                            profiler.end(
                                success=success,
                                info={
                                    "instance_id": int(instance_id),
                                    "rollout_id": rollout_id,
                                    "profiler_warmup": is_profiler_warmup,
                                    "steps": steps,
                                    "terminated": bool(terminated),
                                    "truncated": bool(truncated),
                                    "q_score_final": q_score_final,
                                },
                                failure_mode=None if success else "challenge_episode_unsuccessful",
                            )

                    if is_profiler_warmup:
                        logger.info(
                            "EmbodiedPerf cold-start warmup complete: instance=%s steps=%s success=%s",
                            instance_id,
                            steps,
                            success,
                        )
                        continue

                    result = {
                        "task": args.task_name,
                        "instance_id": int(instance_id),
                        "rollout_id": rollout_id,
                        "steps": steps,
                        "success": success,
                        **metrics,
                    }
                    out_path = os.path.join(json_dir, f"{args.task_name}_{instance_id}_{rollout_id}.json")
                    with open(out_path, "w") as f:
                        json.dump(result, f, indent=2, default=float)
                    q_score = metrics.get("q_score", {}).get("final")
                    video_msg = f" | video -> {video_path}" if record_video else ""
                    logger.info(
                        f"Result: instance={instance_id} rollout={rollout_id} steps={steps} "
                        f"success={success} q_score={q_score} -> {out_path}{video_msg}"
                    )
                    results.append(result)
                except Exception:
                    logger.exception(f"Instance {instance_id} rollout {rollout_id} failed.")
                    raise
                finally:
                    if record_video:
                        evaluator.stop_recording()

        # OmniGibson's global shutdown terminates the process from Evaluator.__exit__, so all
        # post-rollout artifacts must be materialized before leaving this context. Measurements
        # are already closed here, therefore rendering the views cannot pollute episode metrics.
        n = len(results)
        n_success = sum(r["success"] for r in results)
        mean_q = (sum(r.get("q_score", {}).get("final", 0.0) for r in results) / n) if n else 0.0
        logger.info(f"Eval summary: {n_success}/{n} success | mean q_score={mean_q:.3f} | task={args.task_name}")
        if profiler is not None:
            from omnigibson.eval.profiling import (
                materialize_cold_start_free_semantic_views,
                summarize_cold_start_free_profile,
            )

            assert profile_dir is not None
            semantic_views = materialize_cold_start_free_semantic_views(
                profiler.trace_path,
                output_dir=profile_dir / "semantic_timeline",
            )
            summary_path = profile_dir / "summary.json"
            summary = summarize_cold_start_free_profile(
                profiler.trace_path,
                output_path=summary_path,
                semantic_view_references=semantic_views,
            )
            logger.info(
                "EmbodiedPerf summary: %s warm episode(s), first episode excluded -> %s",
                summary["coverage"]["warmEpisodes"],
                summary_path,
            )


if __name__ == "__main__":
    main()
