"""Serve the BEHAVIOR π0.5 baseline with optional model-side stage hooks.

Run this module from the BEHAVIOR-1K checkout while the ET823828/openpi
BEHAVIOR fork is installed in the active environment.

The inference path is adapted from OpenPI (Apache-2.0).
"""

from __future__ import annotations

import dataclasses
import logging
import socket
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from openpi.configs.tasks import TASK_REGISTRY
from openpi.models import model as _model
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper
from openpi.training import config as _config
import torch
import tyro

from integrations.embodiedperf.model_servers._hooks import (
    InstrumentedWebsocketPolicyServer,
    StageRecorder,
)


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    config: str
    dir: str


@dataclasses.dataclass
class Args:
    """Arguments for the instrumented π0.5 BEHAVIOR policy server."""

    robot: str
    task: str
    policy: Checkpoint
    repo_id: str | None = None
    control_mode: str = "receding_horizon"
    action_horizon: int = 16
    host: str = "0.0.0.0"
    port: int = 8000
    record: bool = False
    embodiedperf_stage_log: str | None = None


class _InstrumentedPiPolicy:
    """Preserve OpenPI inference while marking its real host-visible phases."""

    def __init__(self, policy: _policy.Policy, recorder: StageRecorder) -> None:
        required = (
            "_input_transform",
            "_output_transform",
            "_sample_actions",
            "_sample_kwargs",
            "_is_pytorch_model",
        )
        missing = [name for name in required if not hasattr(policy, name)]
        if missing:
            raise RuntimeError(
                "the installed OpenPI policy is incompatible with this pinned adapter; "
                f"missing attributes: {missing}"
            )
        self._policy = policy
        self._recorder = recorder

    def __getattr__(self, name: str) -> Any:
        return getattr(self._policy, name)

    def reset(self) -> None:
        self._policy.reset()

    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        with self._recorder.stage(
            "pi05_observation_encode",
            kind="observation_encode",
        ):
            inputs = jax.tree.map(lambda x: x, obs)
            inputs = self._policy._input_transform(inputs)
            if not self._policy._is_pytorch_model:
                inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
                self._policy._rng, sample_rng_or_device = jax.random.split(
                    self._policy._rng
                )
            else:
                inputs = jax.tree.map(
                    lambda x: torch.from_numpy(np.array(x)).to(
                        self._policy._pytorch_device
                    )[None, ...],
                    inputs,
                )
                sample_rng_or_device = self._policy._pytorch_device
            observation = _model.Observation.from_dict(inputs)
            if self._recorder.enabled:
                if self._policy._is_pytorch_model:
                    if torch.cuda.is_available() and str(
                        self._policy._pytorch_device
                    ).startswith("cuda"):
                        torch.cuda.synchronize(self._policy._pytorch_device)
                else:
                    observation = jax.block_until_ready(observation)

        sample_kwargs = dict(self._policy._sample_kwargs)
        if noise is not None:
            noise = (
                torch.from_numpy(noise).to(self._policy._pytorch_device)
                if self._policy._is_pytorch_model
                else jnp.asarray(noise)
            )
            if noise.ndim == 2:
                noise = noise[None, ...]
            sample_kwargs["noise"] = noise

        model_start = time.monotonic()
        with self._recorder.stage(
            "pi05_model_sample_actions",
            kind="policy_inference",
        ):
            actions = self._policy._sample_actions(
                sample_rng_or_device, observation, **sample_kwargs
            )
            if self._recorder.enabled:
                if self._policy._is_pytorch_model:
                    if torch.cuda.is_available() and str(
                        self._policy._pytorch_device
                    ).startswith("cuda"):
                        torch.cuda.synchronize(self._policy._pytorch_device)
                else:
                    actions = jax.block_until_ready(actions)
        model_time = time.monotonic() - model_start

        with self._recorder.stage(
            "pi05_action_decode",
            kind="action_decode",
        ):
            outputs = {"state": inputs["state"], "actions": actions}
            if self._policy._is_pytorch_model:
                outputs = jax.tree.map(
                    lambda x: np.asarray(x[0, ...].detach().cpu()), outputs
                )
            else:
                outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
            outputs = self._policy._output_transform(outputs)
            outputs["policy_timing"] = {"infer_ms": model_time * 1000.0}
        return outputs


class _InstrumentedB1KPolicyWrapper(B1KPolicyWrapper):
    def __init__(self, *args: Any, recorder: StageRecorder, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._embodiedperf_recorder = recorder

    def process_input(self, obs: dict) -> dict:
        with self._embodiedperf_recorder.stage(
            "pi05_b1k_observation_preprocess",
            kind="observation_preprocess",
        ):
            return super().process_input(obs)


def main(args: Args) -> None:
    task_bucket, task_name = args.task.split("/")
    task_prompt = TASK_REGISTRY[task_bucket][task_name]
    logging.info("Using robot: %s, prompt: %s", args.robot, task_prompt)

    config = _config.get_config(args.policy.config)
    norm_stats_repo_id = args.repo_id or args.task
    logging.info("Using norm stats for repo: %s", norm_stats_repo_id)
    config = dataclasses.replace(
        config,
        data=dataclasses.replace(
            config.data,
            repo_id=norm_stats_repo_id,
            robot_config_name=args.robot,
        ),
    )

    policy = _policy_config.create_trained_policy(
        config, args.policy.dir, default_prompt=task_prompt
    )
    policy_metadata = policy.metadata
    recorder = StageRecorder(enabled=args.embodiedperf_stage_log is not None)
    policy = _InstrumentedPiPolicy(policy, recorder)
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    policy = _InstrumentedB1KPolicyWrapper(
        policy=policy,
        robot=args.robot,
        text_prompt=task_prompt,
        control_mode=args.control_mode,
        action_horizon=args.action_horizon,
        max_len=config.model.action_horizon,
        recorder=recorder,
    )

    hostname = socket.gethostname()
    logging.info(
        "Creating server (host: %s, ip: %s)", hostname, socket.gethostbyname(hostname)
    )
    InstrumentedWebsocketPolicyServer(
        policy=policy,
        recorder=recorder,
        host=args.host,
        port=args.port,
        metadata=policy_metadata,
        stage_log_path=args.embodiedperf_stage_log,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
