"""Serve the BEHAVIOR GR00T N1.7 baseline with model-side stage hooks.

Run this module from the BEHAVIOR-1K checkout while the ET823828/Isaac-GR00T
BEHAVIOR fork is installed in the active environment.

The inference path is adapted from NVIDIA Isaac-GR00T (Apache-2.0).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType
from gr00t.eval.eval_b1k_wrapper import B1KPolicyWrapper, load_modality_config
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype
import numpy as np
import torch
import tyro

from embodiedperf import RemoteStageRecorder
from integrations.embodiedperf.model_servers._hooks import (
    InstrumentedWebsocketPolicyServer,
)


DEFAULT_MODEL_SERVER_PORT = 8000


@dataclass
class ServerConfig:
    """Configuration for the instrumented GR00T N1.7 inference server."""

    model_path: str
    modality_config_path: str
    text_prompt: str
    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    device: str = "cuda"
    control_mode: str = "temporal_ensemble"
    host: str = "127.0.0.1"
    port: int = DEFAULT_MODEL_SERVER_PORT
    strict: bool = True
    embodiedperf_stage_log: str | None = None


class _InstrumentedGr00tPolicy(Gr00tPolicy):
    """Split the upstream GR00T inference pipeline at its real boundaries."""

    def __init__(
        self,
        *args: Any,
        recorder: RemoteStageRecorder,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._embodiedperf_recorder = recorder

    def _get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        with self._embodiedperf_recorder.stage(
            "gr00t_observation_encode",
            kind="observation_encode",
        ):
            unbatched_observations = self._unbatch_observation(observation)
            processed_inputs = []
            states = []
            for obs in unbatched_observations:
                vla_step_data = self._to_vla_step_data(obs)
                states.append(vla_step_data.states)
                messages = [
                    {"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}
                ]
                processed_inputs.append(self.processor(messages))
            collated_inputs = self.collate_fn(processed_inputs)
            collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)

        model_kwargs = dict(collated_inputs)
        if "inputs" not in model_kwargs:
            raise RuntimeError(
                "the installed GR00T collator did not produce the expected 'inputs' field"
            )
        model_inputs = model_kwargs.pop("inputs")
        model_options = model_kwargs.pop("options", None)
        if model_kwargs:
            raise RuntimeError(
                "the installed GR00T collator is incompatible with this pinned adapter; "
                f"unexpected model fields: {sorted(model_kwargs)}"
            )

        with torch.inference_mode():
            with self._embodiedperf_recorder.stage(
                "gr00t_model_input_prepare",
                kind="observation_encode",
                synchronize=True,
            ):
                backbone_inputs, action_inputs = self.model.prepare_input(model_inputs)
            with self._embodiedperf_recorder.stage(
                "gr00t_backbone",
                kind="policy_inference",
                synchronize=True,
            ):
                backbone_outputs = self.model.backbone(backbone_inputs)
            with self._embodiedperf_recorder.stage(
                "gr00t_action_head",
                kind="policy_inference",
                synchronize=True,
            ):
                model_pred = self.model.action_head.get_action(
                    backbone_outputs, action_inputs, model_options
                )

        with self._embodiedperf_recorder.stage(
            "gr00t_action_decode",
            kind="action_decode",
            synchronize=True,
        ):
            normalized_action = model_pred["action_pred"].float()
            batched_states = {
                key: np.stack([state[key] for state in states], axis=0)
                for key in self.modality_configs["state"].modality_keys
            }
            unnormalized_action = self.processor.decode_action(
                normalized_action.cpu().numpy(),
                self.embodiment_tag,
                batched_states,
            )
            casted_action = {
                key: value.astype(np.float32)
                for key, value in unnormalized_action.items()
            }
        return casted_action, {}


class _InstrumentedB1KPolicyWrapper(B1KPolicyWrapper):
    def __init__(
        self,
        *args: Any,
        recorder: RemoteStageRecorder,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._embodiedperf_recorder = recorder

    def process_input(self, obs: dict) -> tuple[dict, int]:
        with self._embodiedperf_recorder.stage(
            "gr00t_b1k_observation_preprocess",
            kind="observation_preprocess",
        ):
            return super().process_input(obs)


def _cuda_synchronize(device: str) -> None:
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)


def main(config: ServerConfig) -> None:
    print("Starting instrumented GR00T inference server...")
    print(f"  Embodiment tag: {config.embodiment_tag}")
    print(f"  Model path: {config.model_path}")
    print(f"  Modality config path: {config.modality_config_path}")
    print(f"  Task prompt: {config.text_prompt}")
    print(f"  Device: {config.device}")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")
    print(f"  EmbodiedPerf stage log: {config.embodiedperf_stage_log or 'disabled'}")

    if config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"model path does not exist: {config.model_path}")
    if not os.path.exists(
        config.modality_config_path
    ) or not config.modality_config_path.endswith(".py"):
        raise FileNotFoundError(
            f"modality config must be an existing Python file: {config.modality_config_path}"
        )

    load_modality_config(config.modality_config_path)
    modality_json = config.modality_config_path.removesuffix(".py") + ".json"
    if not os.path.exists(modality_json):
        raise FileNotFoundError(f"modality config JSON does not exist: {modality_json}")
    with open(modality_json, encoding="utf-8") as stream:
        modality_config = json.load(stream)

    recorder = RemoteStageRecorder(
        source="gr00t_policy_server",
        enabled=config.embodiedperf_stage_log is not None,
        synchronize=lambda: _cuda_synchronize(config.device),
    )
    policy = _InstrumentedGr00tPolicy(
        embodiment_tag=config.embodiment_tag,
        model_path=config.model_path,
        device=config.device,
        strict=config.strict,
        recorder=recorder,
    )
    policy = _InstrumentedB1KPolicyWrapper(
        policy=policy,
        embodiment_tag=config.embodiment_tag,
        modality_config=modality_config,
        text_prompt=config.text_prompt,
        control_mode=config.control_mode,
        recorder=recorder,
    )
    InstrumentedWebsocketPolicyServer(
        policy=policy,
        recorder=recorder,
        host=config.host,
        port=config.port,
        metadata={
            "model": "GR00T-N1.7",
            "text_prompt": config.text_prompt,
        },
        stage_log_path=config.embodiedperf_stage_log,
    ).serve_forever()


if __name__ == "__main__":
    main(tyro.cli(ServerConfig))
