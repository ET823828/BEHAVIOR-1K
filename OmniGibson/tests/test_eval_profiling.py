from contextlib import contextmanager, nullcontext
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

import omnigibson.eval.eval as eval_runner
from omnigibson.eval.evaluator import Evaluator


class _FakePolicy:
    def __init__(self, *, cached=False):
        self.cached = cached

    def forward(self, obs):
        return "action"

    def uses_cached_action(self, obs):
        return self.cached


class _FakeEnv:
    def step(self, action, n_render_iterations):
        assert action == "action"
        assert n_render_iterations == 1
        return {"raw": True}, 0.0, True, False, {"done": {"success": True}}


class _FakeMetric:
    def step(self, *args):
        return None


class _FakeProfiler:
    def __init__(self):
        self.events = []

    @contextmanager
    def stage(self, name, *, kind):
        self.events.append((name, kind))
        yield


def _evaluator(*, cached=False):
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.policy = _FakePolicy(cached=cached)
    evaluator.env = _FakeEnv()
    evaluator.obs = {"processed": True}
    evaluator.robot_action = None
    evaluator._video_path = None
    evaluator.n_trials = 0
    evaluator.n_success_trials = 0
    evaluator.metrics = [_FakeMetric()]
    evaluator._sync_lights_and_get_obs = lambda obs: obs
    evaluator._preprocess_obs = lambda obs: {"processed": obs}
    return evaluator


def test_evaluator_step_without_profiler_preserves_existing_path():
    evaluator = _evaluator()

    terminated, truncated = evaluator.step()

    assert (terminated, truncated) == (True, False)
    assert evaluator.robot_action == "action"
    assert evaluator.obs == {"processed": {"raw": True}}


@pytest.mark.parametrize(
    ("cached", "expected"),
    [
        (
            False,
            [
                ("websocket_policy_round_trip", "communication_wait"),
                ("omnigibson_environment_step", "environment_step"),
            ],
        ),
        (True, [("omnigibson_environment_step", "environment_step")]),
    ],
)
def test_evaluator_profiles_only_real_websocket_calls_and_environment_steps(cached, expected):
    evaluator = _evaluator(cached=cached)
    profiler = _FakeProfiler()

    evaluator.step(profiler=profiler)

    assert profiler.events == expected


def test_main_profiles_normal_rollouts_and_finalizes_before_shutdown(tmp_path, monkeypatch):
    args = SimpleNamespace(
        task_name="turning_on_radio",
        host="127.0.0.1",
        port=8000,
        robot_config=None,
        instance_indices=[1, 2],
        mode="public_test",
        num_rollouts=1,
        max_steps=1,
        env_wrapper="fake.Wrapper",
        policy="websocket",
        output_dir=str(tmp_path / "results"),
        write_video=False,
        video_fps=30,
        headless=True,
        embodiedperf=True,
        embodiedperf_model_key="model",
        embodiedperf_checkpoint="checkpoint",
        embodiedperf_instruction="Turn on the radio.",
        embodiedperf_gpu_ids=[0],
    )
    monkeypatch.setattr(eval_runner, "parse_args", lambda: args)
    monkeypatch.setattr(eval_runner, "seed_everything", lambda _seed: 7)
    monkeypatch.setattr(eval_runner, "resolve_instance_ids", lambda _task, indices, *, mode: indices)

    profiler = MagicMock()
    profiler.trace_path = tmp_path / "results/embodiedperf/traces.jsonl"
    profiler.episode.side_effect = lambda **_kwargs: nullcontext()
    created = []
    finalized = []
    evaluator_closed = False

    api = ModuleType("embodiedperf.benchmarks.behavior1k")

    def create(**kwargs):
        created.append(kwargs)
        return profiler

    def finalize(trace_path, *, output_dir):
        assert not evaluator_closed
        finalized.append((trace_path, output_dir))
        return {
            "report": Path(output_dir) / "report/index.html",
            "data": {"coverage": {"warmEpisodes": 1}},
        }

    api.create_behavior_trace_session = create
    api.finalize_behavior_profile = finalize
    package = ModuleType("embodiedperf")
    package.__path__ = []
    benchmarks = ModuleType("embodiedperf.benchmarks")
    benchmarks.__path__ = []
    monkeypatch.setitem(sys.modules, "embodiedperf", package)
    monkeypatch.setitem(sys.modules, "embodiedperf.benchmarks", benchmarks)
    monkeypatch.setitem(sys.modules, "embodiedperf.benchmarks.behavior1k", api)

    evaluator = MagicMock()
    evaluator.__enter__.return_value = evaluator

    def close_evaluator(*_args):
        nonlocal evaluator_closed
        evaluator_closed = True

    evaluator.__exit__.side_effect = close_evaluator
    evaluator.env.task.success = True
    evaluator.metrics = []
    evaluator.step.return_value = (True, False)
    monkeypatch.setattr(eval_runner, "Evaluator", lambda _cfg: evaluator)

    eval_runner.main()

    profile_dir = Path(args.output_dir) / "embodiedperf"
    assert created[0]["output_dir"] == profile_dir
    assert created[0]["sample_interval_s"] == 0.05
    assert profiler.episode.call_count == 2
    assert profiler.finish_measurement.call_count == 2
    assert profiler.end.call_count == 2
    assert finalized == [(profiler.trace_path, profile_dir)]
    assert evaluator_closed
    assert sorted((tmp_path / "results/json").glob("*.json")) == [
        tmp_path / "results/json/turning_on_radio_1_0.json",
        tmp_path / "results/json/turning_on_radio_2_0.json",
    ]
