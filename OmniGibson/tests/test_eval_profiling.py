from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import omnigibson.eval.eval as eval_runner
import omnigibson.eval.profiling as behavior_profiling
from omnigibson.eval.evaluator import Evaluator


class _FakePolicy:
    def forward(self, obs):
        return "action"

    def uses_cached_action(self, obs):
        return False


class _CachedPolicy(_FakePolicy):
    def uses_cached_action(self, obs):
        return True


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
        self.events.append(("stage", name, kind))
        yield

    def record_step(self, **kwargs):
        self.events.append(("step", kwargs))


def _evaluator(policy=None):
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.policy = policy or _FakePolicy()
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


def test_evaluator_step_without_profiler_preserves_existing_action_path():
    evaluator = _evaluator()

    terminated, truncated = evaluator.step()

    assert (terminated, truncated) == (True, False)
    assert evaluator.robot_action == "action"
    assert evaluator.obs == {"processed": {"raw": True}}
    assert evaluator.n_success_trials == 1


def test_evaluator_profiles_only_websocket_and_environment_boundaries():
    evaluator = _evaluator()
    profiler = _FakeProfiler()

    terminated, truncated = evaluator.step(profiler=profiler, step_index=0)

    assert (terminated, truncated) == (True, False)
    assert [event for event in profiler.events if event[0] == "stage"] == [
        ("stage", "websocket_policy_round_trip", "communication_wait"),
        ("stage", "omnigibson_environment_step", "environment_step"),
    ]
    assert profiler.events[-1] == (
        "step",
        {
            "terminated": True,
            "truncated": False,
            "counters": {"step_index": 1},
        },
    )


def test_cached_client_action_is_not_mislabeled_as_websocket_wait():
    evaluator = _evaluator(_CachedPolicy())
    profiler = _FakeProfiler()

    evaluator.step(profiler=profiler, step_index=0)

    assert [event for event in profiler.events if event[0] == "stage"] == [
        ("stage", "omnigibson_environment_step", "environment_step")
    ]


@pytest.mark.parametrize("step_index", [None, True, -1, 1.5])
def test_profiled_step_rejects_ambiguous_step_index(step_index):
    with pytest.raises(ValueError, match="non-negative integer"):
        _evaluator().step(profiler=_FakeProfiler(), step_index=step_index)


def test_package_bridge_maps_the_single_sampling_interval(tmp_path, monkeypatch):
    captured = {}
    sentinel = object()

    def create(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(behavior_profiling, "_embodiedperf_api", lambda: (create, None))

    result = behavior_profiling.create_behavior_trace_session(
        output_dir=tmp_path,
        model_key="model",
        checkpoint="checkpoint",
        gpu_ids=[0, 1],
        task_name="turning_on_radio",
        policy_name="websocket",
        host="127.0.0.1",
        port=8000,
        power_interval_s=0.1,
    )

    assert result is sentinel
    assert captured["sample_interval_s"] == 0.1
    assert captured["gpu_ids"] == [0, 1]


def test_main_runs_one_unreported_cold_episode_and_delegates_report(tmp_path, monkeypatch):
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
        embodiedperf_output_dir=str(tmp_path / "profile"),
        embodiedperf_model_key="model",
        embodiedperf_checkpoint="checkpoint",
        embodiedperf_instruction="Turn on the radio.",
        embodiedperf_gpu_ids=[0],
        embodiedperf_power_interval_s=0.05,
        embodiedperf_warmup_instance_index=0,
    )
    monkeypatch.setattr(eval_runner, "parse_args", lambda: args)
    monkeypatch.setattr(eval_runner, "seed_everything", lambda _seed: 7)
    monkeypatch.setattr(
        eval_runner,
        "resolve_instance_ids",
        lambda _task, indices, *, mode: [100 + index for index in indices],
    )

    class FakeProfiler:
        def __init__(self):
            self.trace_path = tmp_path / "profile" / "traces.jsonl"
            self.episodes = []
            self.ends = []

        @contextmanager
        def episode(self, **kwargs):
            self.episodes.append(kwargs)
            yield

        def finish_measurement(self):
            return None

        def end(self, **kwargs):
            self.ends.append(kwargs)

    profiler = FakeProfiler()
    monkeypatch.setattr(behavior_profiling, "create_behavior_trace_session", lambda **_kwargs: profiler)
    finalized = []

    def finalize(trace_path, *, output_dir):
        finalized.append((trace_path, output_dir))
        return {
            "report": Path(output_dir) / "report/index.html",
            "data": {"coverage": {"warmEpisodes": 2}},
        }

    monkeypatch.setattr(behavior_profiling, "finalize_behavior_profile", finalize)

    class FakeEvaluator:
        loaded_instances = []

        def __init__(self, _cfg):
            self.env = SimpleNamespace(task=SimpleNamespace(success=True))
            self.metrics = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def reset(self):
            return None

        def load_task_instance(self, instance_id):
            self.loaded_instances.append(instance_id)

        def step(self, **_kwargs):
            return True, False

    monkeypatch.setattr(eval_runner, "Evaluator", FakeEvaluator)

    eval_runner.main()

    assert FakeEvaluator.loaded_instances == [100, 101, 102]
    assert [episode["metadata"]["profiler_warmup"] for episode in profiler.episodes] == [True, False, False]
    assert len(profiler.ends) == 3
    assert finalized == [(profiler.trace_path, Path(args.embodiedperf_output_dir))]
    assert sorted((tmp_path / "results" / "json").glob("*.json")) == [
        tmp_path / "results" / "json" / "turning_on_radio_101_0.json",
        tmp_path / "results" / "json" / "turning_on_radio_102_0.json",
    ]
