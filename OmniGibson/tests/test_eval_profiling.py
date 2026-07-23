from contextlib import contextmanager
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


class _FakeEpisode:
    def __init__(self):
        self.events = []
        self.finished = []

    @contextmanager
    def stage(self, name, *, kind):
        self.events.append((name, kind))
        yield

    def finish(self, *, success, metrics=None):
        self.finished.append((success, metrics))


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


def test_evaluator_step_without_episode_preserves_existing_path():
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
    episode = _FakeEpisode()

    evaluator.step(episode=episode)

    assert episode.events == expected


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
        embodiedperf_model="model",
        embodiedperf_checkpoint="checkpoint",
        embodiedperf_instruction="Turn on the radio.",
        embodiedperf_gpu_ids=[0],
        embodiedperf_sample_interval=0.05,
    )
    monkeypatch.setattr(eval_runner, "parse_args", lambda: args)
    monkeypatch.setattr(eval_runner, "seed_everything", lambda _seed: 7)
    monkeypatch.setattr(eval_runner, "resolve_instance_ids", lambda _task, indices, *, mode: indices)

    episodes = []
    created = []
    finalized = []
    evaluator_closed = False

    profiler = MagicMock()

    @contextmanager
    def open_episode(name, *, metadata):
        episode = _FakeEpisode()
        episodes.append((name, metadata, episode))
        yield episode

    profiler.episode.side_effect = open_episode

    def finalize():
        assert not evaluator_closed
        finalized.append(True)
        return {
            "report": tmp_path / "results/embodiedperf/report/index.html",
            "data": {"coverage": {"warm_episodes": 1}},
        }

    profiler.finalize.side_effect = finalize

    api = ModuleType("embodiedperf.integrations.behavior1k")

    def profiler_from_args(namespace, **kwargs):
        assert namespace.embodiedperf_model == "model"
        created.append(kwargs)
        return profiler

    api.profiler_from_args = profiler_from_args
    package = ModuleType("embodiedperf")
    package.__path__ = []
    integrations = ModuleType("embodiedperf.integrations")
    integrations.__path__ = []
    monkeypatch.setitem(sys.modules, "embodiedperf", package)
    monkeypatch.setitem(sys.modules, "embodiedperf.integrations", integrations)
    monkeypatch.setitem(sys.modules, "embodiedperf.integrations.behavior1k", api)

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

    assert created == [
        {
            "output_dir": str(tmp_path / "results"),
            "task_name": "turning_on_radio",
            "policy_name": "websocket",
            "host": "127.0.0.1",
            "port": 8000,
        }
    ]
    assert [name for name, _metadata, _episode in episodes] == [
        "turning_on_radio/instance-1",
        "turning_on_radio/instance-2",
    ]
    assert [metadata["instance_id"] for _name, metadata, _episode in episodes] == [1, 2]
    for _name, _metadata, episode in episodes:
        assert episode.finished == [(True, {"steps": 1})]
    for call in evaluator.step.call_args_list:
        assert isinstance(call.kwargs["episode"], _FakeEpisode)
    assert finalized == [True]
    assert evaluator_closed
    assert sorted((tmp_path / "results/json").glob("*.json")) == [
        tmp_path / "results/json/turning_on_radio_1_0.json",
        tmp_path / "results/json/turning_on_radio_2_0.json",
    ]


def test_main_without_embodiedperf_never_touches_profiler(tmp_path, monkeypatch):
    args = SimpleNamespace(
        task_name="turning_on_radio",
        host="127.0.0.1",
        port=8000,
        robot_config=None,
        instance_indices=[1],
        mode="public_test",
        num_rollouts=1,
        max_steps=1,
        env_wrapper="fake.Wrapper",
        policy="websocket",
        output_dir=str(tmp_path / "results"),
        write_video=False,
        video_fps=30,
        headless=True,
        embodiedperf=False,
        embodiedperf_model=None,
        embodiedperf_checkpoint=None,
        embodiedperf_instruction=None,
        embodiedperf_gpu_ids=None,
        embodiedperf_sample_interval=0.05,
    )
    monkeypatch.setattr(eval_runner, "parse_args", lambda: args)
    monkeypatch.setattr(eval_runner, "seed_everything", lambda _seed: 7)
    monkeypatch.setattr(eval_runner, "resolve_instance_ids", lambda _task, indices, *, mode: indices)
    monkeypatch.setitem(sys.modules, "embodiedperf", None)

    evaluator = MagicMock()
    evaluator.__enter__.return_value = evaluator
    evaluator.env.task.success = True
    evaluator.metrics = []
    evaluator.step.return_value = (True, False)
    monkeypatch.setattr(eval_runner, "Evaluator", lambda _cfg: evaluator)

    eval_runner.main()

    assert evaluator.step.call_args_list[0].kwargs["episode"] is None
    assert (tmp_path / "results/json/turning_on_radio_1_0.json").is_file()
