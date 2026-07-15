from contextlib import contextmanager
import json

import pytest

import omnigibson.eval.profiling as behavior_profiling
from omnigibson.eval.evaluator import Evaluator
from omnigibson.eval.profiling import (
    create_behavior_trace_session,
    materialize_cold_start_free_semantic_views,
    summarize_cold_start_free_profile,
)
from omnigibson.eval.utils.network_utils import _normalize_server_timing


def _trace(run_id, *, success, episode_ms, energy_j, l0_ms, infer_ms):
    return {
        "run_id": run_id,
        "success": success,
        "episode_time_ms": episode_ms,
        "compute_energy_j": energy_j,
        "average_power_w": 100.0,
        "memory_footprint_mb": 2000.0,
        "metadata": {
            "action_readiness_v1": {
                "records": [
                    {
                        "status": "available",
                        "observation_to_action_ready_ms": l0_ms,
                    }
                ]
            },
            "step_records": [{"metadata": {"server_timing": {"infer_ms": infer_ms}}}],
            "episode_timeline_artifacts_v1": {"schema": "episode_timeline_artifacts_v1"},
        },
    }


def test_summary_excludes_first_episode_from_every_statistic(tmp_path):
    trace_path = tmp_path / "traces.jsonl"
    rows = [
        _trace("cold", success=False, episode_ms=9999.0, energy_j=999.0, l0_ms=999.0, infer_ms=888.0),
        _trace("warm", success=True, episode_ms=1200.0, energy_j=12.0, l0_ms=30.0, infer_ms=20.0),
    ]
    trace_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    summary = summarize_cold_start_free_profile(trace_path, output_path=tmp_path / "summary.json")

    assert summary["excludedRunIds"] == ["cold"]
    assert summary["coverage"]["warmEpisodes"] == 1
    assert summary["metrics"]["successRate"] == 1.0
    assert summary["metrics"]["l0ObservationToActionReadyMs"]["mean"] == 30.0
    assert summary["metrics"]["successfulEpisodeLatencyMs"]["mean"] == 1200.0
    assert summary["metrics"]["successfulEpisodeEnergyJ"]["mean"] == 12.0
    assert summary["metrics"]["serverWrapperActMs"]["mean"] == 20.0


def test_summary_fails_when_no_warm_episode_remains(tmp_path):
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_text(
        json.dumps(_trace("cold", success=False, episode_ms=1.0, energy_j=1.0, l0_ms=1.0, infer_ms=1.0)) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no warm episode remains"):
        summarize_cold_start_free_profile(trace_path, output_path=tmp_path / "summary.json")


def test_summary_uses_successful_warm_episodes_for_latency_and_energy_only(tmp_path):
    trace_path = tmp_path / "traces.jsonl"
    rows = [
        _trace("cold", success=True, episode_ms=1.0, energy_j=1.0, l0_ms=1.0, infer_ms=1.0),
        _trace("warm-fail", success=False, episode_ms=9000.0, energy_j=900.0, l0_ms=90.0, infer_ms=80.0),
        _trace("warm-success", success=True, episode_ms=1200.0, energy_j=12.0, l0_ms=30.0, infer_ms=20.0),
    ]
    trace_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    summary = summarize_cold_start_free_profile(trace_path, output_path=tmp_path / "summary.json")

    assert summary["metrics"]["successRate"] == 0.5
    assert summary["metrics"]["successfulEpisodeLatencyMs"] == {
        "count": 1,
        "mean": 1200.0,
        "p95": 1200.0,
    }
    assert summary["metrics"]["successfulEpisodeEnergyJ"] == {"count": 1, "mean": 12.0}
    assert summary["metrics"]["l0ObservationToActionReadyMs"]["mean"] == 60.0


def test_summary_rejects_negative_metric_values(tmp_path):
    trace_path = tmp_path / "traces.jsonl"
    rows = [
        _trace("cold", success=False, episode_ms=1.0, energy_j=1.0, l0_ms=1.0, infer_ms=1.0),
        _trace("warm", success=True, episode_ms=-1.0, energy_j=1.0, l0_ms=1.0, infer_ms=1.0),
    ]
    trace_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="must be non-negative"):
        summarize_cold_start_free_profile(trace_path, output_path=tmp_path / "summary.json")


def test_semantic_views_are_materialized_for_warm_traces_only(tmp_path, monkeypatch):
    trace_path = tmp_path / "traces.jsonl"
    cold = _trace("cold", success=False, episode_ms=1.0, energy_j=1.0, l0_ms=1.0, infer_ms=1.0)
    warm = _trace("warm", success=True, episode_ms=2.0, energy_j=2.0, l0_ms=2.0, infer_ms=2.0)
    warm["metadata"]["semantic_stage_v1"] = {"run_id": "warm"}
    trace_path.write_text(json.dumps(cold) + "\n" + json.dumps(warm) + "\n", encoding="utf-8")

    def derive(collection, **expected):
        assert collection["run_id"] == expected["expected_run_id"]
        return {"derived": True}

    def render(collection, **expected):
        derive(collection, **expected)
        return "<!doctype html><html></html>\n"

    def build(collection, **expected):
        derive(collection, **expected)
        return {"traceEvents": []}

    def serialize(perfetto):
        return json.dumps(perfetto, sort_keys=True)

    def validate(collection, html, perfetto, **expected):
        derive(collection, **expected)
        assert html.endswith("</html>\n")
        assert perfetto == {"traceEvents": []}
        return {"status": "pass"}

    monkeypatch.setattr(
        behavior_profiling,
        "_semantic_view_functions",
        lambda: (derive, render, build, serialize, validate),
    )

    output_dir = tmp_path / "semantic_timeline"
    references = materialize_cold_start_free_semantic_views(trace_path, output_dir=output_dir)

    assert set(references) == {"warm"}
    assert not (output_dir / "episode_000001").exists()
    assert (output_dir / "episode_000002" / "semantic_stage.html").is_file()
    assert references["warm"]["html"] == "semantic_timeline/episode_000002/semantic_stage.html"


@pytest.mark.parametrize("value", [None, {}, {"infer_ms": -1}, {"infer_ms": float("nan")}, {"infer_ms": True}])
def test_server_timing_rejects_missing_or_malformed_values(value):
    assert _normalize_server_timing(value) is None


def test_server_timing_accepts_known_finite_durations():
    assert _normalize_server_timing({"infer_ms": 12, "prev_total_ms": 15.5, "ignored": 1}) == {
        "infer_ms": 12.0,
        "prev_total_ms": 15.5,
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"gpu_ids": [0, 0]}, "must not contain duplicates"),
        ({"power_interval_s": True}, "finite positive number"),
        ({"port": True}, "between 1 and 65535"),
    ],
)
def test_trace_session_factory_rejects_ambiguous_resource_scope(tmp_path, overrides, message):
    kwargs = {
        "output_dir": tmp_path / "profile",
        "model_key": "model",
        "checkpoint": "checkpoint",
        "gpu_ids": [0],
        "task_name": "turning_on_radio",
        "policy_name": "websocket",
        "host": "127.0.0.1",
        "port": 8000,
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        create_behavior_trace_session(**kwargs)


class _FakePolicy:
    def __init__(self):
        self.last_server_timing = {"infer_ms": 4.0}

    def forward(self, obs):
        return "action"


class _CachedPolicy(_FakePolicy):
    def __init__(self):
        self.last_action = "action"
        self.last_server_timing = None

    def uses_cached_action(self, obs):
        return not obs["need_new_action"] and self.last_action is not None


class _FakeEnv:
    def __init__(self):
        self.env = self
        self.env_config = {"action_frequency": 20}

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

    def record_observation_available(self, **kwargs):
        self.events.append("observation")
        return kwargs["observation_id"]

    @contextmanager
    def stage(self, name, *, kind):
        self.events.append(("stage", name, kind))
        yield

    def act(self, fn, **kwargs):
        self.events.append(("act", kwargs))
        return fn()

    def record_action_delivery(self, **kwargs):
        self.events.append(("delivery", kwargs))

    def finish_action_execution(self, **kwargs):
        self.events.append(("finish", kwargs))

    def record_step(self, **kwargs):
        self.events.append(("step", kwargs))


def test_evaluator_step_without_profiler_preserves_existing_action_path():
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.policy = _FakePolicy()
    evaluator.env = _FakeEnv()
    evaluator.obs = {"processed": True}
    evaluator.robot_action = None
    evaluator._video_path = None
    evaluator._profile_source_observation_id = None
    evaluator.n_trials = 0
    evaluator.n_success_trials = 0
    evaluator.metrics = [_FakeMetric()]
    evaluator._sync_lights_and_get_obs = lambda obs: obs
    evaluator._preprocess_obs = lambda obs: {"processed": obs}

    terminated, truncated = evaluator.step()

    assert (terminated, truncated) == (True, False)
    assert evaluator.robot_action == "action"
    assert evaluator.obs == {"processed": {"raw": True}}
    assert evaluator._profile_source_observation_id is None


def test_evaluator_step_profiles_real_boundaries_without_changing_action():
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.policy = _FakePolicy()
    evaluator.env = _FakeEnv()
    evaluator.obs = {"processed": True}
    evaluator.robot_action = None
    evaluator._video_path = None
    evaluator._profile_source_observation_id = None
    evaluator.n_trials = 0
    evaluator.n_success_trials = 0
    evaluator.metrics = [_FakeMetric()]
    evaluator._sync_lights_and_get_obs = lambda obs: obs
    evaluator._preprocess_obs = lambda obs: {"processed": obs}
    profiler = _FakeProfiler()

    terminated, truncated = evaluator.step(profiler=profiler, step_index=0)

    assert (terminated, truncated) == (True, False)
    assert evaluator.robot_action == "action"
    assert evaluator.n_success_trials == 1
    delivery = next(event for event in profiler.events if event[0] == "delivery")
    assert delivery[1]["control_period_ms"] == 50.0
    step = next(event for event in profiler.events if event[0] == "step")
    assert step[1]["metadata"]["server_timing"] == {"infer_ms": 4.0}
    assert profiler.events.count("observation") == 1


def test_evaluator_step_labels_client_cache_without_claiming_websocket_wait():
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.policy = _CachedPolicy()
    evaluator.env = _FakeEnv()
    evaluator.obs = {"need_new_action": False}
    evaluator.robot_action = None
    evaluator._video_path = None
    evaluator._profile_source_observation_id = None
    evaluator.n_trials = 0
    evaluator.n_success_trials = 0
    evaluator.metrics = [_FakeMetric()]
    evaluator._sync_lights_and_get_obs = lambda obs: obs
    evaluator._preprocess_obs = lambda obs: obs
    profiler = _FakeProfiler()

    evaluator.step(profiler=profiler, step_index=0)

    stage = next(event for event in profiler.events if event[0] == "stage")
    assert stage == ("stage", "client_cached_policy_action", "cache")
    act = next(event for event in profiler.events if event[0] == "act")
    assert act[1]["metadata"]["boundary"] == "client_cached_policy_action"
    step = next(event for event in profiler.events if event[0] == "step")
    assert step[1]["metadata"] == {"server_timing_status": "unavailable"}


class _TwoStepEnv(_FakeEnv):
    def __init__(self):
        super().__init__()
        self.step_count = 0

    def step(self, action, n_render_iterations):
        self.step_count += 1
        terminated = self.step_count == 2
        return {"raw": self.step_count}, 0.0, terminated, False, {"done": {"success": terminated}}


def test_evaluator_step_carries_each_processed_observation_to_exactly_one_action():
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.policy = _FakePolicy()
    evaluator.env = _TwoStepEnv()
    evaluator.obs = {"processed": "reset"}
    evaluator.robot_action = None
    evaluator._video_path = None
    evaluator._profile_source_observation_id = None
    evaluator.n_trials = 0
    evaluator.n_success_trials = 0
    evaluator.metrics = [_FakeMetric()]
    evaluator._sync_lights_and_get_obs = lambda obs: obs
    evaluator._preprocess_obs = lambda obs: {"processed": obs}
    profiler = _FakeProfiler()

    assert evaluator.step(profiler=profiler, step_index=0) == (False, False)
    assert evaluator.step(profiler=profiler, step_index=1) == (True, False)

    acts = [event[1] for event in profiler.events if event[0] == "act"]
    assert [event["source_observation_id"] for event in acts] == ["observation-0", "observation-1"]
    assert [event["output_id"] for event in acts] == ["action-0", "action-1"]
    assert profiler.events.count("observation") == 2
