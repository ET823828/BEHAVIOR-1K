from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
import websockets

from integrations.embodiedperf.model_servers._hooks import (
    InstrumentedWebsocketPolicyServer,
    SERVER_REQUEST_SCHEMA,
    SERVER_STAGES_SCHEMA,
    RequestStageLog,
    StageRecorder,
    _normalize_action_provenance,
    packb,
    unpack_array,
    unpackb,
)


def test_stage_recorder_preserves_nested_boundaries() -> None:
    recorder = StageRecorder(enabled=True)
    recorder.begin_request(3)
    with recorder.stage("outer", kind="policy_inference"):
        with recorder.stage("inner", kind="action_decode"):
            pass

    collection = recorder.finish_request(request_duration_ms=2.5)

    assert collection is not None
    assert collection["schema"] == SERVER_STAGES_SCHEMA
    assert collection["request_index"] == 3
    assert [stage["name"] for stage in collection["stages"]] == ["outer", "inner"]
    assert [stage["depth"] for stage in collection["stages"]] == [0, 1]
    assert all(stage["status"] == "ok" for stage in collection["stages"])
    assert all(stage["duration_ms"] >= 0 for stage in collection["stages"])


def test_stage_recorder_marks_failed_stage_and_recovers() -> None:
    recorder = StageRecorder(enabled=True)
    recorder.begin_request(0)
    with pytest.raises(RuntimeError, match="boom"):
        with recorder.stage("model", kind="policy_inference"):
            raise RuntimeError("boom")

    collection = recorder.finish_request(request_duration_ms=1.0)

    assert collection is not None
    assert collection["stages"][0]["status"] == "error"
    recorder.begin_request(1)
    assert recorder.finish_request(request_duration_ms=0.0)["request_index"] == 1


def test_disabled_recorder_is_a_noop_without_request() -> None:
    recorder = StageRecorder(enabled=False)
    with recorder.stage("ignored", kind="ignored", synchronize=True):
        pass
    assert recorder.finish_request(request_duration_ms=float("nan")) is None


def test_stage_log_refuses_to_mix_runs(tmp_path) -> None:
    path = tmp_path / "requests.jsonl"
    sink = RequestStageLog(path)
    sink.append({"schema": SERVER_REQUEST_SCHEMA, "request_index": 0})

    assert json.loads(path.read_text(encoding="utf-8"))["request_index"] == 0
    with pytest.raises(FileExistsError):
        RequestStageLog(path)


def test_msgpack_round_trip_and_object_array_rejection() -> None:
    value = {"action": np.arange(4, dtype=np.float32)}
    decoded = unpackb(packb(value))
    np.testing.assert_array_equal(decoded["action"], value["action"])

    with pytest.raises(ValueError, match="unsupported dtype"):
        packb({"bad": np.asarray([object()], dtype=object)})

    with pytest.raises(ValueError, match="unsupported dtype"):
        unpack_array(
            {
                b"__ndarray__": True,
                b"data": b"\0" * np.dtype(object).itemsize,
                b"dtype": np.dtype(object).str,
                b"shape": [1],
            }
        )
    with pytest.raises(ValueError, match="payload size"):
        unpack_array(
            {
                b"__ndarray__": True,
                b"data": b"\0",
                b"dtype": np.dtype(np.float32).str,
                b"shape": [1],
            }
        )


def test_action_provenance_rejects_noncausal_buffer_reference() -> None:
    with pytest.raises(ValueError, match="earlier request"):
        _normalize_action_provenance(
            {
                "schema": "b1k_action_provenance_v1",
                "status": "current_observation_not_used",
                "inference_executed": False,
                "request_index": 2,
                "source_request_index": 2,
                "plan_id": 0,
                "action_index_in_plan": 1,
            }
        )


def test_request_index_survives_websocket_reconnect() -> None:
    class FakePolicy:
        def __init__(self) -> None:
            self.request_index = 0
            self.last_action_provenance = None

        def reset(self) -> None:
            self.request_index = 0

        def act(self, _obs: dict) -> np.ndarray:
            request_index = self.request_index
            self.last_action_provenance = {
                "schema": "b1k_action_provenance_v1",
                "status": (
                    "current_observation_used"
                    if request_index == 0
                    else "current_observation_not_used"
                ),
                "inference_executed": request_index == 0,
                "request_index": request_index,
                "source_request_index": 0,
                "plan_id": 0,
                "action_index_in_plan": request_index,
            }
            self.request_index += 1
            return np.zeros(1, dtype=np.float32)

    class FakeWebsocket:
        remote_address = ("127.0.0.1", 12345)

        def __init__(self, requests: list[dict]) -> None:
            self.requests = [packb(request) for request in requests]
            self.sent: list[bytes] = []

        async def recv(self) -> bytes:
            if not self.requests:
                raise websockets.ConnectionClosed(None, None)
            return self.requests.pop(0)

        async def send(self, message: bytes) -> None:
            self.sent.append(message)

    policy = FakePolicy()
    server = InstrumentedWebsocketPolicyServer(
        policy=policy,
        recorder=StageRecorder(enabled=False),
        host="127.0.0.1",
        port=8000,
    )
    first_connection = FakeWebsocket([{"reset": True}, {"obs": 1}, {"obs": 2}])
    second_connection = FakeWebsocket([{"obs": 3}])

    async def exercise_server() -> None:
        await server._serve_connection(first_connection)
        await server._serve_connection(second_connection)

    asyncio.run(exercise_server())

    responses = [
        unpackb(message)
        for message in first_connection.sent[1:] + second_connection.sent[1:]
    ]
    assert [
        response["action_provenance"]["request_index"] for response in responses
    ] == [0, 1, 2]
