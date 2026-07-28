from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
import websockets

from embodiedperf import (
    REMOTE_PROFILE_KEY,
    REMOTE_REQUEST_SCHEMA,
    RemoteStageRecorder,
)
from integrations.embodiedperf.model_servers._hooks import (
    InstrumentedWebsocketPolicyServer,
    RequestStageLog,
    _normalize_action_provenance,
    packb,
    unpack_array,
    unpackb,
)


def test_stage_log_refuses_to_mix_runs(tmp_path) -> None:
    path = tmp_path / "requests.jsonl"
    sink = RequestStageLog(path)
    sink.append({"schema": REMOTE_REQUEST_SCHEMA, "request_index": 0})

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


def test_disabled_server_omits_remote_profile_contract() -> None:
    class FakePolicy:
        last_action_provenance = None

        def reset(self) -> None:
            pass

        def act(self, _obs: dict) -> np.ndarray:
            return np.zeros(1, dtype=np.float32)

    class FakeWebsocket:
        remote_address = ("127.0.0.1", 12345)

        def __init__(self) -> None:
            self.requests = [packb({"obs": 1})]
            self.sent: list[bytes] = []

        async def recv(self) -> bytes:
            if not self.requests:
                raise websockets.ConnectionClosed(None, None)
            return self.requests.pop(0)

        async def send(self, message: bytes) -> None:
            self.sent.append(message)

    websocket = FakeWebsocket()
    server = InstrumentedWebsocketPolicyServer(
        policy=FakePolicy(),
        recorder=RemoteStageRecorder(
            source="test-policy-server",
            enabled=False,
        ),
        host="127.0.0.1",
        port=8000,
    )

    asyncio.run(server._serve_connection(websocket))

    metadata, response = [unpackb(message) for message in websocket.sent]
    assert "embodiedperf_remote_schema" not in metadata
    assert REMOTE_PROFILE_KEY not in response
    assert "action" in response


def test_request_index_and_remote_profile_survive_websocket_reconnect(
    tmp_path,
) -> None:
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
    recorder = RemoteStageRecorder(source="test-policy-server")
    server = InstrumentedWebsocketPolicyServer(
        policy=policy,
        recorder=recorder,
        host="127.0.0.1",
        port=8000,
        stage_log_path=tmp_path / "remote.jsonl",
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
    assert all(
        response[REMOTE_PROFILE_KEY]["source"] == "test-policy-server"
        for response in responses
    )
    assert [
        response[REMOTE_PROFILE_KEY]["metadata"]["request_index"]
        for response in responses
    ] == [0, 1, 2]
    assert unpackb(first_connection.sent[0])["embodiedperf_remote_schema"] == (
        REMOTE_REQUEST_SCHEMA
    )
    logged = [
        json.loads(line)
        for line in (tmp_path / "remote.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["metadata"]["request_index"] for row in logged] == [0, 1, 2]
