"""Shared, optional server-side stage hooks for BEHAVIOR model servers.

The evaluator and policy server run in different processes. These records
therefore stay on the policy server's monotonic clock and must not be presented
as evaluator-local spans without an explicit cross-process clock alignment.

Protocol portions are adapted from OpenPI and Isaac-GR00T (Apache-2.0).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
import functools
import http
import json
import logging
import math
from pathlib import Path
import time
import traceback
from typing import Any
from uuid import uuid4

import msgpack
import numpy as np
import websockets
import websockets.asyncio.server as _server


logger = logging.getLogger(__name__)

SERVER_STAGES_SCHEMA = "embodiedperf.server_stages.v1"
SERVER_REQUEST_SCHEMA = "embodiedperf.server_request.v1"
_MAX_ARRAY_BYTES = 256 * 1024 * 1024
_MAX_MESSAGE_BYTES = 256 * 1024 * 1024


def _non_empty_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


class StageRecorder:
    """Collect nested server-clock stages for one synchronous policy request."""

    def __init__(
        self,
        *,
        enabled: bool,
        synchronize: Callable[[], None] | None = None,
    ) -> None:
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool")
        if synchronize is not None and not callable(synchronize):
            raise TypeError("synchronize must be callable when provided")
        self._enabled = enabled
        self._synchronize = synchronize
        self._request_index: int | None = None
        self._request_start_ns: int | None = None
        self._stack: list[dict[str, Any]] = []
        self._stages: list[dict[str, Any]] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    def begin_request(self, request_index: int) -> None:
        if not self._enabled:
            return
        if (
            isinstance(request_index, bool)
            or not isinstance(request_index, int)
            or request_index < 0
        ):
            raise ValueError("request_index must be a non-negative integer")
        if self._request_start_ns is not None:
            raise RuntimeError("a stage request is already active")
        self._request_index = request_index
        self._request_start_ns = time.perf_counter_ns()
        self._stack = []
        self._stages = []

    @contextmanager
    def stage(
        self,
        name: str,
        *,
        kind: str,
        synchronize: bool = False,
    ) -> Iterator[None]:
        """Record one real host boundary, optionally synchronizing its device work."""

        if not self._enabled:
            yield
            return
        if not isinstance(synchronize, bool):
            raise TypeError("synchronize must be a bool")
        if self._request_start_ns is None:
            raise RuntimeError("stage requires an active request")
        if synchronize and self._synchronize is not None:
            self._synchronize()
        stage = {
            "token": uuid4().hex,
            "name": _non_empty_text(name, "stage name"),
            "kind": _non_empty_text(kind, "stage kind"),
            "start_ns": time.perf_counter_ns(),
            "depth": len(self._stack),
        }
        self._stack.append(stage)
        status = "ok"
        try:
            yield
        except BaseException:
            status = "error"
            raise
        finally:
            if synchronize and self._synchronize is not None:
                self._synchronize()
            end_ns = time.perf_counter_ns()
            if not self._stack or self._stack[-1]["token"] != stage["token"]:
                self.abort_request()
                raise RuntimeError(
                    "server stage contexts must close in last-in-first-out order"
                )
            self._stack.pop()
            request_start_ns = self._request_start_ns
            if request_start_ns is None:
                raise RuntimeError("server stage request ended unexpectedly")
            self._stages.append(
                {
                    "name": stage["name"],
                    "kind": stage["kind"],
                    "start_ms": (stage["start_ns"] - request_start_ns) / 1_000_000.0,
                    "end_ms": (end_ns - request_start_ns) / 1_000_000.0,
                    "duration_ms": (end_ns - stage["start_ns"]) / 1_000_000.0,
                    "depth": stage["depth"],
                    "status": status,
                }
            )

    def finish_request(self, *, request_duration_ms: float) -> dict[str, Any] | None:
        if not self._enabled:
            return None
        if self._request_start_ns is None or self._request_index is None:
            raise RuntimeError("no stage request is active")
        if self._stack:
            raise RuntimeError("cannot finish a request while a server stage is open")
        if (
            isinstance(request_duration_ms, bool)
            or not isinstance(request_duration_ms, (int, float))
            or not math.isfinite(request_duration_ms)
            or request_duration_ms < 0
        ):
            raise ValueError("request_duration_ms must be finite and non-negative")
        collection = {
            "schema": SERVER_STAGES_SCHEMA,
            "clock": "server_process_perf_counter",
            "request_index": self._request_index,
            "request_duration_ms": float(request_duration_ms),
            "stages": sorted(
                self._stages, key=lambda item: (item["start_ms"], -item["end_ms"])
            ),
        }
        self._request_index = None
        self._request_start_ns = None
        self._stack = []
        self._stages = []
        return collection

    def abort_request(self) -> None:
        self._request_index = None
        self._request_start_ns = None
        self._stack = []
        self._stages = []


class RequestStageLog:
    """Fail-closed JSONL sink for model-server request stages."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("x", encoding="utf-8"):
            pass

    def append(self, record: Mapping[str, Any]) -> None:
        if not isinstance(record, Mapping):
            raise TypeError("stage-log record must be a mapping")
        payload = json.dumps(dict(record), sort_keys=True, allow_nan=False)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(payload)
            stream.write("\n")


class InstrumentedWebsocketPolicyServer:
    """BEHAVIOR-compatible policy server that can emit optional stage records."""

    def __init__(
        self,
        *,
        policy: Any,
        recorder: StageRecorder,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: Mapping[str, Any] | None = None,
        stage_log_path: str | Path | None = None,
    ) -> None:
        if not isinstance(recorder, StageRecorder):
            raise TypeError("recorder must be a StageRecorder")
        if (
            isinstance(port, bool)
            or not isinstance(port, int)
            or not 1 <= port <= 65_535
        ):
            raise ValueError("port must be an integer between 1 and 65535")
        if recorder.enabled != (stage_log_path is not None):
            raise ValueError("stage hooks and stage_log_path must be enabled together")
        self._policy = policy
        self._recorder = recorder
        self._host = _non_empty_text(host, "host")
        self._port = port
        self._stage_log = (
            RequestStageLog(stage_log_path) if stage_log_path is not None else None
        )
        self._metadata = dict(metadata or {})
        self._server_session_id = uuid4().hex
        self._metadata["server_session_id"] = self._server_session_id
        if recorder.enabled:
            self._metadata["embodiedperf_server_stages_schema"] = SERVER_STAGES_SCHEMA
        self._connection_lock = asyncio.Lock()
        self._request_index = 0
        self._reset_index = 0

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        logger.info("Starting websocket server on %s:%s", self._host, self._port)
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=_MAX_MESSAGE_BYTES,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: Any) -> None:
        if self._connection_lock.locked():
            await websocket.close(
                code=1013, reason="Policy server already has an active client"
            )
            return
        async with self._connection_lock:
            await self._serve_connection(websocket)

    async def _serve_connection(self, websocket: Any) -> None:
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = Packer()
        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                result = unpackb(await websocket.recv(), strict_map_key=False)
                if not isinstance(result, Mapping):
                    raise TypeError("policy request must decode to a mapping")
                if "reset" in result:
                    self._policy.reset()
                    self._request_index = 0
                    self._reset_index += 1
                    continue

                obs = deepcopy(result)
                infer_start = time.monotonic()
                self._recorder.begin_request(self._request_index)
                action = self._policy.act(obs)
                infer_ms = (time.monotonic() - infer_start) * 1000.0
                stages = self._recorder.finish_request(request_duration_ms=infer_ms)
                if stages is not None:
                    stages["server_session_id"] = self._server_session_id
                    stages["reset_index"] = self._reset_index

                action_provenance = _normalize_action_provenance(
                    getattr(self._policy, "last_action_provenance", None)
                )
                if action_provenance is not None:
                    provenance_index = action_provenance["request_index"]
                    if provenance_index != self._request_index:
                        raise RuntimeError(
                            "policy action provenance request_index does not match the server request"
                        )

                timing = {"infer_ms": infer_ms}
                if prev_total_time is not None:
                    timing["prev_total_ms"] = prev_total_time * 1000.0
                response = {
                    "action": _to_numpy(action),
                    "server_timing": timing,
                }
                if action_provenance is not None:
                    response["action_provenance"] = action_provenance
                if stages is not None:
                    response["embodiedperf_server_stages"] = stages

                await websocket.send(packer.pack(response))
                prev_total_time = time.monotonic() - start_time
                if self._stage_log is not None:
                    self._stage_log.append(
                        {
                            "schema": SERVER_REQUEST_SCHEMA,
                            "server_session_id": self._server_session_id,
                            "reset_index": self._reset_index,
                            "request_index": self._request_index,
                            "server_timing": timing,
                            "action_provenance": action_provenance,
                            "stage_collection": stages,
                        }
                    )
                self._request_index += 1
            except websockets.ConnectionClosed:
                self._recorder.abort_request()
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                self._recorder.abort_request()
                logger.error(
                    "Error in connection from %s:\n%s",
                    websocket.remote_address,
                    traceback.format_exc(),
                )
                try:
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error",
                    )
                except AttributeError:
                    await websocket.close(code=1011, reason="Internal server error")
                raise


def _to_numpy(action: Any) -> np.ndarray:
    cpu = getattr(action, "cpu", None)
    if callable(cpu):
        action = cpu()
    numpy = getattr(action, "numpy", None)
    if callable(numpy):
        action = numpy()
    result = np.asarray(action)
    if result.dtype.kind in {"O", "V", "c"}:
        raise ValueError(f"unsupported action dtype: {result.dtype}")
    if result.nbytes > _MAX_ARRAY_BYTES:
        raise ValueError("action payload exceeds the server safety limit")
    return result


def _normalize_action_provenance(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != "b1k_action_provenance_v1"
    ):
        raise ValueError("action provenance must use schema b1k_action_provenance_v1")
    status = value.get("status")
    if status not in {"current_observation_used", "current_observation_not_used"}:
        raise ValueError("action provenance has an unknown status")
    inference_executed = value.get("inference_executed")
    if type(inference_executed) is not bool or inference_executed != (
        status == "current_observation_used"
    ):
        raise ValueError("action provenance has inconsistent inference semantics")
    normalized = {
        "schema": "b1k_action_provenance_v1",
        "status": status,
        "inference_executed": inference_executed,
    }
    for key in (
        "request_index",
        "source_request_index",
        "plan_id",
        "action_index_in_plan",
    ):
        item = value.get(key)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(
                f"action provenance field {key} must be a non-negative integer"
            )
        normalized[key] = item
    if status == "current_observation_used":
        if normalized["source_request_index"] != normalized["request_index"]:
            raise ValueError(
                "fresh action provenance must reference the current request"
            )
    elif normalized["source_request_index"] >= normalized["request_index"]:
        raise ValueError("buffered action provenance must reference an earlier request")
    return normalized


def _health_check(connection: Any, request: Any) -> Any | None:
    if hasattr(request, "path") and request.path == "/healthz":
        if hasattr(connection, "respond"):
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return http.HTTPStatus.OK, {"Content-Type": "text/plain"}, b"OK\n"
    return None


def pack_array(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def unpack_array(obj: dict[bytes, Any]) -> Any:
    if b"__ndarray__" in obj:
        dtype = np.dtype(obj[b"dtype"])
        if dtype.kind in ("V", "O", "c"):
            raise ValueError(f"unsupported dtype: {dtype}")
        shape = obj[b"shape"]
        if not isinstance(shape, (list, tuple)) or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0
            for size in shape
        ):
            raise ValueError("array shape must contain non-negative integers")
        expected_bytes = math.prod(shape) * dtype.itemsize
        data = obj[b"data"]
        if not isinstance(data, bytes) or len(data) != expected_bytes:
            raise ValueError("array payload size does not match dtype and shape")
        if expected_bytes > _MAX_ARRAY_BYTES:
            raise ValueError("array payload exceeds the server safety limit")
        return np.frombuffer(data, dtype=dtype).reshape(shape)
    if b"__npgeneric__" in obj:
        dtype = np.dtype(obj[b"dtype"])
        if dtype.kind in ("V", "O", "c"):
            raise ValueError(f"unsupported dtype: {dtype}")
        return dtype.type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


__all__ = [
    "SERVER_REQUEST_SCHEMA",
    "SERVER_STAGES_SCHEMA",
    "InstrumentedWebsocketPolicyServer",
    "Packer",
    "RequestStageLog",
    "StageRecorder",
    "packb",
    "unpackb",
]
