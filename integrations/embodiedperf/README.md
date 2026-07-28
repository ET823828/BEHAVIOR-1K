# EmbodiedPerf model-server kit

BEHAVIOR evaluation runs the simulator and model server in separate processes.
The evaluator hook can measure the WebSocket wait, environment step, and full
episode, but it cannot truthfully mark model work on the server's JAX/XLA or
PyTorch/CUDA runtime. This kit adds optional model-side boundaries to the two
official 2026 baselines without moving model code into the simulator process.

| Model | Required model fork | Pinned revision | Entry point |
|---|---|---|---|
| π0.5 | `ET823828/openpi@feat/embodiedperf-profiler` | `95569716613166609b7dc74489f36768e36b5df5` | `integrations.embodiedperf.model_servers.serve_pi05` |
| GR00T N1.7 | `ET823828/Isaac-GR00T@feat/embodiedperf-profiler` | `6120233a4314736a07958749c120a0868946e132` | `integrations.embodiedperf.model_servers.serve_gr00t` |

This adapter is validated against EmbodiedPerf `0.2.0a8` at
`33d7853330a4d701bc13ca35e70d02f7a0dad301`.

The model forks provide the corrected BEHAVIOR observation namespace, exact
task-prompt handling, and `b1k_action_provenance_v1`. The entry points in this
repository own only the model boundary wrappers and compatible WebSocket
transport. Request timing and validation come from EmbodiedPerf's
benchmark-neutral `RemoteStageRecorder`.

## Hook contract

Hooks are disabled unless `--embodiedperf-stage-log PATH` is passed. When
enabled, every successful policy request is appended to a fresh JSONL file and
the same record is returned under the reserved WebSocket response key
`_embodiedperf_profile`. The evaluator client removes that field, returns the
action through the unchanged policy API, and calls
`episode.attach_remote(record)`.

Each log row is a transport-neutral `embodiedperf.remote_request.v1` record:

- `source_session_id` and `request_id` provide globally unique replay
  protection;
- `duration_ms` and nested stages use only the policy server's `perf_counter`;
- `metadata` carries BEHAVIOR-specific `reset_index`, `request_index`,
  `server_timing`, and `b1k_action_provenance_v1`.

The log path must not already exist. This fail-closed rule prevents records
from separate server runs being silently mixed. The core recorder contains no
WebSocket, BEHAVIOR, OpenPI, GR00T, NumPy, or PyTorch dependency; a future
server-side policy can reuse it and define only its real model boundaries.

### π0.5 boundaries

| Stage | Kind | Exact boundary |
|---|---|---|
| `pi05_b1k_observation_preprocess` | `observation_preprocess` | Camera resize/stack plus BEHAVIOR proprio/prompt packaging in `B1KPolicyWrapper.process_input` |
| `pi05_observation_encode` | `observation_encode` | OpenPI input transforms, batching/device-array conversion, and `Observation.from_dict`; profiling waits for device placement before closing the stage |
| `pi05_model_sample_actions` | `policy_inference` | The JIT-compiled `sample_actions` call; profiling waits with `jax.block_until_ready` before closing the stage |
| `pi05_action_decode` | `action_decode` | Device-to-host conversion plus OpenPI output transforms/unnormalization |

`pi05_model_sample_actions` intentionally remains one stage. The current
OpenPI JAX executable fuses vision-language and flow/action-expert work; this
script does not invent a vision/action split that is unavailable at its real
runtime boundary. In receding-horizon mode, buffered-action requests contain
the B1K preprocessing stage but no model-sampling or action-decode stage.

### GR00T boundaries

| Stage | Kind | Exact boundary |
|---|---|---|
| `gr00t_b1k_observation_preprocess` | `observation_preprocess` | BEHAVIOR camera/state/language packaging |
| `gr00t_observation_encode` | `observation_encode` | `VLAStepData`, processor, collator, and dtype conversion |
| `gr00t_model_input_prepare` | `observation_encode` | Backbone/action input construction and device transfer |
| `gr00t_backbone` | `policy_inference` | `model.backbone(...)` |
| `gr00t_action_head` | `policy_inference` | `model.action_head.get_action(...)`, including iterative action generation |
| `gr00t_action_decode` | `action_decode` | Device-to-host action transfer, unnormalization, and physical action reconstruction |

The four GPU-facing GR00T boundaries synchronize CUDA before and after the
marked call. Their durations are synchronized host-wall intervals, not sums of
individual CUDA kernels.

## Run π0.5

Use absolute paths so checkpoint and asset resolution does not depend on the
BEHAVIOR checkout:

```bash
export BEHAVIOR_DIR=/path/to/BEHAVIOR-1K
export EMBODIEDPERF_DIR=/path/to/embodiedperf
export OPENPI_DIR=/path/to/openpi
export STAGE_LOG=/path/to/fresh-output/pi05-server-stages.jsonl

cd "$OPENPI_DIR"
PYTHONPATH="$BEHAVIOR_DIR:$EMBODIEDPERF_DIR/src:$OPENPI_DIR/src:$OPENPI_DIR/packages/openpi-client/src" \
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
.venv/bin/python -m integrations.embodiedperf.model_servers.serve_pi05 \
  --robot b1k/R1Pro \
  --task "b1k/$TASK_NAME" \
  --repo-id "$REPO_ID" \
  --policy.config pi05_b1k \
  --policy.dir "$PATH_TO_CKPT" \
  --control-mode receding_horizon \
  --action-horizon 16 \
  --port 8000 \
  --embodiedperf-stage-log "$STAGE_LOG"
```

## Run GR00T N1.7

The prompt must be the exact instruction passed to the evaluator's
`--embodiedperf-instruction`:

```bash
export BEHAVIOR_DIR=/path/to/BEHAVIOR-1K
export EMBODIEDPERF_DIR=/path/to/embodiedperf
export GROOT_DIR=/path/to/Isaac-GR00T
export STAGE_LOG=/path/to/fresh-output/gr00t-server-stages.jsonl

cd "$GROOT_DIR"
PYTHONPATH="$BEHAVIOR_DIR:$EMBODIEDPERF_DIR/src:$GROOT_DIR" CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python -m integrations.embodiedperf.model_servers.serve_gr00t \
  --model-path "$PATH_TO_CKPT" \
  --modality-config-path examples/b1k/r1pro.py \
  --embodiment-tag NEW_EMBODIMENT \
  --text-prompt "$TASK_INSTRUCTION" \
  --host 127.0.0.1 \
  --port 8000 \
  --embodiedperf-stage-log "$STAGE_LOG"
```

Install the same EmbodiedPerf revision in the evaluator and model-server
environments (the `PYTHONPATH` examples above are convenient for checkout
development). Run the normal BEHAVIOR evaluator in its separate `behavior`
environment. Omit `--embodiedperf-stage-log` to disable model-stage
synchronization and logging.

## Interpretation and validation

The evaluator report aggregates remote request and stage mean/p95 latency by
source and kind. Evaluator and policy-server process clocks are different, so
remote spans are intentionally absent from the local Perfetto timeline.
`source_session_id`/`request_id` prevent replay; BEHAVIOR correlation remains
available in record metadata. Do not align raw timestamps across processes
without an explicit clock-synchronization method.

Evaluator CPU/RSS sampling still covers only the simulator/evaluator process
tree. Selected-GPU telemetry is a whole-device observation on the evaluator
host, so it may include a co-located server but is not attributed to a remote
stage. A server on another host or device needs its own profiler deployment if
server resource metrics are required.

Before publishing model-stage latency, run the same workload with hooks off
and on and report the matched overhead. Synchronization makes stage boundaries
interpretable but can serialize work that is otherwise asynchronous.

CPU-only validation:

```bash
python -m pytest -q integrations/embodiedperf/model_servers/test_hooks.py
python -m py_compile integrations/embodiedperf/model_servers/*.py
```

Full validation requires loading each real checkpoint, serving at least two
rollouts, checking monotonically increasing request indices within every reset
group, and confirming that fresh-inference provenance agrees with the presence
of model stages.
