# System profiling with EmbodiedPerf

The challenge evaluator has an optional, default-off integration with
[EmbodiedPerf](https://github.com/ET823828/embodiedperf). It preserves the normal
evaluation path when disabled and adds client-observed latency, episode-level
system telemetry, and an automatically generated timeline when enabled.

## Measurement boundary

One profiler episode starts after `Evaluator.reset()` and ends as soon as the
rollout terminates. Task-instance loading, environment reset, policy reset,
checkpoint loading, and result serialization are outside the measured episode.
Video encoding remains inside the active loop when `--write-video` is enabled.

The integration records:

- causal L0 latency from a processed observation being available in the
  evaluator to the corresponding action becoming available, but only when
  validated server provenance says that request consumed the current
  observation;
- the policy server's reported `policy_wrapper.act` duration for every call,
  plus a separate current-observation-only distribution, when the response
  includes `server_timing.infer_ms` and validated action provenance;
- successful active-rollout latency and energy, average power over all warm
  attempts, peak GPU memory, GPU utilization counters, and evaluator-process CPU;
- explicit host stages for WebSocket communication, the OmniGibson environment
  step, and challenge observation preprocessing; client-side cache stages are
  added when that optional evaluator cache is enabled;
- per-episode JSON, static HTML, and Perfetto timeline artifacts.

The first raw trace record in each evaluator invocation is classified as cold
start and excluded from **all aggregate statistics and published warm-episode
references**. The raw record is retained for provenance. Pass
`--embodiedperf-warmup-instance-index 0` to run a dedicated, unreported cold
episode before the selected instances. The standard ten public instances then
produce eleven raw traces: one excluded warmup plus ten warm, reported episodes.

This cold-start-free profiler summary is a systems view, not the official
challenge score: the evaluator's normal per-rollout result files remain
unchanged and retain all selected instances.

Policy action-cache hits, whether implemented in the evaluator or inside the
policy server, remain visible as executed actions but are marked causally
unavailable for current-observation L0. A cached action was generated from an
earlier observation, so its lookup and transport time is not mixed into the
current observation-to-action-ready distribution.

The profiling extension uses the optional `b1k_action_provenance_v1` response
object. A fresh inference reports `current_observation_used`; a later action
drained from the same predicted plan reports `current_observation_not_used`,
together with its request, source-request, plan, and in-plan action indices.
Missing, malformed, or contradictory provenance fails closed: the action still
executes normally, but that call is excluded from causal L0 and counted as
unknown provenance. The field is profiling metadata only and does not change
the challenge observation or action format.

## Scope limits

`--embodiedperf-gpu-ids` uses physical NVML GPU ids. The reported power and
memory cover the sum of exactly those local GPUs, so include both the simulator
and policy-server GPU for full-stack energy, or only the policy GPU for
model-serving energy. Remote-server power and CPU cannot be inferred by the
evaluator. CPU samples cover the evaluator process only.

The host `communication_wait` span is not a vision encoder, backbone, or kernel
timeline. Model-internal L2 requires instrumentation in the policy-server
process. GR00T can use PyTorch module hooks for this; the JAX/XLA π0.5 baseline
requires JAX/XLA-aware instrumentation. The evaluator deliberately does not
label either as available from client-side timing.

| Profiling surface | π0.5 | GR00T N1.7 | Current evaluator integration |
| --- | --- | --- | --- |
| Causal current-observation L0 and active-rollout latency | Supported | Supported | Available with validated provenance |
| Local GPU power, energy, memory, utilization | Supported | Supported | Available for explicit physical GPU ids |
| Server wrapper duration (all calls and current-observation subset) | Supported | Supported | Available with timing and provenance fields |
| Evaluator CPU and host-stage timeline | Supported | Supported | Available; policy-server CPU is out of process |
| Vision/backbone/action-head L2 | JAX/XLA adapter required | Server-side PyTorch markers/hooks required | Not claimed by this client adapter |

This split keeps the benchmark-side patch model-agnostic. A server-side L2
adapter can be added independently without changing the official observation,
action, result, or submission formats.

## Setup

Install EmbodiedPerf into the existing `behavior` environment used by the
evaluator:

```bash
conda activate behavior
python -m pip install \
  "embodiedperf @ git+https://github.com/ET823828/embodiedperf.git@834a23bf5984065fdb53aae910fc718b989ec38a"
```

Keep the normal baseline policy server running, then profile all ten reported
public instances. This example samples physical GPU 0 for OmniGibson and GPU 1
for the local policy server:

```bash
CUDA_VISIBLE_DEVICES=0 python -m omnigibson.eval.eval \
  --task-name turning_on_radio \
  --host 127.0.0.1 --port 8000 \
  --instance-indices 0 1 2 3 4 5 6 7 8 9 \
  --num-rollouts 1 \
  --output-dir outputs/turning_on_radio \
  --embodiedperf \
  --embodiedperf-warmup-instance-index 0 \
  --embodiedperf-model-key pi05_b1k_turning_on_radio \
  --embodiedperf-checkpoint /absolute/path/to/pi05_turn_on_the_radio \
  --embodiedperf-instruction "Turn on the radio receiver that's on the table in the living room." \
  --embodiedperf-gpu-ids 0 1
```

The dedicated warmup never writes an official result JSON or video. Keep
`--write-video` disabled for paper-facing latency, CPU, power, and energy runs:
frame composition and MP4 encoding occur inside the active control loop and
would otherwise become part of those measurements. Run a separate unprofiled
invocation when rollout videos are needed.

For the provided GR00T checkpoint, change only the identity fields:

```bash
  --embodiedperf-model-key groot_n1d7_turning_on_radio \
  --embodiedperf-checkpoint /absolute/path/to/turning_on_radio_GR00T-checkpoint-150000 \
  --embodiedperf-instruction "Turn on the radio receiver that's on the table in the living room."
```

Artifacts are written under `<output-dir>/embodiedperf/` by default:

```text
embodiedperf/
├── traces.jsonl                 # all raw episodes, including cold start
├── summary.json                 # cold-start-free aggregate contract and metrics
├── gpu_profile/                 # per-episode raw GPU samples
├── cpu_profile/                 # evaluator-process action-window samples
├── episode_timeline/            # causal L0 event HTML and Perfetto views
└── semantic_timeline/           # warm-only host-stage HTML and Perfetto views
```

Use `--embodiedperf-output-dir` to choose a different fresh directory and
`--embodiedperf-power-interval-s` to change the default 50 ms NVML cadence. The
profiler refuses to append to existing `traces.jsonl` or `summary.json` files so
that separate runs cannot be mixed silently.
