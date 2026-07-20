# System profiling with EmbodiedPerf

The challenge evaluator has an optional, default-off integration with
[EmbodiedPerf](https://github.com/ET823828/embodiedperf). BEHAVIOR owns only the
episode and evaluator-call boundaries; the installed Python package owns system
sampling, validation, aggregation, timelines, and report rendering.

## What is measured

One profiler episode starts after `Evaluator.reset()` and ends when the rollout
terminates. Task loading, reset, checkpoint loading, result serialization, and
the generated report are outside that interval.

The integration records:

- active-rollout episode latency and success;
- client-observed WebSocket policy round trips, including serialization and
  response decoding;
- every `env.step()` wall interval;
- evaluator process-tree CPU utilization and RSS;
- utilization, power, episode energy, and memory for the explicitly selected whole GPUs;
- one representative host timeline, plus a Perfetto version of that timeline.

Client-side cached actions are not labeled as WebSocket waits. Time outside the
two explicit stages remains visible as a gap inside the episode instead of being
assigned to a misleading category.

The first raw trace in each evaluator process is retained as cold start and
excluded from every aggregate. Pass `--embodiedperf-warmup-instance-index 0` to
run that trace as a dedicated, unreported rollout before the selected instances.
The normal challenge JSON and video formats do not change.

## Scope limits

`--embodiedperf-gpu-ids` accepts physical NVML ids. Power and memory are sums
over those whole GPUs, while utilization is their sampled mean. Include both the
simulator and local policy-server GPUs for full-stack telemetry, or only the
policy GPU for model-serving telemetry. A remote policy server cannot be
measured by the evaluator.

CPU covers the evaluator process tree. WebSocket timing is an evaluator-observed
round trip, not model execution time. The minimal adapter does not instrument
action readiness or model internals; those require separate policy-server
instrumentation.

## Setup

Install the optional runtime in the existing `behavior` environment:

```bash
conda activate behavior
python -m pip install \
  "embodiedperf[behavior1k] @ git+https://github.com/ET823828/embodiedperf.git@8f46d6d59480555e0ef795b002b58af78a0bcd4b"
```

Keep the baseline policy server running, then enable profiling on the evaluator:

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
  --embodiedperf-checkpoint /absolute/path/to/checkpoint \
  --embodiedperf-instruction "Turn on the radio receiver that's on the table in the living room." \
  --embodiedperf-gpu-ids 0 1
```

Use a fresh profiler output directory for every run. `--write-video` remains
available, but its per-step frame composition and encoding occur inside the
control loop and therefore contribute to measured episode latency when enabled;
the final writer flush occurs after the measured interval.

Artifacts are written under `<output-dir>/embodiedperf/` by default:

```text
embodiedperf/
├── traces.jsonl
├── cpu_profile/
├── gpu_profile/
└── report/
    ├── index.html
    ├── summary.json
    ├── validation.json
    └── representative_timeline.perfetto.json
```

The finalizer fails rather than replacing missing spans or resource samples with
zero. The HTML contains aggregate metrics, including `compute_energy_j` and its
compatibility alias `energy_proxy_j`, plus the first warm episode timeline. Raw
per-episode artifacts remain on disk but are not listed in the report.
