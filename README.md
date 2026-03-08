# anemll-profile

ANE (Apple Neural Engine) profiler for CoreML models. Analyzes per-op cost estimates, device placement, and measures actual prediction throughput.

## Install

```bash
brew install anemll/tap/anemll-profile
```

Or build from source:
```bash
make && sudo make install
```

## Usage

```bash
anemll-profile model.mlmodelc
anemll-profile model.mlpackage
anemll-profile /path/to/model          # auto-detects .mlmodelc or .mlpackage
anemll-profile -a model.mlmodelc       # include GPU in device assignment
```

## What it reports

- **Op-Type Runtime Breakdown** — per-op-type estimated runtime, GFLOP/s, GB/s, memory/compute bound
- **Measured Prediction** — actual wall-clock time, iter/s, weight bandwidth GB/s
- **Top Expensive Ops** — the 20 slowest operations
- **Conv Detail** — convolution ops with channel counts and work unit efficiency
- **CPU/GPU Fallback** — ops not on ANE with specific compiler reasons (e.g., "Cannot support standalone slice_update", "Unsupported tensor data type: int32")

## How it works

1. Loads `MLComputePlan` to get per-op device assignment and cost weights
2. Captures Espresso `[CostModelFeature]` logs via forked `/usr/bin/log stream`
3. Parses `Unsupported op` compiler messages for ANE fallback reasons
4. Runs actual predictions with dummy inputs to measure real throughput
5. Computes weight-only DRAM bandwidth (excludes L2-resident activations)

## Requirements

- macOS 14+ (Sonoma) — requires `MLComputePlan` API
- Xcode Command Line Tools

## License

MIT
