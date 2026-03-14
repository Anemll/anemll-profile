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
anemll-profile --list-functions model.mlpackage
anemll-profile --function add_one model.mlpackage
anemll-profile --all-functions model.mlpackage
```

## Multifunction Models

`anemll-profile` now supports Core ML multifunction model assets.

- Plain `anemll-profile model.mlpackage` profiles the model's default function
- `--list-functions` prints the available function names
- `--function NAME` profiles one specific function
- `--all-functions` emits a full report for each function in sequence

For multi-function model selection and discovery, Core ML requires the macOS 15+ APIs
(`MLModelAsset` and `MLModelConfiguration.functionName`). Plain single-function profiling
continues to work on macOS 14+.

## What it reports

- **Op-Type Runtime Breakdown** — per-op-type estimated runtime, GFLOP/s, GB/s, memory/compute bound
- **Measured Prediction** — actual wall-clock time, iter/s, weight bandwidth GB/s
- **Top Expensive Ops** — the 20 slowest operations
- **Conv Detail** — convolution ops with channel counts and work unit efficiency
- **CPU/GPU Fallback** — ops not on ANE with specific compiler reasons (e.g., "Cannot support standalone slice_update", "Unsupported tensor data type: int32")
- **Function Routing** — the active/default function or explicitly selected function for multi-function models

## How it works

1. Loads `MLComputePlan` to get per-op device assignment and cost weights
2. Captures Espresso `[CostModelFeature]` logs via forked `/usr/bin/log stream`
3. Parses `Unsupported op` compiler messages for ANE fallback reasons
4. Runs actual predictions with dummy inputs to measure real throughput
5. Computes weight-only DRAM bandwidth (excludes L2-resident activations)

## Requirements

- macOS 14+ (Sonoma) — requires `MLComputePlan` API
- macOS 15+ — required for `--list-functions`, `--function`, and `--all-functions`
- Xcode Command Line Tools

## Development

```bash
make test
```

## License

MIT
