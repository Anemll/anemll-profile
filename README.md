# anemll-profile

<img src="assets/anemll-profile-icon.png" alt="ANEMLL Profiler Icon" width="240">

ANE (Apple Neural Engine) profiler for Core ML models. Analyzes per-op cost estimates, device placement, and measures actual prediction throughput.

Designed for agent and developer workflows when profiling and optimizing Apple Neural Engine Core ML models. The text report is human-readable, and the JSON export is intended to make it easy for agents to inspect placement, interruptions, fallbacks, and measured performance.

For a compact agent-oriented workflow, see [AGENTS.md](AGENTS.md).

## Install

```bash
brew install anemll/tap/anemll-profile
```

Or build from source:
```bash
make && sudo make install
```

## Upgrade

If installed with Homebrew:
```bash
brew update && brew upgrade anemll/tap/anemll-profile
```

If installed from source:
```bash
git pull
make && sudo make install
```

Verify the installed version:
```bash
anemll-profile --version
```

## What's New

- **Agent guide (`AGENTS.md`)** — a first-class workflow guide for agents, including install steps, required dependencies, recommended JSON usage, and how to interpret profiler output for optimization work
- **ANE graph interruptions** — interruption analysis highlights non-ANE islands that break continuous ANE execution; the detour can be CPU or GPU, not just CPU
- **Latency-ranked interruption hot spots** — interruptions are ranked by estimated switch penalty plus island runtime to highlight the most expensive detours first
- **Function timeline view** — each function now shows a compact accelerator timeline with labeled interruption islands and ignored leading/trailing non-accelerator runs
- **Configurable switch heuristic** — use `--interrupt-ms` / `--interrupt-boundary-ms` to tune the estimated cost per accelerator boundary
- **Structured JSON export** — use `-j` / `--json FILE` to write a machine-readable report for agents and downstream tooling

## Usage

```bash
anemll-profile model.mlmodelc
anemll-profile model.mlpackage
anemll-profile /path/to/model          # auto-detects .mlmodelc or .mlpackage
anemll-profile -a model.mlmodelc       # include GPU in device assignment
anemll-profile --interrupt-ms 150 model.mlmodelc   # change heuristic ANE boundary cost
anemll-profile -j report.json model.mlmodelc       # write structured JSON report
anemll-profile --list-functions model.mlpackage    # list multifunction model functions
anemll-profile --function add_one model.mlpackage  # profile a specific function
anemll-profile --all-functions model.mlpackage     # profile every function in sequence
```

## Multifunction Models

`anemll-profile` supports Core ML multifunction model assets.

- Plain `anemll-profile model.mlpackage` profiles the model's default function
- `--list-functions` prints the available function names
- `--function NAME` profiles a specific function
- `--all-functions` emits a full report for each function in sequence

Multi-function selection and discovery require the macOS 15+ Core ML APIs
(`MLModelAsset` and `MLModelConfiguration.functionName`). Plain single-function
profiling continues to work on macOS 14+.

## What it reports

- **Op-Type Runtime Breakdown** — per-op-type estimated runtime, GFLOP/s, GB/s, memory/compute bound
- **ANE Graph Interruptions** — detects non-ANE islands that interrupt ANE execution, ranks them by estimated latency tax, and shows a function timeline of where they occur
- **Measured Prediction** — actual wall-clock time, iter/s, weight bandwidth GB/s
- **Top Expensive Ops** — the 20 slowest operations
- **Conv Detail** — convolution ops with channel counts and work unit efficiency
- **CPU/GPU Fallback** — ops not on ANE with specific compiler reasons (e.g., "Cannot support standalone slice_update", "Unsupported tensor data type: int32")
- **Function Routing** — the active/default function or explicitly selected function for multi-function models

## How it works

1. Loads `MLComputePlan` to get per-op device assignment and cost weights
2. Captures Espresso `[CostModelFeature]` logs via forked `/usr/bin/log stream`
3. Parses `Unsupported op` compiler messages for ANE fallback reasons
4. Analyzes ordered MLComputePlan ops to find ANE graph interruption islands, including CPU or GPU detours
5. Applies a heuristic ANE boundary penalty (300 ms by default) to rank interruption hot spots
6. Runs actual predictions with dummy inputs to measure real throughput
7. Computes weight-only DRAM bandwidth (excludes L2-resident activations)

## Requirements

- macOS 14+ (Sonoma) — requires `MLComputePlan` API
- macOS 15+ — required for `--list-functions`, `--function`, and `--all-functions`
- Xcode Command Line Tools
- `Foundation` and `CoreML` system frameworks included with macOS

Recommended for agent automation:

- `jq` for parsing `-j` JSON reports

## Development

```bash
make test
```

## License

MIT
