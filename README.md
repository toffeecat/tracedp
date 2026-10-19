# TraceDP evaluation: command-line guide

Run every command from the repository root.

## 1. Install

```bash
cd lh_platform
pip install -e .
```

Requires Python 3.11 or later. Dependencies are listed in `lh_platform/pyproject.toml` and `lh_platform/requirements-dev.lock.txt`.

## 2. Environment variables

| Variable | Purpose |
|---|---|
| `LONGH_STORAGE_ROOT` | Storage root (required by the scripts). Cells, run records and receipts are written under `runs/traceback/controlled_injection_v1/` beneath it |
| `LONGH_TERMINAL_BENCH_ROOT` | Terminal-Bench 2.1 root; must contain `tasks/` |
| `LONGH_PI_EXECUTABLE` | Path to the Pi executable; if unset, `pi` from `PATH` is used |
| `LONGH_DOCKER_CONTEXT` | Docker context; default `default` |
| `LONGH_HARBOR` | Entry point for Harbor 0.21.0; required for paid runs |
| `MINIMAX_CN_API_KEY` | API key for the MiniMax-M3 executor; endpoint `https://api.minimaxi.com/v1` |
| `OPENCODE_API_KEY` | API key for the opencode-go executor (hosted Pi runs) |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL` | OpenAI-compatible endpoint. The `traceback` commands of `lh-platform` read `OPENAI_API_KEY` by default |

Pass API keys through environment variables. Do not write them into files or commit them:

```bash
export MINIMAX_CN_API_KEY='<your key>'
```

## 3. Data requirements

The repository does not include source trajectories, task directories or frozen menus. Before running, prepare:

- Source trajectory run directories under `$LONGH_STORAGE_ROOT/runs/traceback/` (naming rule: `CENSUS_RUN` in `lh_platform/scripts/controlled_injection.py`)
- Terminal-Bench 2.1 tasks under `$LONGH_TERMINAL_BENCH_ROOT/tasks/`
- Frozen menus at `$LONGH_STORAGE_ROOT/runs/traceback/controlled_injection_v1/menus/<task>/frozen_menu.json`

Building cells requires Docker for container replay.

## 4. Build injected cells (no model calls)

```bash
python3 lh_platform/scripts/controlled_injection.py build-cells \
  --task <task-name> \
  --menu "$LONGH_STORAGE_ROOT/runs/traceback/controlled_injection_v1/menus/<task-name>/frozen_menu.json" \
  --job-root "$LONGH_STORAGE_ROOT/runs/traceback/controlled_injection_v1/cell_replays" \
  --cells-dir "$LONGH_STORAGE_ROOT/runs/traceback/controlled_injection_v1/cells" \
  --tag <batch-tag>
```

`--tag` separates batches of the same task so that replay task names do not collide.

## 5. Paired runs (model calls; API key required)

```bash
python3 lh_platform/scripts/run_controlled_injection_pairs.py <cell.json> [<cell.json> ...] \
  --menus-root <menus-dir> \
  --template <matrix-template> \
  --matrix-root <matrix-dir> \
  --requests-root <requests-dir> \
  --manifest <manifest.jsonl> \
  --corpus-root <corpus-dir> \
  --repeats 3
```

`--corpus-root` points to a reference corpus directory; this repository does not ship it. `--arms` selects the arms to run; by default all arms run.

## 6. Summarize run results

```bash
python3 lh_platform/scripts/summarize_injection_pairs.py <manifest.jsonl> [<manifest.jsonl> ...] --json summary.json
```

## 7. Baseline bridge arm (Who&When binary search + TraceElephant-style bridging)

```bash
python3 lh_platform/scripts/run_binary_bridge_arm.py \
  --stage {locate,prepare,run} \
  --matrix <matrix> \
  --source-session <session> \
  --run-id <run-id> \
  --request <request.json> \
  --corpus-root <corpus-dir>
```

`--stage` selects one stage per run. `--arm-id` selects a variant of the bridge arm.

## 8. TraceDP v3 planning stage

```bash
python3 lh_platform/scripts/run_supervisor_plan_stage.py \
  --matrix <matrix> \
  --source-session <session> \
  --request <request.json> \
  --corpus-root <corpus-dir> \
  --cell <cell.json> \
  [--arm-id <arm-id>] [--dry-run]
```

## 9. Run the official verifier only

```bash
python3 lh_platform/scripts/verify_only.py \
  --task <task-name> \
  --image <docker-image> \
  --label <label> \
  --job-root <job-dir> \
  --out <result.json> \
  [--attempts 3]
```

## 10. Unified command line (trace methods)

After installation, the `lh-platform` command is available:

```bash
lh-platform list-trace-methods
lh-platform run-trace-method <method_id> <request.json> \
  --corpus-root <corpus-dir> \
  --model <model-id> \
  --base-url <https://.../v1> \
  --output-path <new-receipt.json>
```

According to `--help`, `--output-path` should be a new, immutable path. Do not overwrite an existing file.
