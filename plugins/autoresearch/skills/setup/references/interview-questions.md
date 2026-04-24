# Interview questions — verbatim copy

Read this when constructing the `AskUserQuestion` call if uncertain about wording, option ordering, or direction mapping. These are the canonical question/option strings; deviate only if the project-specific context genuinely requires it (e.g. a custom runner that needs a different label).

## The call

A single `AskUserQuestion` invocation with six questions, all in one shot. Do not split into multiple calls — the UX goal is a single compact form the user fills in once.

## Question 1 — research goal

- **header**: `Research goal`
- **question**: `What should this experiment optimize for? Pick a theme or describe a custom one.`
- **multiSelect**: `false`
- **options**: 3–4 inferred candidates + `"Other (describe)"` as the last option. Each candidate is a short phrase (≤ 60 chars). Example candidates (generic — real options come from repo analysis):
  - `Improve val/loss via loss-term ablation`
  - `Reduce peak VRAM in attention block`
  - `Explore projector depth`
  - `Sweep learning-rate schedules`
  - `Other (describe)`
- If the user picks "Other", the free-text becomes the goal. Otherwise the chosen phrase is the goal verbatim.

## Question 2 — primary metric

- **header**: `Primary metric`
- **question**: `Which wandb metric decides whether a run improved? (direction matters.)`
- **multiSelect**: `false`
- **options**: top metric candidates from wandb summary scan, labeled with direction:
  - `val/loss (min)`
  - `val/bpb (min)`
  - `eval/top1 (max)`
  - `train/loss (min)`
  - `Other (type metric=direction)`
- If the user picks "Other", prompt for freeform input as `name=min` or `name=max` — parse accordingly. If direction is ambiguous, re-ask for direction only.

## Question 3 — hard constraints

- **header**: `Hard constraints`
- **question**: `Constraints that invalidate a run if violated. Pick any / none.`
- **multiSelect**: `true`
- **options**: `"None"` is the default; 2–3 resource-bound candidates if detected, plus free-text:
  - `None`
  - `peak_vram_mb <= 45000`
  - `tokens_per_sec >= 5000`
  - `wall_seconds <= 600`
  - `Other (name op value)`
- Parse each selected constraint into `{name, op, threshold}`. Operators: `<=`, `<`, `>=`, `>`. If "None" is selected alongside others, treat the others as authoritative and ignore "None".

## Question 4 — mutation scope

- **header**: `Mutation scope`
- **question**: `Modules / classes train.py may monkey-patch. Advisory only — not enforced in code.`
- **multiSelect**: `true`
- **options**: dotted-path candidates from Phase 1, multi-select, with escape hatch. The paths should be namespaced under the host project's package — e.g. `<your_project>.training.losses.YourLoss`, not bare `training.losses.YourLoss`:
  - `<your_project>.training.losses.YourLoss`
  - `<your_project>.model.attn.YourBlock`
  - `<your_project>.model.projector.Projector`
  - `Other (comma-separated dotted paths)`
  - `(advisory — will skip)`
- If `(advisory — will skip)` is selected, omit `--mutation-scope` from `ar init`.

## Question 5 — default run duration

- **header**: `Run duration`
- **question**: `Default per-run time budget in seconds (overridable per autoresearch:run).`
- **multiSelect**: `false`
- **options**:
  - `300` (5 min — small models, quick ablations)
  - `600` (10 min — default)
  - `900` (15 min)
  - `1800` (30 min — heavy models, multi-GPU)
  - `Custom`

## Question 6 — runner

- **header**: `Runner`
- **question**: `How should ar launch training? Pick the inferred one or customize.`
- **multiSelect**: `false`
- **options**: Phase 1's inferred runner first, then the others, then Custom:
  - `accelerate launch --config_file configs/accelerate.yaml`  (example; use the detected DDP config file)
  - `torchrun --nproc-per-node N`  (substitute the project's actual GPU count)
  - `python`
  - `Custom`
- The exact string is stored and invoked verbatim by `ar`, so include `--config_file`, `--nproc-per-node`, and similar args.

## Direction-inference table

Use this when auto-labeling metric candidates in Question 2. If a key matches multiple substrings, the first match wins.

| Substring (case-insensitive) | Direction |
|---|---|
| `loss`, `nll`, `ppl`, `perplex`, `bpb`, `bits_per`, `error`, `wer`, `cer`, `fid`, `lpips` | min |
| `acc`, `top1`, `top5`, `auc`, `f1`, `recall`, `precision`, `bleu`, `rouge`, `exact_match`, `em`, `map`, `iou`, `dice`, `psnr`, `ssim`, `reward`, `score`, `win_rate` | max |
| `vram`, `memory`, `mb`, `gb`, `wall`, `time`, `latency`, `seconds` | min (usually a constraint, not primary) |
| `throughput`, `tokens_per_sec`, `samples_per_sec`, `steps_per_sec` | max (usually a constraint, not primary) |

If none match, label the metric `(direction?)` in the option text and force the user to disambiguate via the free-text path.

## Ordering note

Ask all six questions in a **single** `AskUserQuestion` call. Claude Code's `AskUserQuestion` supports multiple `questions` per call and renders them as a compact form — respect this for UX.
