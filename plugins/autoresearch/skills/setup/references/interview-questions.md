# Interview questions — verbatim copy

Read this when constructing the `AskUserQuestion` call if uncertain about wording, option ordering, or direction mapping. These are the canonical question/option strings; deviate only if the project-specific context genuinely requires it (e.g. a custom runner that needs a different label).

## The call

A single `AskUserQuestion` invocation with up to 13 questions, in one shot. Items 2, 3, and 13 are skipped based on the user's answers to items 1 (entry pattern) and 6 (metric backend). Follow-up free-text AskUserQuestion calls fire only for the tensorboard-glob and custom-extract-code captures; those are mechanical and exempt from the normal prompt budget.

## Question 1 — entry pattern

- **header**: `Entry pattern`
- **question**: `What shape is your training entrypoint? (Drives train.py rendering.)`
- **multiSelect**: `false`
- **options**: Phase 1 detection first (pre-selected), then the rest. Examples:
  - `argparse-cli — runpy-invoke an "if __name__ == '__main__':" script`
  - `hydra — invoke a @hydra.main-decorated entrypoint with key=value overrides`
  - `function — call an importable main(**kwargs)`
  - `custom — I'll wire it manually in train.py`
- If detection was ambiguous on the main module dotted path, ask one free-text follow-up ("Which module is the entry point? e.g. `myproject.training.main`").

## Question 2 — baseline CLI args / Hydra overrides (conditional)

Skip if item 1 was `function` or `custom`.

- **header**: `Baseline overrides`
- **question**: `Baseline overrides the agent starts from (editable in CLI_OVERRIDES per run). Keep or edit?`
- **multiSelect**: `false`
- **options**:
  - `Keep as-is` (pre-selected; show detected dict/list as preview text)
  - `Edit — I'll paste a JSON value`
  - `Empty — start with {} / []`
- Preview for argparse-cli: `{"config":"base","learning_rate":1.5e-4,"num_train_epochs":1}`
- Preview for hydra: `["optimizer.lr=3e-4","model.hidden_dim=512","trainer.max_epochs=1"]`

## Question 3 — resume flag / override key (conditional)

Skip if item 1 was `function` or `custom`.

- **header**: `Resume anchor`
- **question**: `Which flag (argparse) or override key (hydra) resumes training from a checkpoint?`
- **multiSelect**: `false`
- **options** (argparse-cli): detected flag first (`resume_from_checkpoint (detected)` etc.), then:
  - `resume_from_checkpoint`
  - `resume`
  - `ckpt_path`
  - `pretrained_checkpoint`
  - `None — no resume flag (use AR_RESUME_CKPT env var instead)`
  - `Other (type)`
- **options** (hydra): detected key first (`trainer.resume_from_checkpoint (detected)` etc.), then:
  - `trainer.resume_from_checkpoint`
  - `trainer.ckpt_path`
  - `resume_from`
  - `None — use AR_RESUME_CKPT env var`
  - `Other (type)`
- Record name without `--` prefix (or as a dotted override key).

## Question 4 — distributed framework

- **header**: `Distributed framework`
- **question**: `Which distributed-training framework does the host use?`
- **multiSelect**: `false`
- **options**: Phase 1 detection first (pre-selected), then the rest:
  - `accelerate`
  - `deepspeed`
  - `fsdp`
  - `ddp`
  - `lightning` (pytorch-lightning 2.x / lightning.pytorch)
  - `none`
  - `auto — let ar run introspect the runner command`
- Default is `auto` if Phase 1 detection was ambiguous.

## Question 5 — research goal

- **header**: `Research goal`
- **question**: `What should this experiment optimize for? Pick a theme or describe a custom one.`
- **multiSelect**: `false`
- **options**: 3–4 inferred candidates + `Other (describe)` as the last option. Each candidate is a short phrase (≤ 60 chars). Example candidates (generic — real options come from repo analysis):
  - `Improve val/loss via loss-term ablation`
  - `Reduce peak VRAM in attention block`
  - `Explore projector depth`
  - `Sweep learning-rate schedules`
  - `Other (describe)`
- If the user picks `Other`, the free-text becomes the goal. Otherwise the chosen phrase is the goal verbatim.

## Question 6 — metric backend (new in v0.3.0)

- **header**: `Metric backend`
- **question**: `How does training surface metrics the agent should read?`
- **multiSelect**: `false`
- **options**: Phase 1 detection first (pre-selected), then the rest:
  - `wandb — read wandb/run-*/files/wandb-summary.json`
  - `tensorboard — read events.out.tfevents.* scalar events`
  - `log — regex-scan training's stdout (captured into run.log)`
  - `custom — paste a Python snippet that reads run_dir + run_log_text`
  - `auto — try wandb → tensorboard → log on each run`
- **Pushy phrasing** when Phase 1 detection confidence was low (e.g. one stale tfevents file and no wandb runs): "Low-confidence detection — please confirm explicitly; the `auto` fallback will try each in order but a concrete choice avoids per-run probing overhead."
- Follow-up free-text AskUserQuestion (fires only when):
  - `tensorboard` chosen → ask `"Which glob matches the event files?"`, pre-populated with Phase 1's suggestion (e.g. `runs/*/events.out.tfevents.*`, `lightning_logs/*/events.out.tfevents.*`). Becomes `--tb-events-glob`.
  - `custom` chosen → ask `"Paste a short Python body. Receives run_dir: Path and run_log_text: str; must return dict[str, float]. Example: return {'val_loss': float(re.search(r'val_loss=([0-9.]+)', run_log_text).group(1))}"`. Becomes `--metric-extract-code` (base64-encoded).

## Question 7 — checkpoint glob (new in v0.3.0)

- **header**: `Checkpoint glob`
- **question**: `Where does training save checkpoints? AR-SAVE uses this to discover the latest file after each run.`
- **multiSelect**: `false`
- **options**: Phase 1-detected candidates first (pre-selected), then the rest:
  - `output_dir/checkpoint-*/` (HF Trainer)
  - `lightning_logs/*/checkpoints/*.ckpt` (Lightning default)
  - `outputs/*/checkpoints/best.pt` (Hydra + Lightning composed)
  - `checkpoints/*.pt` (plain torch.save)
  - `accelerate_state/*/` (accelerate.save_state)
  - `skip — host doesn't save to a conventional location`
  - `Other (glob)`
- `skip` triggers fallback to AR-SAVE's legacy in-process torch-state capture.

## Question 8 — primary metric

- **header**: `Primary metric`
- **question**: `Which metric decides whether a run improved? (direction matters.)`
- **multiSelect**: `false`
- **options**: top metric candidates from Phase 1 (wandb summary / tensorboard scalar tags / grepped stdout regex anchors), labeled with direction:
  - `val/loss (min)`
  - `val/bpb (min)`
  - `eval/top1 (max)`
  - `train/loss (min)`
  - `Other (type metric=direction)`
- If the user picks `Other`, prompt for freeform input as `name=min` or `name=max` — parse accordingly. If direction is ambiguous, re-ask for direction only.

## Question 9 — hard constraints

- **header**: `Hard constraints`
- **question**: `Constraints that invalidate a run if violated. Pick any / none.`
- **multiSelect**: `true`
- **options**: `None` is the default; 2–3 resource-bound candidates if detected, plus free-text:
  - `None`
  - `peak_vram_mb <= 45000`
  - `tokens_per_sec >= 5000`
  - `wall_seconds <= 600`
  - `Other (name op value)`
- Parse each selected constraint into `{name, op, threshold}`. Operators: `<=`, `<`, `>=`, `>`. If `None` is selected alongside others, treat the others as authoritative and ignore `None`.

## Question 10 — mutation scope

- **header**: `Mutation scope`
- **question**: `Modules / classes train.py may monkey-patch. Advisory only — not enforced in code.`
- **multiSelect**: `true`
- **options**: dotted-path candidates from Phase 1, multi-select, with escape hatch. The paths should be namespaced under the host project's package — e.g. `myproject.training.losses.YourLoss`, not bare `training.losses.YourLoss`:
  - `myproject.training.losses.YourLoss`
  - `myproject.model.attn.YourBlock`
  - `myproject.model.projector.Projector`
  - `Other (comma-separated dotted paths)`
  - `(advisory — will skip)`
- If `(advisory — will skip)` is selected, omit `--mutation-scope` from `ar init`.

## Question 11 — default run duration

- **header**: `Run duration`
- **question**: `Default per-run time budget in seconds (overridable per autoresearch:run).`
- **multiSelect**: `false`
- **options**:
  - `300` (5 min — small models, quick ablations)
  - `600` (10 min — default)
  - `900` (15 min)
  - `1800` (30 min — heavy models, multi-GPU)
  - `Custom`

## Question 12 — runner

- **header**: `Runner`
- **question**: `How should ar launch training? Pick the inferred one or customize.`
- **multiSelect**: `false`
- **options**: Phase 1's inferred runner first, then the others, then Custom:
  - `accelerate launch --config_file configs/accelerate.yaml` (example; use the detected config file)
  - `torchrun --nproc-per-node N` (substitute the project's actual GPU count)
  - `python -m myproject.training.main` (common for Hydra apps)
  - `python`
  - `Custom`
- The exact string is stored and invoked verbatim by `ar`, so include `--config_file`, `--nproc-per-node`, and similar args.

## Question 13 — wandb project (conditional)

Skip if item 6 (metric backend) is not `wandb` or `auto`.

- **header**: `wandb project`
- **question**: `Which wandb project should runs log to?`
- **multiSelect**: `false`
- **options**: detected value from `wandb/run-*/files/config.yaml` (`_wandb.project`) pre-selected, plus `Other (type)`. Optional — empty accepted; the template falls back to `{pyproject.name}` then `<your-wandb-project>`.

## Direction-inference table

Use this when auto-labeling metric candidates in question 8. If a key matches multiple substrings, the first match wins.

| Substring (case-insensitive) | Direction |
|---|---|
| `loss`, `nll`, `ppl`, `perplex`, `bpb`, `bits_per`, `error`, `wer`, `cer`, `fid`, `lpips` | min |
| `acc`, `top1`, `top5`, `auc`, `f1`, `recall`, `precision`, `bleu`, `rouge`, `exact_match`, `em`, `map`, `iou`, `dice`, `psnr`, `ssim`, `reward`, `score`, `win_rate` | max |
| `vram`, `memory`, `mb`, `gb`, `wall`, `time`, `latency`, `seconds` | min (usually a constraint, not primary) |
| `throughput`, `tokens_per_sec`, `samples_per_sec`, `steps_per_sec` | max (usually a constraint, not primary) |

If none match, label the metric `(direction?)` in the option text and force the user to disambiguate via the free-text path.

## Ordering note

Ask items in a **single** `AskUserQuestion` call where possible. Claude Code's `AskUserQuestion` supports multiple `questions` per call and renders them as a compact form — respect this for UX. The only unavoidable multi-round case is when item 6 selects `tensorboard` or `custom` — the follow-up free-text captures for `--tb-events-glob` or `--metric-extract-code` each fire as a second AskUserQuestion after the main form submits.
