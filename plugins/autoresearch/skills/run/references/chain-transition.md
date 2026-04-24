# Chain Transition — Full Procedure

This reference expands loop step 6's "chain transition" branch. When an expr terminates with `chain_remaining > 0`, the loop does **not** exit — it designs a follow-up expr, materializes it via `ar chain-init`, and re-enters non-interactively against the child slug.

## Preconditions for chain transition

A transition is attempted only when:
- `result.json.should_terminate == true` on the last run.
- `.autoresearch/.chain-session.json.chain_remaining > 0`.
- The current expr has a `best_ckpt/` (non-empty). Without a parent ckpt, chain has nothing to propagate; fall back to exit.
- No `.autoresearch/{cur}/.ar-unclean` sentinel outstanding. `ar chain-init` refuses in that case.

## Step 6a — Generate parent report

```bash
uv run python .autoresearch/ar.py report --expr {current}
```

Output: a markdown document (hundreds of tokens) containing:
- Parent expr's goal + mutation scope from program.md.
- All parameters that varied across runs + their metric impact (from results.tsv).
- The best run's configuration (parameters, monkey-patches, RUN_NOTE).
- The top-3 and bottom-3 runs with brief notes.
- Chain lineage (if this expr is itself a chain child, its own `chain_decision.json` rationale).

Read this into context. `ar chain-init` will also hash it (sha256) and store the hash in the child's `chain_decision.json` so retroactive tampering of the parent report is detectable.

## Step 6b — Internal reasoning (no user prompt)

Before composing the rationale or calling `chain-init`, think through:

1. **What axis actually moved the primary metric in the parent?** Scan the report for the parameter whose changes correlate with the biggest metric deltas. E.g., "loss weight λ swept from 0.1 to 1.0; λ=0.3 produced the best val/loss."

2. **What's under-explored?** From program.md's mutation scope, list modules/functions that were declared in-scope but never meaningfully touched. E.g., program.md listed `training.losses.UnifiedLoss`, `model.projector.Projector`, `model.na_gla_block.NAGLABlock` but all 80 runs only varied λ inside `UnifiedLoss`. The other two modules are under-explored.

3. **Is the parent's primary metric the right one for the follow-up?** The child expr may choose a **different** primary metric — e.g. parent optimized val/loss (training-stability focus); child now has a warm projector and should optimize `eval/ucf101_top1` (downstream task focus). This is explicitly allowed.

4. **Why does this direction follow from the parent's results?** Not "here's another random thing to try" — articulate the causal link. E.g.: "λ=0.3 produced stable temporal attention; with temporal attn warm, projector depth is now the binding constraint on downstream accuracy."

5. **What's the new mutation scope?** Usually narrower than the parent's — focus on the under-explored axes identified above.

## Step 6c — Compose the rationale

Write one paragraph (~4-8 sentences), directly to the point. Structure:

1. **What the parent established** (1 sentence).
2. **What that makes possible** (1-2 sentences).
3. **The child's hypothesis** (1-2 sentences).
4. **Why this child direction is the right follow-up** (1-2 sentences).

Example:

> Parent expr 260424-loss-ablation established that UnifiedLoss with λ=0.3 produces stable val/loss (0.821, down from 0.92 baseline) across the last 40 runs without exceeding the 45 GB VRAM constraint. With temporal attention now thermally stable, the binding constraint on downstream UCF101 accuracy shifts from training-time loss weighting to projector representational capacity — in the parent, projector.depth was held at 2 while the rest of the stack varied. The child expr will fix the best-λ configuration and vary projector depth ∈ {2, 3, 4} + hidden-dim ∈ {384, 512, 768}, targeting eval/ucf101_top1 (max) as the new primary metric. This direction is the right follow-up because (a) parent's best config is the correct starting point for downstream eval rather than a fresh init, (b) projector is declared in-scope in program.md but was never varied, and (c) UCF101 top-1 is the eventual release metric, so optimizing it directly now that the loss curve is settled is more informative than further loss-weighting sweeps.

This rationale is persisted verbatim to the child's `chain_decision.json` under the `"rationale"` field. Keep it specific and causal — it is the post-hoc audit artifact.

## Step 6d — Call `ar chain-init`

```bash
uv run python .autoresearch/ar.py chain-init \
  --from-expr 260424-loss-ablation \
  --new-slug 260425-projector-depth \
  --parent-ckpt .autoresearch/260424-loss-ablation/best_ckpt \
  --goal "Explore projector depth and hidden dim now that temporal attention is warm; target downstream UCF101 top-1." \
  --mutation-scope "model.projector.Projector,training.losses.UnifiedLoss" \
  --primary-metric "eval/ucf101_top1" \
  --primary-direction max \
  --rationale "Parent expr 260424-loss-ablation established UnifiedLoss λ=0.3 produces stable val/loss=0.821 ... [full paragraph] ..." \
  --runner inherit \
  --seconds inherit \
  --constraints inherit
```

Flag semantics:
- `--from-expr` — parent slug. Must exist with a `best.json` and `best_ckpt/`.
- `--new-slug` — YYMMDD-kebab. Must not exist.
- `--parent-ckpt` — absolute or relative path to `{parent}/best_ckpt`. `chain-init` verifies the directory is non-empty and that `state.pt` is readable.
- `--goal` / `--mutation-scope` — render into the child's `program.md`.
- `--primary-metric` / `--primary-direction` — render into child's `program.md`; the child may choose a different primary than the parent.
- `--rationale` — **required, non-empty**. Empty or whitespace-only rationale causes `chain-init` to abort.
- `--runner inherit` / `--seconds inherit` / `--constraints inherit` — copy the parent's `.ar-session.json` settings into the child's `.ar-session.json`. Also acceptable as explicit values (e.g. `--seconds 900`), but `inherit` is the chain-mode default.

What `chain-init` does on success:
1. Runs `ar init` internally to scaffold `.autoresearch/{new-slug}/` (program.md, prepare.py, train.py from templates with parent-ckpt declaration wired into program.md's `parent_ckpt` field; prepare.resume_ckpt_path() reads that field).
2. Copies parent's `.ar-session.json` → child's `.ar-session.json`.
3. Computes sha256 of the `ar report` output captured at transition time.
4. Writes `.autoresearch/{new-slug}/chain_decision.json` (schema below), then `chmod 0444` it.
5. Updates `.autoresearch/.chain-session.json`: decrements `chain_remaining`, appends `{new-slug}` to `chain_trail`.
6. Exports `AR_NONINTERACTIVE=1` into the environment for the re-entering agent.
7. Writes `.chain-session.json.current_slug = "{new-slug}"` so the skill's non-interactive preflight can discover it.
8. Prints the new slug on stdout and exits 0.

### `chain_decision.json` schema

```json
{
  "parent_expr": "260424-loss-ablation",
  "parent_primary": {"name": "val/loss", "direction": "min", "best_value": 0.821},
  "child_primary": {"name": "eval/ucf101_top1", "direction": "max"},
  "child_goal": "Explore projector depth ...",
  "child_mutation_scope": ["model.projector.Projector", "training.losses.UnifiedLoss"],
  "rationale": "<the full paragraph composed in step 6c>",
  "parent_report_sha256": "<sha256 hex>",
  "created_at": "2026-04-25T03:47:12Z",
  "chain_position": 2
}
```

`chain_position` is 0-indexed. Position 0 is the user-initiated root expr (has no chain_decision.json). The first chained child is position 1, the next is position 2, etc.

## Step 6e — Re-enter the loop

After `chain-init` succeeds:
1. `AR_NONINTERACTIVE=1` is now in the environment.
2. Re-invoke the skill's logic (conceptually: return to "Preflight — mode detection first" in SKILL.md).
3. Non-interactive branch triggers: read child slug from `.chain-session.json.current_slug`, skip all AskUserQuestion, go straight to step 1 against the child.
4. The child's first `ar run` loads parent-ckpt via `prepare.resume_ckpt_path()` in the frozen `AR-RESUME` block of train.py. A successful resume is visible as step-count > 1 and loss/metric continuity on the first training step.

## `chain-init` failure

If `chain-init` exits non-zero:
- Read stderr for the reason. Common causes: parent ckpt missing, disk space insufficient (needs ≥ 2× ckpt size free), new-slug already exists, rationale empty.
- **One retry allowed.** Before retrying, delete any partial `.autoresearch/{new-slug}/chain_decision.json` that may have been written (step 4 above happens atomically but an even earlier failure could leave partial scaffolding). Regenerate a fresh rationale if relevant (e.g., if the previous attempt hashed a stale report).
- If the retry also fails: chain ends. Exit the session cleanly with a summary of why. Do not attempt further chains.

## Chain budget exhaustion

When `chain_remaining` reaches 0 after a successful `chain-init`, the newly materialized child expr still runs to its own termination. Chain exhaustion stops further chaining **after** the current child completes, not before it starts.

When the current child terminates and `chain_remaining == 0`, step 6 in the child's loop takes the "disabled or exhausted" branch and exits cleanly.

## Sanity checks before committing to a chain transition

Before calling `chain-init`, verify:
- Parent `best_ckpt/state.pt` exists and is not zero-byte. `ls -la .autoresearch/{cur}/best_ckpt/` should show `state.pt` at multi-MB/GB size.
- No `.ar-unclean` sentinel in the parent.
- Disk free ≥ 2× ckpt size (`df -h .`).
- The rationale paragraph is concrete, not a placeholder ("tbd" / "explore more" are not acceptable — `chain-init` won't check semantics but the audit trail is worthless if the rationale is vacuous).

If any precondition fails, exit the session rather than forcing a broken transition. A failed chain is strictly worse than a clean termination.
