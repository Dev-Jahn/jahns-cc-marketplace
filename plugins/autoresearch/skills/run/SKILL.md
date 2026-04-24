---
name: autoresearch:run
description: This skill should be used when the user asks to "start the autoresearch loop", "kick off overnight iteration", "begin autonomous experiment runs", "run /autoresearch:run", "run the autoresearch expr <slug>", "continue the autoresearch loop", "resume autoresearch", "chain through follow-up experiments", or otherwise hand off an ML experiment to the autonomous runner. Drives the self-propelling train.py iteration loop on a configured `.autoresearch/{expr}/` experiment — one-line edit, `ar run`, read `result.json`, decide next edit, repeat — for hours or days until a termination condition fires. Context-minimized so thousands of iterations fit in a single session. Invoke immediately without asking clarifying questions beyond the structured interview; the skill itself is self-driving and must never stop mid-loop to ask the user "continue?" — Ctrl+C is the only authorized interrupt.
argument-hint: [expr-slug]
allowed-tools: Read, Bash, Edit, Grep, Glob, AskUserQuestion
---

# autoresearch:run

Drive the autonomous train.py iteration loop on a prepared `.autoresearch/{expr}/` experiment. The loop is self-propelling: one parameter or monkey-patch edit per iteration, one `ar run` bash invocation per iteration, one `result.json` read per iteration, then decide the next edit. Terminate only on the configured condition (primary-metric threshold, plateau, max runs, chain budget exhausted) or a Ctrl+C from the user.

The design invariant behind every rule below: **context cost per iteration must stay bounded so hundreds or thousands of iterations fit in one session.** Every rule in this skill exists to honor that invariant.

## Mental model in 30 seconds

- One `{expr}/train.py` is the only file to edit. Everything else is read-only.
- `ar run` handles: snapshot → launch → watchdog → wandb parse → verdict → advance/revert → termination eval → `result.json`.
- `best.json` + `runs/{best_id}/train.py` are the ground truth. `ar` auto-restores train.py on every non-advance disposition, so a crashing edit disappears — write the next edit against the restored baseline.
- Chain mode: when the expr terminates with chain budget remaining, design a follow-up expr, call `ar chain-init --rationale "..."`, re-enter non-interactively against the child slug.

## Preflight — mode detection first

Branch on the `AR_NONINTERACTIVE` environment variable **before** any AskUserQuestion call. The branch exists so chain-mode re-entry can reach the loop without stalling on an interactive prompt (which would deadlock because the user is asleep).

```bash
# Check the flag once, at skill entry.
if [ "${AR_NONINTERACTIVE:-0}" = "1" ]; then
    # Non-interactive: chain re-entry path.
    # - Child slug is in .autoresearch/.chain-session.json.current_slug
    # - Session settings already materialized in {child}/.ar-session.json by ar chain-init
    # - Go directly to loop step 1 against the child slug. Skip all AskUserQuestion.
    :
else
    # Interactive: user-initiated invocation. Run the full interview below.
    :
fi
```

### Non-interactive branch (chain re-entry)

When `AR_NONINTERACTIVE=1`:

1. Read child slug from `.autoresearch/.chain-session.json` (field `current_slug`).
2. Do **not** run any AskUserQuestion calls. The skill has already been handed a child expr whose `.ar-session.json` was populated by `ar chain-init` with inherited settings (duration, termination_conditions, hard_constraints, runner, resume_mode="continue", baseline implicitly acknowledged via `chain_decision.json`).
3. Jump directly to loop step 1 below against the child slug.
4. The `AR_NONINTERACTIVE` flag is scoped to the current re-entry only. It does not persist into any session file and has no effect on a later user-initiated invocation.

### Interactive branch (user-initiated)

When `AR_NONINTERACTIVE` is unset or `0`:

1. **Slug resolution:**
   - If positional arg given, use it. Confirm `.autoresearch/{slug}/` exists; error out if not.
   - If omitted, run `ls -t .autoresearch/ | grep -v '^\.'` to get directories ordered by mtime. Take the first. If two top candidates have mtimes within ~60 seconds (ambiguous), use AskUserQuestion to pick.
2. **Session interview** via AskUserQuestion. Ask in this order; skip a question only if its answer is already materialized in `.ar-session.json` and the user hasn't asked to change it:
   1. **Run duration** — default = value in `program.md`. Offer 300 / 600 / 900 / 1800 s + custom.
   2. **Termination conditions** (multi-select): primary threshold (ask for value), plateau(N) (ask for N), max runs (ask for N), unlimited. At least one must be selected unless chain mode is unlimited.
   3. **Baseline acknowledgment** — only if `best.json` does not exist. Single confirmation that the first run will be treated as the baseline; it establishes `best.json` regardless of metric value.
   4. **Resume mode** — only if `runs/` is non-empty. Options: "continue from current train.py" / "revert to best and continue" / "start fresh" (dangerous; requires confirmation; wipes `runs/` and `best.json`).
   5. **Chain mode**: disabled (default) / max N chains (ask for N, default 3) / unlimited.
3. Persist the answers to `.autoresearch/{expr}/.ar-session.json` (let `ar` write this — do not hand-write it).
4. Persist chain-scope state to `.autoresearch/.chain-session.json` (`chain_remaining`, empty `chain_trail: [<current_slug>]`).
5. Proceed to loop step 1.

## The loop protocol

Execute steps 1-7 on repeat until termination or Ctrl+C. Do not deviate. Do not stop to ask mid-loop — see rule "Never stop to ask" below.

### Step 1 — State query (once per iteration start)

```bash
uv run python .autoresearch/ar.py status --compact
```

Output is ≤ 10 lines: expr name, primary metric + best value, total runs + advance/revert counts, termination progress (e.g. `max_runs: 12/50`, `plateau: 3/5`), and whether the working `train.py` diverges from the last-best snapshot.

Read this output. That's the entire state. Do **not** re-read `best.json`, `results.tsv`, or prior `result.json` files to reconstruct state — `ar status` already aggregates them.

### Step 2 — Decide the next edit

Consult, in this order, only what is needed:
1. `program.md` — only if not already in session context. Contains goal, mutation scope, primary metric spec, hard constraints.
2. `ar status --last 5` — tail of `results.tsv` showing the last 5 runs' parameters and outcomes.
3. Previous best's train.py — only if the edit under consideration would benefit from diffing against the current best. Read the specific line range, not the whole file.

**Budget target: ≤ 2-3 Read tool calls per iteration after the first.** Reuse what is already in session context. Prefer `grep` / line-range `Read` over full-file `Read` on anything larger than ~200 lines. Do not re-read files seen earlier in the session unless `ar status` indicates they changed.

Form a hypothesis: "If I change X, the primary metric should move in direction D because Y." Short is fine. This becomes the `RUN_NOTE` in the next step.

### Step 3 — Edit `{expr}/train.py` only

Edit the one file. Parameter tweaks, monkey-patches, structural changes — all inside `{expr}/train.py`. Set the `RUN_NOTE = "..."` string near the top of train.py with a single-line rationale ≤ 200 chars; `ar` will sanitize and append it to `results.tsv`. See the "Write discipline" section for the full list of things that are off-limits.

### Step 4 — Execute

```bash
uv run python .autoresearch/ar.py run
```

That is the entire command. `ar run` owns: startup reconciliation, snapshot (copying current `train.py` to `runs/{run_id}/train.py`), `prepare.py` contract check, subprocess launch with process-tree-aware watchdog, wandb pointer binding, metric extraction + constraint eval, two-stage atomic `best.json` + `best_ckpt/` commit (on advance), train.py restore (on any non-advance verdict), `results.tsv` append, termination eval, `result.json` write.

**Do not run anything else here.** Do not check process status, do not run `wandb` commands, do not touch checkpoints manually, do not inspect `runs/` during the run. `ar` is the sole orchestrator.

### Step 5 — Read `result.json`

```bash
cat .autoresearch/{expr}/runs/{last_run_id}/result.json
```

Or use `Read` on the file. Small structured file; parse the top-level fields:

- `status` ∈ {`ok`, `crash`, `timeout`, `invalid`, `interrupted`, `unclean`}
- `verdict` ∈ {`advance`, `revert`, `invalid`, `crash`, `timeout`, `interrupted`}
- `valid`, `improved_over_best`, `primary.value`, `should_terminate`, `terminated_by`, `note`.

Interpretation table (quick reference):

| status | verdict | what happened | what to do next |
|---|---|---|---|
| ok | advance | run completed, valid, improved primary | `best.json` updated, checkpoint promoted, train.py unchanged. Continue to step 6. |
| ok | revert | run completed, valid, did not improve OR constraint violated | train.py already restored to last best. Continue to step 6. |
| invalid | revert | run completed but metric extraction failed | train.py restored. Consider whether the edit broke the data contract. Continue. |
| crash | crash | subprocess exit != 0, not timeout | train.py restored. One fix attempt allowed — see "Crash recovery". |
| timeout | timeout | watchdog SIGKILLed | train.py restored. Interpret as "this edit is too slow under current `--seconds`"; prefer a smaller change next. |
| interrupted | interrupted | Ctrl+C during run | Loop should exit; user is intervening. |
| unclean | crash | watchdog kill verification failed; orphan processes survived | Fatal for this session. Stop; tell the user a sentinel needs manual cleanup. Do not loop again. |

The unified revert rule: on any `verdict != advance`, `ar` has **already** reverted `{expr}/train.py` to `runs/{best_id}/train.py` (or to the original baseline if no best exists yet). The next iteration's edit therefore operates on a clean, known-good baseline.

### Step 6 — Termination + chain transition

If `should_terminate == false`: go to step 1.

If `should_terminate == true`:

- If chain mode is disabled, or `chain_remaining == 0`: exit cleanly. The expr is done. Print a one-line summary (e.g. `expr {expr-slug} terminated: plateau(5) — best val/loss=0.812 at r0037`) and stop.
- If chain mode is enabled and `chain_remaining > 0`: design the next expr and transition. See `references/chain-transition.md` for the full procedure. Core sketch:
  1. `uv run python .autoresearch/ar.py report --expr {current}` → capture markdown summary into context.
  2. Internally reason (no AskUserQuestion): what axis moved the primary metric? what's under-explored? why is this next direction the right one? Compose a one-paragraph `--rationale` explaining that reasoning — this string is persisted verbatim to `chain_decision.json` (read-only, `chmod 0444`) and is the post-hoc audit artifact. Rationale is required; empty rationale aborts `chain-init`.
  3. Call:
     ```bash
     uv run python .autoresearch/ar.py chain-init \
       --from-expr {current} \
       --new-slug {yymmdd}-{kebab-topic} \
       --parent-ckpt .autoresearch/{current}/best_ckpt \
       --goal "..." \
       --mutation-scope "module.A,module.B" \
       --primary-metric "..." \
       --primary-direction min|max \
       --rationale "..." \
       --runner inherit \
       --seconds inherit \
       --constraints inherit
     ```
  4. `ar chain-init` exports `AR_NONINTERACTIVE=1` and prints the new slug. Re-enter the loop non-interactively against the new slug — return to step 1 with the child expr.

### Step 7 — Never stop to ask

Do **not** prompt "Should I continue?", "Is this edit ok?", "Should I try X next?" mid-loop. Do not offer to pause. The whole point of the skill is unattended autonomy — the user is asleep, on another continent, or in a meeting. Ctrl+C is the only authorized interrupt. The interactive preflight questions (step "Interactive branch" above) are the only AskUserQuestion calls allowed in this entire skill.

Why this rule is strict: every "should I continue?" prompt wastes a night of compute. If a genuinely unrecoverable situation arises (unclean sentinel, repeated chain-init failures, disk full), exit and surface the reason in a final message — don't ask for permission to keep trying.

## Write discipline — hard rule

Writes are permitted only to `{expr}/train.py`. Everything else is off-limits during the loop.

Explicit forbidden list:
- Do not create new files anywhere under `.autoresearch/` or the host project.
- Do not edit `{expr}/prepare.py`. It is the data/metric contract; changing it silently breaks comparability across runs.
- Do not edit `{expr}/program.md`, `{expr}/best.json`, `{expr}/results.tsv`, `{expr}/runs/**`, `{expr}/best_ckpt/**`, `{expr}/batch_contract.json`, `.autoresearch/.chain-session.json`, `.autoresearch/{expr}/.ar-session.json`, or any file under `.autoresearch/` other than `{expr}/train.py`.
- Do not edit **anything** in the host project outside `.autoresearch/`. The project's source tree (e.g. `<your_project>/model/`, `<your_project>/training/`, `<your_project>/attn/`, or whatever the host uses), config files, pyproject.toml, CLAUDE.md — all read-only.
- Do not `git add`, `git commit`, `git reset`, or touch the working tree's git state. The experiment loop lives entirely outside git.
- Do not manually touch `runs/{run_id}/state.pt`, `best_ckpt/state.pt`, `best_ckpt/meta.json`, or any wandb artifact.

Why this invariant is hard: the whole revert/advance/chain machinery depends on one and only one file mutating per iteration. Anything outside train.py is immune to `ar`'s restore logic; a stray edit to `prepare.py` breaks the data contract for every subsequent run with no automatic recovery. Keeping the write surface to one file makes the loop's semantics tractable.

Runtime monkey-patching of host-project modules **from within train.py** is explicitly allowed — that is the designed mechanism for exploring structural changes without writing to host source. The fresh-subprocess boundary (`uv run` spawns a new interpreter per run) guarantees monkey-patches never leak across runs.

## Read discipline — soft nudge

Broader reads are permitted. Budget targets to keep context bounded:

- **Target ≤ 2-3 Read tool calls per iteration** after initial orientation in the first iteration.
- **Avoid re-reading files seen earlier in the session.** If in doubt, trust the session context you already have.
- **Prefer `grep` or line-range `Read`** over full-file `Read` on anything over ~200 lines. Use `Grep` with specific patterns; use `Read` with `offset`/`limit` for targeted chunks.
- **Do not read `run.log` in bulk.** It is raw stdout+stderr and can be hundreds of thousands of tokens. If a stacktrace is needed, use `ar tail --run {id} --lines 80` (see crash recovery below).
- **Do read, as needed:** `program.md` (once per session, typically), `prepare.py` (only to understand the metric contract — read once and remember), `runs/{best_id}/train.py` (targeted line ranges when diffing against current edits), host-project source files (sparingly, for comprehension of the API being patched).

If a specific read keeps recurring (e.g. re-checking the same host-project file every 5 iterations), stop — the cost is compounding across the night. Internalize it once.

## Crash recovery

When `status == "crash"`:

1. Fetch the stacktrace region only:
   ```bash
   uv run python .autoresearch/ar.py tail --run {run_id} --lines 80
   ```
   This returns the last ~80 lines of `run.log`, filtered toward the Python traceback if one exists. Hard-capped — does not return the whole log.
2. Read the stacktrace. Identify the root cause.
3. One fix attempt allowed. Because `ar` has already restored `train.py` to the last-best snapshot, the fix is applied on top of the known-good baseline — not on top of the crashing edit. Re-apply the intended structural change with the identified bug corrected, set `RUN_NOTE` to reflect the fix ("retry: lr=3e-4 with correct dtype cast"), and run.
4. If it crashes again: treat this whole direction as a discarded hypothesis. Move on to a different edit. Do not attempt a third fix on the same bug — the cost-benefit has inverted.

Note: `ar`'s auto-restore means the `Edit` tool on `train.py` between iterations is always editing the last-best baseline, not a partially-applied crashing change. This keeps edits clean.

## Termination conditions — operational meaning

- `primary_threshold`: primary metric reached the configured threshold (e.g. `val/loss <= 0.80`).
- `plateau(N)`: the last N runs did not improve over the current best.
- `max_runs`: total runs (including crashes/timeouts) reached the configured max.
- `chain_budget`: top-level chain remaining count reached 0.
- `manual`: Ctrl+C or an explicit `interrupted` status on the last run.

`terminated_by` in `result.json` surfaces which of these fired. Use it to phrase the final summary when exiting (or to shape the chain-decision rationale when transitioning).

## Iteration budget discipline — summary

One iteration should cost, in bash:
1. `ar status --compact` (sometimes; skip if nothing changed on the previous iteration)
2. `ar run`
3. `cat result.json` (or an equivalent `Read`)

Steady-state target: **≤ 2 bash invocations per iteration** and **≤ 2-3 `Read` calls per iteration**. Crash iterations add one `ar tail`. Chain transitions add one `ar report` and one `ar chain-init`. Anything beyond that is context leakage — diagnose it.

## One concrete iteration example

Context: `{expr-slug}` expr, primary = `val/loss` (min), current best = 0.851 at r0037.

```
# Step 1 — state
$ uv run python .autoresearch/ar.py status --compact
expr: {expr-slug}
primary: val/loss (min) — best: 0.851 @ r0037
runs: 42 total (18 advance / 21 revert / 3 crash)
termination: plateau 2/5, max_runs 42/200
train.py: in-sync with best
```

```
# Step 2 — decide
# Session already has program.md. Check last 5 runs briefly.
$ uv run python .autoresearch/ar.py status --last 5
r0038 revert  val/loss=0.868  note: "weight_decay 0.1 → 0.2"
r0039 advance val/loss=0.851  note: "lr 3e-4 → 2.5e-4, beta2 0.98"
r0040 revert  val/loss=0.859  note: "add dropout 0.1 to projector"
r0041 revert  val/loss=0.855  note: "grad clip 1.0 → 0.5"
r0042 revert  val/loss=0.853  note: "warmup 500 → 1000"
# Hypothesis: the lr change at r0039 moved the needle; try a smaller lr still.
```

```python
# Step 3 — edit {expr}/train.py (using Edit tool, single line change)
# Old: LEARNING_RATE = 2.5e-4
# New: LEARNING_RATE = 2e-4
# Old: RUN_NOTE = "warmup 500 -> 1000"
# New: RUN_NOTE = "lr 2.5e-4 -> 2e-4 (extend r0039 direction)"
```

```
# Step 4 — execute
$ uv run python .autoresearch/ar.py run
[r0043] launched: python train.py  |  seconds=600
[r0043] wandb run: xyz789  |  val/loss=0.844  peak_vram=42800
[r0043] advance: val/loss 0.851 → 0.844  (best updated)
```

```
# Step 5 — read result
$ cat .autoresearch/{expr-slug}/runs/r0043/result.json
# status: ok, verdict: advance, primary.value: 0.844, should_terminate: false
```

```
# Step 6 — not terminating, go to step 1 with updated best.
```

Total context cost for this iteration: 3 bash invocations, 0 additional `Read` calls (used `ar status` output, not file reads). Next iteration will probably skip the `ar status --last 5` call since nothing has changed since it was just printed.

## Additional Resources

### Reference Files

- **`references/loop-protocol.md`** — Step-by-step operational detail for each loop phase, edge cases (first iteration, resume after interrupt, empty `runs/`), and worked examples of `result.json` interpretation.
- **`references/chain-transition.md`** — Full chain-transition procedure: what `ar report` contains, how to compose the rationale, the exact `ar chain-init` flags and inheritance semantics, handling of `chain-init` failure and retry, and non-interactive re-entry mechanics.
- **`references/write-discipline.md`** — Exhaustive list of files under `.autoresearch/` and the host project with their permitted operations (read / edit-train.py-only / fully-forbidden), plus the rationale for each boundary.

Consult these when a specific phase of the loop needs deeper detail than the inline prose provides. For steady-state operation, this SKILL.md body is sufficient.
