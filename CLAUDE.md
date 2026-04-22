# CLAUDE.md — behavioral guidelines

Guidelines to reduce common LLM coding mistakes on this repo.

**Tradeoff:** these bias toward caution over speed. For trivial tasks use judgment.

## 0. Project context

This repository is a very high stakes production code.

**Stakes.** *The lives of many humans rely on this project.* This is the framing, not a metaphor — a working instruction to calibrate rigor. It means: no confident guesses, no silent fallbacks, no "good enough" when the measurement is cheap, no plausible-looking code without verification. Research-grade rigor is the bar. The best Quality posible is the expected. No errors is our moto.

Concretely:
- Take the work seriously. No toy baselines dressed up as results, no "good enough" when a proper measurement is cheap.
- Every experimental claim needs a baseline, a metric, and a fixed seed. Argument-aware retrieval vs. chunk retrieval is the central comparison — do not accidentally make it unfair.
- Cite the source when you use a dataset, metric, or method from the literature. Record the exact variant (model checkpoint, hyperparameters, split).
- Keep runs reproducible: pinned versions, deterministic splits, logged configs. A result that can't be re-run isn't a result.
- When a shortcut is the right call (e.g. a throwaway sanity check), say so out loud and mark it — don't let it slip into reported numbers.
- Default to writing for a future reader who wants to understand *why* a design choice was made, not just *what* the code does.

If a task conflicts with these, surface the conflict rather than silently cutting corners.

## 1. Think Before Coding

*Don't assume. Don't hide confusion. Surface tradeoffs.*

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

*Minimum code that solves the problem. Nothing speculative.*

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

*Touch only what you must. Clean up only your own mess.*

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

*Define success criteria. Loop until verified.*

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Project-specific: no silent fallbacks

Any fallback path in this codebase MUST emit a `[WARN]` log describing:
- what was expected,
- what actually happened,
- what fallback was chosen.

Silent `except Exception: pass`, defaulted config values without announcing them, and quietly disabled features are all bugs. If a reasonable fallback exists, take it AND log it. If no reasonable fallback exists, `raise`.

Log format: `print(f"[WARN] {context}: expected={expected}, got={actual}, fallback={chosen}", flush=True)` or equivalent.

## 6. Verify before using

*If unsure, research online first. Don't guess from training memory.*

Before using a library API, model checkpoint, config flag, or tool invocation you are not certain about:

- Use WebSearch or WebFetch to consult current docs.
- Check the version we are actually installing before writing code against it.
- Prefer minimal, verified syntax over a clever pattern you half-remember.
- Skim a library's README / quickstart before writing against it — not after a bug.

Applies especially to fast-moving 2024–2026 tooling: vLLM serving args, Qdrant client API, Kuzu Cypher syntax, HuggingFace dataset/model loaders, Hydra config composition, PyTorch 2.x training patterns, and anything in the RAG / argument-mining ecosystem. RAG tooling APIs move faster than training data — a 2024 pattern may not match 2026 reality.

## 7. Cross-validate with parallel agents — selectively

*Only cross-check where the failure mode is being wrong about the world, not where it's being wrong about code.*

**DO spawn a validator when:**

- **Citing a paper or external source** — any claim sourced from a paper, benchmark, or external doc (numbers, method names, design decisions, "X paper says Y"). Hallucinated citations are the single most expensive error to ship.
- **Implementing math from a paper or library** — equations, loss functions, geometric transforms, RANSAC variants, any numerical routine where an off-by-one or wrong sign survives unit tests but produces garbage at scale.
- **Theorizing or proposing an architecture** — when the recommendation depends on claims that aren't directly in the code (e.g. "X library supports Y feature", "Z model has N parameters", "A is faster than B"). Validator confirms the premise, not the syntax.
- **Load-bearing research code** — training loops, eval harnesses, statistical analysis. Two independent reviewers for anything whose numbers will end up in a report or paper.

**DO NOT spawn a validator for:**

- **Deletions** — removing dead code, deleting files, cleaning up unused imports. If it compiles without it, it's gone; nothing to validate.
- **Mechanical edits** — renames, moves, formatting, scaffolding, config changes.
- **Direct observation** — reading files, running commands, reporting what the code plainly says.
- **Obvious bug fixes** — where the fix is a one-line correction of a clearly-wrong expression.
- **Pure refactors that preserve behavior** — if the tests pass before and after, you don't need a second opinion.

When in doubt, ask: "Could I be wrong about *the world* here (a paper, a library, a benchmark number), or just about *the code*?" Validators catch the first. Tests catch the second.

## 8. Commits

*Commit when a logical unit is done — don't wait to be asked.*

- After a coherent piece of work lands (tests passing, scope clean), stage and commit.
- Group changes by concern: scaffolding, a feature, a bug fix, a refactor — each gets its own commit.
- **Never mention Claude, Claude Code, AI collaboration, Anthropic, or Co-Authored-By attribution in any committed artifact — commit messages, PR descriptions, READMEs, docs, or code comments.** This is a hard rule. Commit messages describe the change, not the author; READMEs describe the project, not how it was built.
- Standard message style: short imperative summary line (≤72 chars), blank line, optional body explaining *why*. Match existing repo style.
- Do NOT push to the remote; commits stay local unless the user asks.
- Do NOT `--amend` previously-pushed commits. Create new commits on top.

## 9. Push, experiment logs, periodic status reports

*Commits live on origin, not just locally. Findings and plan revisions persist as dated artifacts.*

- **Push after every commit batch.** When you finish committing one or more related commits, `git push origin <branch>`. Never force-push. Never push to a branch someone else is actively working on without coordinating.
- **Reports live in `docs/report/`** (singular) with the existing numbered-topic convention: `docs/report/NN_short-dash-slug/<file>.md`, where `NN` is the next sequential two-digit number (check `ls docs/report/ | sort -n | tail -1`) and `<file>.md` is one of `findings.md`, `status-report.md`, `plan.md`, or `postmortem.md` depending on purpose. Do NOT create `docs/experiments/` or `docs/reports/` (plural) — both are deprecated; the whole project writes into `docs/report/`.
- **Write a report for every meaningful run, benchmark, evaluation, or architectural decision.** Cover: what you ran and why, what you measured, what surprised you, cross-validation trail (agents spawned, blockers raised, how addressed), files touched, next steps ranked. Match the style of recent entries (see `docs/report/41_pose-graph-gtsam-gnc/findings.md` for a template). Lead with `# Report NN — <Title>` and a TL;DR table so the next reader can triage in 30 seconds. Don't assume you'll remember; assume the next session starts cold.
- **Status reports on cadence.** Every ~4 commits since the last report, add a new numbered folder with `status-report.md` summarizing: what shipped, test count, outstanding items, any revised plan (as a sibling `plan.md`). When in doubt whether a report is due, write one.
- `docs/` is gitignored — reports stay local. That's intentional; they're working notes, not release artifacts. Recreate the folder if it gets removed.

## 10. Model selection

*Use the smallest model that fits the task.*

- For small, mechanical, or low-stakes tasks (trivial edits, file reads, lookups, scripted refactors, answering short factual questions), use **Claude Sonnet** rather than Claude Opus.
- Reserve **Claude Opus** for load-bearing work: architectural decisions, non-trivial implementations, multi-step reasoning, code review of load-bearing pieces, and anything covered by sections 0 and 7.
- When unsure, start with Sonnet; escalate to Opus if the task turns out to need deeper reasoning.

---

*These guidelines are working if:* fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
