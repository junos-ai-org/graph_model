# CLAUDE.md — graph_model fork (junos-ai-org)

> Loaded automatically into every Claude Code session in this repo.
> **You are running autonomously on a remote pod. Read this first; then read the runbook (link below).**

## What this repo is

A **public fork** of [`DarioVajda/graph_model`](https://github.com/DarioVajda/graph_model) — the official code for **GTLM** ("Teaching LLMs to See Graphs," arXiv [2605.10247](https://arxiv.org/abs/2605.10247)).

- **License:** MIT (Dario's). `LICENSE` is intact and authoritative.
- **Author:** Dario Vajda (University of Ljubljana). Use + extension for research was granted on 2026-05-24.
- **Upstream:** `DarioVajda/graph_model`. Do **not** open PRs to upstream from this fork.

## Why we forked it

junos-ai-org is building [`graph-reasoning-llm`](https://github.com/junos-ai-org/graph-reasoning-llm) — research on native graph reasoning in LLMs.

Our **Phase 0** goal is to **reproduce GTLM's Family Tree experiment** as published, end-to-end, on our own infrastructure. The reproduction becomes the **untyped baseline** for later phases. **No novelty yet.**

## Hard rules (do not break)

- **All Phase 0 work happens on the branch `experiment/001-repro-family-tree`** in this fork. `main` stays identical to upstream Dario's `main`.
- **No new bias classes** and **no architecture restructuring** — Phase 0 *scope* rule. You may edit configs, write smoke tests, and modify dataloader logic as needed to reproduce. You may **not** add new `BaseBias` subclasses or restructure `src/models/`.
- **Only `our_tests/` (Family Tree)** as the primary experiment; fallback ladder in the runbook.
- **No PRs to upstream.**
- **Budget cap: $30** of pod compute for all of Phase 0. Hard-stop at $25 with $5 buffer.
- **Public visibility** is correct for Phase 0. Phase 1 visibility is a separate decision — not your concern.

## How the work is organized

You work in **two repos cloned side-by-side**:

```
~/graph_model              ← this fork; check out branch `experiment/001-repro-family-tree`
~/graph-reasoning-llm      ← research repo (runbook + experiment state, work on branch experiment-setup)
```

- **At session start:** verify `git -C ~/graph_model branch --show-current` returns `experiment/001-repro-family-tree`. If not, `git checkout experiment/001-repro-family-tree`.
- **The runbook** is at `~/graph-reasoning-llm/docs/reproduction_plan.md`. Read it once at session start; re-read after every stage.
- **The active experiment** is `~/graph-reasoning-llm/experiments/001-repro-gtlm-family-tree/`. Write `progress.md` and `decisions.md` there.
- **Code changes to the fork** (configs, smoke tests, dataloader tweaks) commit to the `experiment/001-repro-family-tree` branch directly. **State** (progress, decisions, logs, artifacts) goes in the research repo's experiment dir.
- **Commit and push after every substantive event** (success, failure, decision, blocked). So a fresh session can pick up.

## Tools available on the pod (verify in preflight)

- `gh`, `git` — authenticated, push access to both repos.
- `python` (≥ 3.10), `pip-tools`, `pip`.
- `huggingface-cli` — HF token loaded (Llama-3.2-1B access required, it's gated).
- `wandb` — API key loaded.
- NVIDIA driver + CUDA + a GPU (≥24 GB VRAM).
- `runpodctl` — for pod cost tracking.

## Autonomy + escalation

You execute the runbook **autonomously**. Escalate (write `BLOCKED-<reason>.md` in the experiment dir + commit + halt the session) only when:

- A failure persists after **2 retries** at the same stage.
- Cumulative pod cost approaches **$25** (with $5 buffer to the $30 cap).
- A stage exceeds its declared time cap (defined in the runbook).
- Preflight fails (pre-existing setup problem — not yours to fix).

When you halt, the last line of `progress.md` should be a timestamped `## HALTED — <reason>` entry, followed by a `git push`.

## Phase 1 visibility (not your problem)

Once Phase 1 work begins (adding `RelationBias`), we may move to a private branch or a separate private repo. **For Phase 0: public fork, public main, public commits.** Don't worry about it.

## If you only read one thing

→ `~/graph-reasoning-llm/docs/reproduction_plan.md`
