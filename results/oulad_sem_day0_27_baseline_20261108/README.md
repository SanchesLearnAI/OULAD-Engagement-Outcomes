# Day 0–27 pipeline baseline (archived 2026-11-08)

This folder preserves the tracked deliverables of the `oulad-sem/` day-0–27
design that the pipeline ran before the research question moved to the TMA
two-week study (see `research/TMA_PRIMARY_PROTOCOL.md`).

**Status: historical baseline.** It answers "within similar day-0–27 click
volume, which W1/W2 patterns accompany Fail?" — the original proposal window.
The TMA study answers a different question with a different anchor, so the two
sets of numbers must not be ranked against each other.

## Provenance

| Item | Value |
|---|---|
| Source commit | `origin/oliver` @ `e7cd98d` (P1–P5 applied, Fail = 1) |
| Full run id | `run-20261107T173321Z-4deefb` |
| Simulation run id | `sim-20261107T172956Z-d470d3` |
| Config SHA-256 | `6dd89b411646871c0e3623a4fc7d22aac983cf795f15fb3fca2de7bf07ae0043` |
| Validation | 0 errors / 0 warnings (554 events, S00–S12 ok) |
| Report | main text 8 pages, 16 pages total |
| Tests | 13 passed |

Files here are verbatim copies of the tracked `oulad-sem/outputs/tables/*.csv`
at that commit, plus `Report.pdf` and the run's `STATUS.json`. Figures are not
duplicated here: the sealed package
(`release/oliver-fixed/Group_09_Project.zip`, SHA-256 in that folder's
`SEAL.md`) already contains the complete `tables/` and `figures/` set from
exactly this run, and TMA figures now occupy `oulad-sem/outputs/figures/`.

## What was in it

- T1 cohort 22,437 enrolments; Fail 7,052 (0.314); no-skill PR-AUC 0.323.
- Model ladder M0–M5 on grouped, temporal and leave-one-module-out splits;
  M4 contains assessment scores and is an upper bound only.
- CEM matching with the treated group kept as Pass/Distinction, so the W2
  differences keep their original sign.
- Row-level split control deliberately leaks 310 students to prove the grouped
  split does not.

## Why it is kept

1. The report still cites it as the original-window baseline.
2. S12 can regenerate those numbers by restoring the archived config; keeping
   the tables makes the old numbers auditable without rerunning 15 minutes of
   computation.
3. Reviewers asked which numbers came from which design; this folder answers
   that directly.

Do not add student-level files here. `outputs/` and `runs/` stay git-ignored.
