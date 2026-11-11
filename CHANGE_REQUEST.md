
---

# CR-2026-11-11 — Restore the pre-deadline and relative-gap analyses

**Date:** 2026-11-11

## Reason

The integration kept only the post-deadline half of the TMA two-week design and
dropped two feature families that the source branch had: the pre-deadline
click-burst features and the signed relative pre/post click gap. Both were
exploratory on the TMA branch, but dropping them silently also dropped the
evidence that the pre-window is unbalanced and that the last-day click
concentration differs by outcome.

## What changes

- `src/tma/daily.py` adds `burst_features()` and `relative_click_gap()`, plus the
  derived `both_zero_clicks` flag. `build_daily_features()` attaches them when the
  pre window is exactly 14 days.
- S04 lists the new columns in `t04_feature_dictionary.csv`.
- S06 reports the restored indicators in `t06_effects.csv`.
- S11 gains two exploratory blocks behind `exploration:` flags:
  - `volume_clusters` (C5): outcome-blind post-click-volume clusters and
    within-cluster paired effects;
  - `relative_gap` (C6): a 17-feature arm that adds the gap to the frozen M2.
- `reporting.py` adds a pre-deadline table to the matched-comparison section.
- No primary indicator, primary model, split, window or matching rule changes.

## Affected stages

S04, S06, S11 (new exploratory steps) and S12 (report table).

## Frozen config

- Previous SHA-256: `f0830967e3afb70a3ed5f99fd41be415a926d8fbf5a517bd4a9c194a68cc1bb9`
- **New SHA-256:** `2bc4d5928c655f891121cbea2ca9062794f8eff07c54abd6bec1797dbbb8f040`

## Run impact

New run ids required. The primary tables must be identical to the previous run
because the new columns are additions that the frozen ladder does not read; the
verification script `verify_restored_parity.py` asserts that and the gap
distribution against the TMA branch's published table.
