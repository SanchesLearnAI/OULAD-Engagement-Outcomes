"""Structural validator: schema, sequence, gates, artifact hashes and handoffs.

The stage-specific hard checks are listed in ``HARD_CHECKS`` so the TMA stages
can add their own without touching the validator body.
"""
from __future__ import annotations

import json
from collections import defaultdict

import jsonschema

from .config import contracts_dir, project_root, sha256_file
from .logging_utils import STAGES
from .reporting import citation_scan

# Checks that must be ``ok`` on every run.  Anything a rehearsal may legally
# downgrade to ``warn`` is deliberately absent here.
HARD_CHECKS = {
    "S04": ("no_forbidden_columns", "no_future_submission", "score_not_in_strict_features"),
    "S05": ("main_train_test_student_overlap", "main_val_test_student_overlap",
            "group_fold_disjoint", "temporal_student_overlap", "lomo_student_overlap"),
    "S06": ("pairs_have_no_reused_record", "zero_only_matches_zero",
            "primary_family_complete"),
    "S08": ("binary_label_fail",),
}


def validate_run(run_id: str, require_complete: bool = True):
    root = project_root()
    run_dir = root / "runs" / run_id
    errors, warnings = [], []
    if not run_dir.exists():
        return {"errors": [f"run directory missing: {run_dir}"], "warnings": [],
                "stages": {}}
    root_status_path = root / "STATUS.json"
    root_status = (json.loads(root_status_path.read_text())
                   if root_status_path.exists() else {})
    is_current = root_status.get("run_id") == run_id

    schema = json.loads((contracts_dir() / "log_event.schema.json").read_text())
    lines = (run_dir / "run.jsonl").read_text().splitlines()
    events = []
    last_seq = 0
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except Exception as ex:
            errors.append(f"line {i}: invalid JSON: {ex}")
            continue
        try:
            jsonschema.validate(e, schema)
        except jsonschema.ValidationError as ex:
            errors.append(f"line {i}: schema violation: {ex.message}")
        if e.get("run_id") != run_id:
            errors.append(f"line {i}: run_id mismatch")
        if e.get("seq", 0) <= last_seq:
            errors.append(f"line {i}: seq not increasing ({e.get('seq')} <= {last_seq})")
        last_seq = e.get("seq", last_seq)
        events.append(e)

    by_stage = defaultdict(list)
    last_artifact = {}
    for e in events:
        by_stage[e.get("stage")].append(e)
        if e.get("event") == "artifact":
            for o in e.get("outputs", []):
                last_artifact[o["path"]] = o
    stage_status = {}
    for s in STAGES:
        ev = by_stage.get(s, [])
        starts = [e for e in ev if e["event"] == "stage_start"]
        ends = [e for e in ev if e["event"] == "stage_end"]
        gates = [e for e in ev if e["event"] in ("gate_pass", "gate_fail")]
        required = require_complete or s != "S12"
        if required:
            if not starts:
                errors.append(f"{s}: missing stage_start")
            if not ends:
                errors.append(f"{s}: missing stage_end")
            if not gates:
                errors.append(f"{s}: missing gate_pass/gate_fail")
        stage_status[s] = (ends[-1]["status"] if ends else None)
        sj = run_dir / "stages" / f"{s}.stage.json"
        ho = run_dir / "handoff" / f"{s}.md"
        if required:
            if not sj.exists():
                errors.append(f"{s}: missing {sj.name}")
            if not ho.exists():
                errors.append(f"{s}: missing handoff {ho.name}")
        if gates and not (s == "S12" and not require_complete):
            g = gates[-1]
            if g["event"] == "gate_fail" or g["status"] == "fail":
                errors.append(f"{s}: gate failed: {g.get('message')}")

    for path, o in last_artifact.items():
        if not is_current and not path.startswith(f"runs/{run_id}/"):
            continue
        p = root / path
        if not p.exists():
            errors.append(f"missing artifact {path}")
            continue
        if o.get("sha256"):
            actual = "sha256:" + sha256_file(p)
            if actual != o["sha256"]:
                errors.append(f"hash mismatch {path}")
    if not is_current:
        ma_path = run_dir / "manifest_all.json"
        if not ma_path.exists():
            errors.append("manifest_all.json missing for non-current run")
        else:
            ma = json.loads(ma_path.read_text())
            if ma.get("run_id") != run_id:
                errors.append("manifest_all.json run_id mismatch")
            if not ma.get("artifacts"):
                errors.append("manifest_all.json has no artifacts")

    for stage, names in HARD_CHECKS.items():
        path = run_dir / "stages" / f"{stage}.stage.json"
        if not path.exists():
            continue
        checks = {c["name"]: c["status"] for c in json.loads(path.read_text())["checks"]}
        for name in names:
            if name in checks and checks[name] == "fail":
                errors.append(f"{stage} hard check failed: {name}={checks[name]}")

    spath = root / "STATUS.json" if is_current else run_dir / "STATUS.final.json"
    if spath.exists():
        st = json.loads(spath.read_text())
        if st.get("run_id") != run_id:
            errors.append(f"{spath.name} run_id mismatch")
        if require_complete:
            if st.get("run_status") != "complete":
                errors.append(f"STATUS.run_status={st.get('run_status')} (expected complete)")
            if st.get("next_stage") is not None:
                errors.append(f"STATUS.next_stage={st.get('next_stage')} (expected null)")
            for s in STAGES:
                if st.get("stages", {}).get(s, {}).get("status") not in ("pass", "warn"):
                    errors.append(f"STATUS.{s} not pass/warn")
    else:
        errors.append(f"{spath.name} missing")

    if is_current:
        for rel in ("REPORT.md", "Report.tex"):
            p = root / rel
            if p.exists():
                errs, warns = citation_scan(p.read_text())
                errors += [f"{rel}: {x}" for x in errs]
                warnings += [f"{rel}: {x}" for x in warns]
    return {"errors": errors, "warnings": warnings, "stages": stage_status,
            "n_events": len(events)}
