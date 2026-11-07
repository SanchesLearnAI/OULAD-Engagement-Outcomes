#!/usr/bin/env python3
"""Execute exactly one stage, enforcing the Runbook's gate order.

Usage:
  python scripts/run_stage.py --stage S00 --mode simulation --run-id sim-... \
      --config config/config.yaml
"""
from __future__ import annotations

import argparse
import importlib
import os
import random
import re
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["OULAD_ROOT"] = str(ROOT)

import numpy as np  # noqa: E402

from src.config import load_config  # noqa: E402
from src.logging_utils import (STAGES, RunContext, init_status, load_status,  # noqa: E402
                               utc_now)

RUN_RE = re.compile(r"^(run|sim)-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{6}$")
MODULES = {
    "S00": "stages.s00_acquire",
    "S01": "stages.s01_raw_audit",
    "S02": "stages.s02_master",
    "S03": "stages.s03_eda",
    "S04": "stages.s04_features",
    "S05": "stages.s05_splits",
    "S06": "stages.s06_matching",
    "S07": "stages.s07_clustering",
    "S08": "stages.s08_models",
    "S09": "stages.s09_sequential",
    "S10": "stages.s10_survival",
    "S11": "stages.s11_robustness",
    "S12": "stages.s12_package",
}


class Tee:
    def __init__(self, stream, path):
        self.stream = stream
        self.f = open(path, "a", encoding="utf-8")

    def write(self, s):
        self.stream.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self):
        self.stream.flush()
        self.f.flush()

    def close(self):
        self.flush()
        self.f.close()


def default_run_id(mode: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = "sim" if mode == "simulation" else "run"
    return f"{prefix}-{stamp}-{secrets.token_hex(3)}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--mode", required=True, choices=["simulation", "full"])
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    # The CLI mode is authoritative; config.mode is the template default.
    cfg.data["mode"] = args.mode
    run_id = args.run_id or default_run_id(args.mode)
    if not RUN_RE.match(run_id):
        print(f"invalid run_id {run_id!r}; expected run/sim-YYYYMMDDTHHMMSSZ-xxxxxx",
              file=sys.stderr)
        return 2
    if args.mode == "simulation" and not run_id.startswith("sim-"):
        print("simulation mode requires a sim- run_id", file=sys.stderr)
        return 2
    if args.mode == "full" and not run_id.startswith("run-"):
        print("full mode requires a run- run_id", file=sys.stderr)
        return 2

    # Frozen config must match config.sha256.
    sha_file = ROOT / "config" / "config.sha256"
    if sha_file.exists():
        expected = sha_file.read_text().split()[0]
        if expected != cfg.sha256:
            print("E_ENV: config/config.yaml does not match config/config.sha256; "
                  "write CHANGE_REQUEST.md and start a new run_id", file=sys.stderr)
            return 3

    status = load_status()
    fresh = status is None or status.get("run_id") != run_id
    if fresh and args.stage != "S00":
        print(f"refusing {args.stage}: no STATUS.json for run {run_id}; start at S00",
              file=sys.stderr)
        return 3
    if not fresh:
        stages = status.get("stages", {})
        prev = STAGES[STAGES.index(args.stage) - 1] if args.stage != "S00" else None
        if prev and stages.get(prev, {}).get("status") not in ("pass", "warn"):
            print(f"refusing {args.stage}: {prev} is "
                  f"{stages.get(prev, {}).get('status', 'missing')}, not pass/warn",
                  file=sys.stderr)
            return 3
        if status.get("next_stage") != args.stage:
            print(f"refusing {args.stage}: STATUS.next_stage is "
                  f"{status.get('next_stage')}", file=sys.stderr)
            return 3
        # S12 only packages existing outputs, so it may be re-run to refresh
        # the submission archive without recomputing the analysis.
        if args.stage != "S12" and stages.get(args.stage, {}).get("status") in ("pass", "warn"):
            print(f"refusing {args.stage}: already "
                  f"{stages[args.stage]['status']} for this run; use a new run_id",
                  file=sys.stderr)
            return 3

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    if os.environ.get("PYTHONHASHSEED") != "0":
        print("warning: PYTHONHASHSEED is not 0; set it before launch for full "
              "determinism", file=sys.stderr)

    ctx = RunContext(cfg, run_id)
    if fresh:
        init_status(ctx)

    stdout = Tee(sys.__stdout__, ctx.run_dir / "stdout" / f"{args.stage}.stdout.log")
    stderr = Tee(sys.__stderr__, ctx.run_dir / "stderr" / f"{args.stage}.stderr.log")
    sys.stdout, sys.stderr = stdout, stderr
    code = 1
    try:
        print(f"[{args.stage}] run={run_id} mode={args.mode} "
              f"config={cfg.hash_ref}", flush=True)
        module = importlib.import_module(MODULES[args.stage])
        with ctx.stage(args.stage) as st:
            module.run(ctx, st)
        # If finish() raised a StageFailure the context still finalises and
        # re-raises; normal completion is pass/warn.
        status_now = load_status()
        code = 0 if (status_now and status_now["stages"][args.stage]["status"]
                     in ("pass", "warn")) else 1
    except BaseException as e:  # noqa: BLE001 - the stage context already logged
        print(f"[{args.stage}] FAILED: {type(e).__name__}: {e}", file=sys.stderr,
              flush=True)
        code = 1
    finally:
        sys.stdout, sys.stderr = stdout.stream, stderr.stream
        stdout.close()
        stderr.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
