#!/usr/bin/env python3
"""Validate a completed run: schema, sequence, gates, hashes, handoffs, citations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os
os.environ.setdefault("OULAD_ROOT", str(ROOT))

from src.validate import validate_run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    res = validate_run(args.run_id, require_complete=True)
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print(f"run={args.run_id} events={res.get('n_events', 0)} "
              f"errors={len(res['errors'])} warnings={len(res['warnings'])}")
        for e in res["errors"]:
            print("ERROR:", e)
        for w in res["warnings"]:
            print("WARN :", w)
        print("stages:", json.dumps(res["stages"], sort_keys=True))
    return 1 if res["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
