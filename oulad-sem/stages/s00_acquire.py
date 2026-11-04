"""S00 — environment + raw data acquisition + hashes."""
from __future__ import annotations

import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

from src.config import project_root, sha256_file
from src.logging_utils import atomic_write_json, utc_now

REQUIRED = ["courses.csv", "assessments.csv", "vle.csv", "studentInfo.csv",
            "studentRegistration.csv", "studentAssessment.csv", "studentVle.csv"]
URL = ("https://archive.ics.uci.edu/static/public/349/"
       "open+university+learning+analytics+dataset.zip")


def _count_rows_and_header(path: Path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
        n = sum(1 for _ in f)
    return n, len(header), header


def _safe_extract(zip_path: Path, dest: Path) -> None:
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            target = (dest / info.filename).resolve()
            if not str(target).startswith(str(dest) + os.sep):
                raise RuntimeError(f"unsafe zip member: {info.filename}")
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)


def run(ctx, st):
    root = project_root()
    raw = ctx.paths.raw_dir
    raw.mkdir(parents=True, exist_ok=True)

    with st.step("env_check"):
        disk = shutil.disk_usage(root)
        env = {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "uname": " ".join(platform.uname()),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "cwd": str(Path.cwd()),
            "root": str(root),
            "disk_free_gb": round(disk.free / 1e9, 2),
            "config_hash": ctx.cfg.hash_ref,
            "run_id": ctx.run_id,
            "mode": ctx.mode,
            "timestamp": utc_now(),
        }
        for mod in ("pandas", "numpy", "sklearn", "scipy", "yaml", "jsonschema",
                    "pyarrow", "matplotlib", "seaborn", "statsmodels"):
            try:
                m = __import__(mod)
                env[f"pkg_{mod}"] = getattr(m, "__version__", "unknown")
            except Exception as e:  # noqa: BLE001
                env[f"pkg_{mod}"] = f"MISSING: {e}"
        try:
            freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                    capture_output=True, text=True, timeout=120)
            env["pip_freeze"] = freeze.stdout.splitlines()
        except Exception as e:  # noqa: BLE001
            env["pip_freeze"] = [f"unavailable: {e}"]
        atomic_write_json(ctx.run_dir / "env.json", env)
        st.artifact(ctx.run_dir / "env.json", rows=1, message="environment snapshot")
        st.metric("env", {"python": platform.python_version(),
                          "disk_free_gb": env["disk_free_gb"],
                          "PYTHONHASHSEED": env["PYTHONHASHSEED"]})
        st.check("python_3_10_plus", sys.version_info >= (3, 10),
                 f"python {platform.python_version()}")
        st.check("disk_16gb_plus", disk.free >= 16e9,
                 f"free {disk.free/1e9:.1f} GB", warn_only=True)

    with st.step("raw_data"):
        missing = [f for f in REQUIRED if not (raw / f).exists()]
        if missing:
            zpath = raw / "oulad.zip"
            if not zpath.exists():
                st.warn(f"raw CSVs missing ({missing}); downloading {URL}", "download")
                r = subprocess.run(["curl", "-L", "--fail", "--retry", "3",
                                    "-o", str(zpath), URL], capture_output=True,
                                   text=True, timeout=900)
                if r.returncode != 0:
                    from src.logging_utils import StageFailure
                    raise StageFailure(f"download failed: {r.stderr[-500:]}",
                                       "E_RESOURCE")
            zsha = sha256_file(zpath)
            (raw / "oulad.zip.sha256").write_text(f"{zsha}  oulad.zip\n")
            _safe_extract(zpath, raw)
        missing = [f for f in REQUIRED if not (raw / f).exists()]
        st.check("seven_csvs", not missing, f"missing={missing}",
                 metrics={"files": len(REQUIRED) - len(missing)})
        if missing:
            return

    with st.step("raw_manifest"):
        files = []
        for name in REQUIRED:
            p = raw / name
            rows, cols, header = _count_rows_and_header(p)
            files.append({
                "name": name,
                "path": str(p.relative_to(root)),
                "sha256": sha256_file(p),
                "rows": rows,
                "cols": cols,
                "columns": header,
                "bytes": p.stat().st_size,
            })
            st.artifact(p, rows=rows, cols=cols, message=f"raw {name}")
        zi = raw / "oulad.zip"
        manifest = {
            "schema_version": "1.0",
            "run_id": ctx.run_id,
            "mode": ctx.mode,
            "generated_at": utc_now(),
            "config_hash": ctx.cfg.hash_ref,
            "source_url": URL,
            "archive": ({
                "path": str(zi.relative_to(root)),
                "sha256": sha256_file(zi),
                "bytes": zi.stat().st_size,
            } if zi.exists() else None),
            "files": files,
        }
        atomic_write_json(ctx.run_dir / "manifest.json", manifest)
        st.artifact(ctx.run_dir / "manifest.json", rows=len(files),
                    message="raw manifest")
        st.metric("raw_manifest", {
            "files": len(files),
            "studentinfo_rows": next(f["rows"] for f in files
                                     if f["name"] == "studentInfo.csv"),
            "studentvle_rows": next(f["rows"] for f in files
                                    if f["name"] == "studentVle.csv"),
            "total_bytes": sum(f["bytes"] for f in files),
        })

    with st.step("label_check"):
        si = pd.read_csv(raw / "studentInfo.csv", na_values=["?"])
        counts = si["final_result"].value_counts(dropna=False).to_dict()
        st.metric("final_result", {"counts": {str(k): int(v) for k, v in counts.items()},
                                   "n_classes": int(si["final_result"].nunique())})
        st.check("studentinfo_rows_32593", len(si) == 32593,
                 f"rows={len(si)}", metrics={"rows": int(len(si))})
        st.check("four_final_result_classes",
                 si["final_result"].nunique() == 4 and si["final_result"].notna().all(),
                 f"classes={sorted(si['final_result'].dropna().unique())}")
