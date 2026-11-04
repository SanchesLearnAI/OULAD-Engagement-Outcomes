"""S12 — manifest, report, poster, README, ZIP and final validation."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

from src.config import project_root, sha256_file
from src.logging_utils import (StageFailure, atomic_write_json, atomic_write_text,
                               load_status, utc_now)
from src.reporting import (build_latex, build_markdown, build_poster,
                           build_readme, citation_scan, compile_pdf,
                           pdf_pages)
from src.validate import validate_run


def _manifest_all(ctx) -> dict:
    items = {}
    for line in (ctx.run_dir / "run.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e.get("event") == "artifact":
            for o in e.get("outputs", []):
                items[o["path"]] = {"path": o["path"], "sha256": o.get("sha256"),
                                    "rows": o.get("rows"), "cols": o.get("cols"),
                                    "stage": e.get("stage")}
    return {"schema_version": "1.0", "run_id": ctx.run_id, "mode": ctx.mode,
            "config_hash": ctx.cfg.hash_ref, "artifacts": sorted(
                items.values(), key=lambda x: x["path"])}


def _requirements_lock(root: Path):
    p = root / "REQUIREMENTS.lock"
    if p.exists():
        return p
    try:
        r = subprocess.run(["python3", "-m", "pip", "freeze"], capture_output=True,
                           text=True, timeout=180)
        p.write_text(r.stdout)
    except Exception:
        p.write_text("# pip freeze unavailable\n")
    return p


def _data_download(ctx, root: Path) -> Path:
    raw_manifest = ctx.run_dir / "manifest.json"
    rows = []
    if raw_manifest.exists():
        try:
            m = json.loads(raw_manifest.read_text())
            for f in m.get("files", []):
                rows.append(f"| {f['name']} | {f['rows']} | `{f['sha256']}` |")
        except Exception:
            pass
    text = """# Data download

The project uses the public Open University Learning Analytics Dataset (OULAD),
available from UCI:

<https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip>

```bash
mkdir -p data/raw/oulad
curl -L --fail --retry 3 -o data/raw/oulad/oulad.zip \\
  "https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip"
shasum -a 256 data/raw/oulad/oulad.zip
unzip -o data/raw/oulad/oulad.zip -d data/raw/oulad/
```

The ZIP does not contain the raw CSVs. The audit in this run recorded:

| File | Rows | SHA-256 |
|---|---:|---|
""" + "\n".join(rows) + "\n"
    p = root / "DATA_DOWNLOAD.md"
    atomic_write_text(p, text)
    return p


def _zip(ctx, report_pdf, poster_pdf, readme, report_md, req_lock, data_doc):
    root = project_root()
    out = root / "Group_09_Project.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p, arc in ((report_pdf, "Report.pdf"),
                       (readme, "README.md"), (report_md, "REPORT.md"),
                       (req_lock, "REQUIREMENTS.lock"),
                       (data_doc, "DATA_DOWNLOAD.md")):
            if p.exists():
                z.write(p, arc)
        for rel in ("config/config.yaml", "config/config.sha256"):
            p = root / rel
            if p.exists():
                z.write(p, rel)
        for rel in ("Makefile",):
            p = root / rel
            if p.exists():
                z.write(p, rel)
        for d in ("src", "stages", "scripts", "tests"):
            for p in (root / d).rglob("*.py"):
                if "__pycache__" in str(p) or p.name.startswith("inspect_raw"):
                    continue
                z.write(p, str(p.relative_to(root)))
        for d in ("tables", "figures"):
            for p in (ctx.paths.outputs_dir / d).glob("*"):
                if p.is_file() and p.name != "t10_hazard_ratios.csv":
                    z.write(p, str(p.relative_to(ctx.paths.outputs_dir)))
        for rel in ("run.jsonl", "env.json", "manifest.json", "manifest_all.json",
                    "STATUS.final.json"):
            p = ctx.run_dir / rel
            if p.exists():
                z.write(p, f"runs/{ctx.run_id}/{rel}")
        for d in ("stages", "handoff"):
            for p in (ctx.run_dir / d).glob("*"):
                z.write(p, f"runs/{ctx.run_id}/{d}/{p.name}")
        cards = ctx.run_dir / "artifacts" / "model_cards.json"
        if cards.exists():
            z.write(cards, f"runs/{ctx.run_id}/artifacts/model_cards.json")
    # The revision guide requires a single Group_09 archive; remove the old
    # literal-name copy if it exists.
    old_xx = root / "Group_XX_Project.zip"
    if old_xx.exists():
        old_xx.unlink()
    return out


def _main_text_pages(pdf: Path) -> int:
    try:
        r = subprocess.run(["pdftotext", "-layout", str(pdf), "-"],
                           capture_output=True, text=True)
        m = re.search(r"main text ends on page\s+(\d+)", r.stdout.replace("\n", " "))
        return int(m.group(1)) if m else -1
    except Exception:
        return -1


def _pages_from_log(log: str) -> int:
    m = re.search(r"Output written on .*?\((\d+) pages?", log)
    return int(m.group(1)) if m else -1


def _write_b2_failed(ctx, st):
    """WP5.2: keep the failed B2 row out of Table 7 but preserve it in appendix."""
    p = ctx.paths.tables_dir / "t09_incremental.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    b2 = d[(d["cutoff"] == 56) & (d["cohort"] == "fixed") &
           (d["method"] == "B2_offset")].copy()
    if len(b2) == 0:
        return
    out = ctx.paths.tables_dir / "t09_b2_failed.csv"
    b2.to_csv(out, index=False)
    st.artifact(out, rows=len(b2), cols=b2.shape[1],
                message="failed B2 appendix table")


def run(ctx, st):
    root = project_root()
    out = ctx.paths.outputs_dir
    _write_b2_failed(ctx, st)

    with st.step("manifest"):
        ma = _manifest_all(ctx)
        p = ctx.run_dir / "manifest_all.json"
        atomic_write_json(p, ma)
        st.artifact(p, rows=len(ma["artifacts"]), message="all-artifact manifest")

    with st.step("report"):
        tex = out / "Report.tex"
        tex.write_text(build_latex(ctx))
        ok, log = compile_pdf(tex, out, runs=2)
        if not ok:
            raise StageFailure("Report.tex failed to compile: " + log[-2000:], "E_LOGIC")
        pages = pdf_pages(out / "Report.pdf")
        if pages < 0:
            pages = _pages_from_log(log)
        st.artifact(out / "Report.pdf", message="Report.pdf")
        main_pages = _main_text_pages(out / "Report.pdf")
        st.metric("report", {"pages": pages, "main_pages": main_pages,
                             "mode": ctx.mode})
        st.check("report_pdf_exists", (out / "Report.pdf").exists(), "Report.pdf")
        st.check("report_page_count", 6 <= pages <= 20,
                 f"total pages={pages}; appendix/references excluded from main-text count")
        st.check("main_text_page_count", 6 <= main_pages <= 10,
                 f"main text pages={main_pages} (target 6-10)")
        text = tex.read_text()
        st.check("disclosures_present", "Disclosures" in text,
                 "AIGC / external collaborator disclosure section present")
        errs, warns = citation_scan(text)
        st.check("citation_scan_errors", not errs, "; ".join(errs) or "no rejected numbers")
        if errs:
            raise StageFailure("citation scan failed: " + "; ".join(errs), "E_LOGIC")
        st.check("citation_scan_warnings", True, "; ".join(warns) or "no qualifier warnings")
        md = root / "REPORT.md"
        atomic_write_text(md, build_markdown(ctx))
        st.artifact(md, message="REPORT.md")
        shutil.copyfile(tex, root / "Report.tex")
        if ctx.mode == "full":
            shutil.copyfile(out / "Report.pdf", root / "Report.pdf")

    with st.step("poster"):
        ptex = out / "Poster.tex"
        ptex.write_text(build_poster(ctx))
        ok, plog = compile_pdf(ptex, out, runs=1)
        if not ok:
            raise StageFailure("Poster.tex failed to compile: " + plog[-2000:], "E_LOGIC")
        st.artifact(out / "Poster.pdf", message="Poster.pdf")
        st.check("poster_pdf_exists", (out / "Poster.pdf").exists(), "Poster.pdf")
        if ctx.mode == "full":
            shutil.copyfile(out / "Poster.pdf", root / "Poster.pdf")

    with st.step("readme"):
        readme = root / "README.md"
        atomic_write_text(readme, build_readme(ctx))
        st.artifact(readme, message="README.md")
        req = _requirements_lock(root)
        st.artifact(req, message="REQUIREMENTS.lock")
        dd = _data_download(ctx, root)
        st.artifact(dd, message="DATA_DOWNLOAD.md")

    with st.step("validation"):
        v = validate_run(ctx.run_id, require_complete=False)
        st.metric("pre_validation", {"errors": len(v["errors"]),
                                     "warnings": len(v["warnings"])})
        st.check("pre_validation_errors", not v["errors"],
                 "; ".join(v["errors"][:10]) or "no structural errors")
        if v["errors"]:
            raise StageFailure("pre-S12 validation failed: " + "; ".join(v["errors"][:10]),
                               "E_LOGIC")
        for w in v["warnings"]:
            st.warn(w, "validation")

    with st.step("status_final"):
        stf = load_status() or {}
        stf.setdefault("stages", {})["S12"] = {
            "status": "pass", "started_at": st.started_at,
            "finished_at": utc_now(),
            "log": f"runs/{ctx.run_id}/run.jsonl",
            "stage_json": f"runs/{ctx.run_id}/stages/S12.stage.json",
            "artifacts": [], "notes": "final status written before packaging",
        }
        stf.update({"run_id": ctx.run_id, "mode": ctx.mode,
                    "current_stage": "S12", "next_stage": None,
                    "run_status": "complete", "updated_at": utc_now()})
        sfp = ctx.run_dir / "STATUS.final.json"
        atomic_write_json(sfp, stf)
        st.artifact(sfp, message="STATUS.final.json")

    with st.step("zip"):
        report_pdf = out / "Report.pdf"
        poster_pdf = out / "Poster.pdf"
        readme = root / "README.md"
        report_md = root / "REPORT.md"
        req = root / "REQUIREMENTS.lock"
        dd = root / "DATA_DOWNLOAD.md"
        zp = _zip(ctx, report_pdf, poster_pdf, readme, report_md, req, dd)
        st.artifact(zp, message="Group_09_Project.zip")
        with zipfile.ZipFile(zp) as z:
            names = z.namelist()
        st.check("zip_no_raw_data",
                 not any(n.startswith("data/") or n.endswith("studentVle.csv")
                         for n in names),
                 "raw full data is not inside the ZIP")
        st.check("zip_has_report", any(n == "Report.pdf" for n in names),
                 "Report.pdf in ZIP")
        st.check("zip_poster_excluded", not any(n == "Poster.pdf" for n in names),
                 "Poster.pdf is submitted separately, not inside the final ZIP")
        st.check("zip_has_status_final",
                 any(n == f"runs/{ctx.run_id}/STATUS.final.json" for n in names),
                 "STATUS.final.json in ZIP")
        st.check("zip_no_stale_hazard",
                 not any(n.endswith("t10_hazard_ratios.csv") for n in names),
                 "stale static-Cox hazard table is not inside the ZIP")
        wp5_tables = ("tables/t08_context_delta.csv",
                      "tables/t09_b2_failed.csv",
                      "tables/t11_placebo_exposure.csv")
        st.check("zip_has_wp5_tables",
                 all(t in names for t in wp5_tables),
                 "WP5 context-delta, B2-appendix and placebo-exposure tables in ZIP")
        zips = sorted(p.name for p in root.glob("Group_*_Project.zip"))
        st.check("single_submission_zip", zips == ["Group_09_Project.zip"],
                 f"submission ZIPs={zips}")
        st.metric("zip", {"entries": len(names),
                          "bytes": zp.stat().st_size})
