"""Append-only event log, stage summaries, handoffs and atomic STATUS updates.

The contract is EXECUTION_SIMULATION.md sections 3 and 4.  Every event is
validated against 02_contracts/log_event.schema.json before it is written.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import jsonschema

from .config import Config, Paths, project_root, sha256_file

STAGES = [f"S{i:02d}" for i in range(13)]
NEXT_STAGE = {s: (STAGES[i + 1] if i + 1 < len(STAGES) else None)
              for i, s in enumerate(STAGES)}

NEXT_STAGE_MUST_NOT = {
    "S00": "do not re-download or overwrite raw CSVs; do not change config",
    "S01": "do not drop rows based on final_result; do not treat '?' as a value",
    "S02": "do not silently filter the cohort on date_unregistration; use the audit",
    "S03": "do not assume Pass/Fail click distributions overlap",
    "S04": "do not add score/is_banked/future-dated fields to strict features",
    "S05": "do not replace the grouped split with a row-level split",
    "S06": "do not write causal language; matching is descriptive",
    "S07": "do not name clusters as learning styles; report stability",
    "S08": "do not tune on the test set; M4 is an upper bound only",
    "S09": "do not use a W1 probability fitted on the same fold it predicts",
    "S10": "do not fold Withdrawn into Fail",
    "S11": "do not report rejected numbers; do not hide negative controls",
    "S12": "do not invent references or include raw full data in the ZIP",
}


def utc_now() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root()))
    except Exception:
        return str(path)


class StageFailure(RuntimeError):
    def __init__(self, message: str, error_type: str = "E_LOGIC"):
        super().__init__(message)
        self.error_type = error_type


class RunContext:
    def __init__(self, cfg: Config, run_id: str):
        self.cfg = cfg
        self.run_id = run_id
        self.mode = cfg["mode"]
        self.simulated = self.mode == "simulation"
        self.paths = Paths.build(cfg, run_id)
        self.run_dir = self.paths.run_dir
        self.jsonl = self.run_dir / "run.jsonl"
        self.schema = json.loads(
            (project_root() / "02_contracts" / "log_event.schema.json").read_text()
        )
        self.seq = self._load_seq()
        self._fh = open(self.jsonl, "a", encoding="utf-8")

    def _load_seq(self) -> int:
        if not self.jsonl.exists():
            return 0
        last = 0
        with open(self.jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    last = int(json.loads(line)["seq"])
                except Exception:
                    continue
        return last

    def close(self) -> None:
        try:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()
        except Exception:
            pass

    def log(self, *, stage: str, event: str, status: str = "ok",
            step: str | None = None, inputs=None, outputs=None, metrics=None,
            message: str | None = None, error: dict | None = None,
            duration_ms: int = 0) -> dict:
        self.seq += 1
        rec = {
            "schema_version": "1.0",
            "seq": self.seq,
            "ts": utc_now(),
            "run_id": self.run_id,
            "mode": self.mode,
            "stage": stage,
            "step": step,
            "event": event,
            "status": status,
            "agent": "executor",
            "config_hash": self.cfg.hash_ref,
            "git_commit": "no_git",
            "duration_ms": int(duration_ms),
            "inputs": inputs or [],
            "outputs": outputs or [],
            "metrics": metrics or {},
            "message": message,
            "error": error,
            "simulated": self.simulated,
        }
        try:
            jsonschema.validate(rec, self.schema)
        except jsonschema.ValidationError as e:
            # A malformed log line is itself a stage failure (section 3.5).
            raise StageFailure(f"log schema violation: {e.message}", "E_LOGIC")
        self._fh.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._fh.flush()
        if self.cfg["logging"].get("flush_each_event", True):
            os.fsync(self._fh.fileno())
        return rec

    def stage(self, stage: str) -> "Stage":
        return Stage(self, stage)


class Stage:
    def __init__(self, ctx: RunContext, stage: str):
        self.ctx = ctx
        self.stage = stage
        self.started_at = utc_now()
        self.t0 = time.perf_counter()
        self.checks: list[dict] = []
        self.metrics: dict = {}
        self.artifacts: list[dict] = []
        self.warnings: list[str] = []
        self.failed = False
        self.fail_reason = None
        self.error_type = None
        self.notes = ""
        self.extra = {}
        ctx.log(stage=stage, event="stage_start", message=f"{stage} start")

    # -- event helpers -----------------------------------------------------
    def check(self, name: str, ok: bool, message: str = "", *, metrics=None,
              warn_only: bool = False) -> bool:
        status = "ok" if ok else ("warn" if warn_only else "fail")
        rec = {"name": name, "status": status, "message": message,
               "metrics": metrics or {}}
        self.checks.append(rec)
        if ok:
            self.ctx.log(stage=self.stage, event="check", status="ok",
                         step=name, message=message, metrics=metrics or {})
        elif warn_only:
            self.warnings.append(f"{name}: {message}")
            self.ctx.log(stage=self.stage, event="check", status="warn",
                         step=name, message=message, metrics=metrics or {})
        else:
            self.failed = True
            self.fail_reason = f"check failed: {name}: {message}"
            self.ctx.log(stage=self.stage, event="check", status="fail",
                         step=name, message=message, metrics=metrics or {})
        return bool(ok)

    def require(self, name: str, ok: bool, message: str = "", *, metrics=None):
        if not self.check(name, ok, message, metrics=metrics):
            raise StageFailure(f"{name}: {message}", "E_METRIC")

    def metric(self, name: str, metrics: dict, message: str = "") -> None:
        self.metrics.update(metrics)
        self.ctx.log(stage=self.stage, event="metric", step=name,
                     metrics=metrics, message=message or name)

    def warn(self, message: str, step: str = "warning") -> None:
        self.warnings.append(message)
        self.ctx.log(stage=self.stage, event="warning", status="warn",
                     step=step, message=message)

    def artifact(self, path: Path, *, rows=None, cols=None, message: str = "") -> dict:
        path = Path(path)
        if not path.is_absolute():
            path = project_root() / path
        if not path.exists():
            raise StageFailure(f"declared artifact missing: {path}", "E_LOGIC")
        digest = sha256_file(path)
        ref = {"path": _rel(path), "sha256": f"sha256:{digest}"}
        if rows is not None:
            ref["rows"] = int(rows)
        if cols is not None:
            ref["cols"] = int(cols)
        self.artifacts.append(ref)
        self.ctx.log(stage=self.stage, event="artifact", step="artifact",
                     outputs=[ref], message=message or f"wrote {ref['path']}")
        return ref

    @contextmanager
    def step(self, name: str, message: str = ""):
        t0 = time.perf_counter()
        self.ctx.log(stage=self.stage, event="step_start", step=name,
                     message=message or name)
        try:
            yield self
        except Exception as e:
            self.ctx.log(stage=self.stage, event="step_end", status="fail",
                         step=name, duration_ms=int((time.perf_counter() - t0) * 1000),
                         message=f"{type(e).__name__}: {e}")
            raise
        self.ctx.log(stage=self.stage, event="step_end", step=name,
                     duration_ms=int((time.perf_counter() - t0) * 1000),
                     message="ok")

    # -- finalisation ------------------------------------------------------
    def _stage_json(self, status: str, error) -> dict:
        return {
            "schema_version": "1.0",
            "stage": self.stage,
            "run_id": self.ctx.run_id,
            "mode": self.ctx.mode,
            "simulated": self.ctx.simulated,
            "status": status,
            "started_at": self.started_at,
            "finished_at": utc_now(),
            "duration_ms": int((time.perf_counter() - self.t0) * 1000),
            "config_hash": self.ctx.cfg.hash_ref,
            "checks": self.checks,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "warnings": self.warnings,
            "notes": self.notes,
            "error": error,
            **self.extra,
        }

    def _handoff(self, status: str, error) -> str:
        lines = [f"# {self.stage} handoff", ""]
        lines.append(f"- run_id: {self.ctx.run_id}")
        lines.append(f"- mode: {self.ctx.mode}")
        lines.append(f"- status: {status}")
        lines.append(f"- config_hash: {self.ctx.cfg.hash_ref}")
        lines.append("- outputs:")
        for a in self.artifacts:
            extra = f" (rows={a.get('rows')}, cols={a.get('cols')})" if a.get("rows") is not None else ""
            lines.append(f"  - {a['path']} {a['sha256']}{extra}")
        lines.append("- checks:")
        for c in self.checks:
            lines.append(f"  - [{c['status']}] {c['name']}: {c['message']}")
        lines.append("- metrics:")
        for k, v in self.metrics.items():
            if isinstance(v, (dict, list)):
                lines.append(f"  - {k}: {json.dumps(v, ensure_ascii=False)}")
            else:
                lines.append(f"  - {k}: {v}")
        lines.append("- open issues:")
        if self.warnings:
            for w in self.warnings:
                lines.append(f"  - {w}")
        else:
            lines.append("  - none")
        if error:
            lines.append(f"- error: {json.dumps(error, ensure_ascii=False)}")
        lines.append(f"- next stage must not: {NEXT_STAGE_MUST_NOT.get(self.stage, 'n/a')}")
        return "\n".join(lines) + "\n"

    def _status(self, status: str, error) -> dict:
        root = project_root()
        spath = root / "STATUS.json"
        if spath.exists():
            try:
                st = json.loads(spath.read_text())
            except Exception:
                st = {}
        else:
            st = {}
        if st.get("run_id") != self.ctx.run_id:
            st = {
                "schema_version": "1.0",
                "run_id": self.ctx.run_id,
                "mode": self.ctx.mode,
                "updated_at": utc_now(),
                "current_stage": self.stage,
                "next_stage": self.stage,
                "stages": {},
                "blocked": None,
                "notes": "",
                "run_status": "running",
            }
        st["stages"] = st.get("stages", {})
        st["stages"][self.stage] = {
            "status": status,
            "started_at": self.started_at,
            "finished_at": utc_now(),
            "log": _rel(self.ctx.jsonl),
            "stage_json": _rel(self.ctx.run_dir / "stages" / f"{self.stage}.stage.json"),
            "artifacts": [a["path"] for a in self.artifacts],
            "notes": self.notes,
        }
        st["current_stage"] = self.stage
        st["updated_at"] = utc_now()
        if status in ("pass", "warn"):
            st["blocked"] = None
            st["next_stage"] = NEXT_STAGE[self.stage]
            if self.stage == "S12":
                st["run_status"] = "complete"
            else:
                st["run_status"] = "running"
            st["notes"] = f"{self.stage} {status}; next={st['next_stage']}"
        else:
            st["blocked"] = {
                "stage": self.stage,
                "reason": self.fail_reason or "stage failed",
                "log_seq": self.ctx.seq,
                "error_type": self.error_type or "E_LOGIC",
            }
            st["next_stage"] = self.stage
            st["run_status"] = "blocked"
            st["notes"] = f"{self.stage} failed; blocked"
        return st

    def finish(self, exc: BaseException | None = None) -> str:
        error = None
        if exc is not None:
            self.failed = True
            self.error_type = getattr(exc, "error_type", "E_LOGIC")
            self.fail_reason = self.fail_reason or f"{type(exc).__name__}: {exc}"
            error = {"type": self.error_type, "message": str(exc),
                     "traceback_ref": _rel(self.ctx.run_dir / "stderr" /
                                           f"{self.stage}.stderr.log")}
            self.ctx.log(stage=self.stage, event="error", status="fail",
                         step="exception", message=str(exc), error=error)
        if self.failed:
            status = "fail"
        elif self.warnings:
            status = "warn"
        else:
            status = "pass"

        if status == "fail":
            self.ctx.log(stage=self.stage, event="gate_fail", status="fail",
                         metrics=self.metrics,
                         message=self.fail_reason or "gate failed")
        else:
            self.ctx.log(stage=self.stage, event="gate_pass",
                         status=("warn" if status == "warn" else "ok"),
                         metrics=self.metrics,
                         message=("passed with warnings: " + "; ".join(self.warnings)
                                  if self.warnings else "all gates passed"))

        duration = int((time.perf_counter() - self.t0) * 1000)
        # The log schema's status enum is ok|warn|fail|skip; "pass" is a
        # STATUS/stage-json status, not an event status.
        self.ctx.log(stage=self.stage, event="stage_end",
                     status=("ok" if status == "pass" else status),
                     duration_ms=duration,
                     message=f"{self.stage} {status}")

        # 4. stage json, 5. STATUS, then handoff (atomic).
        sj = self._stage_json(status, error)
        atomic_write_json(self.ctx.run_dir / "stages" / f"{self.stage}.stage.json", sj)
        atomic_write_json(project_root() / "STATUS.json", self._status(status, error))
        atomic_write_text(self.ctx.run_dir / "handoff" / f"{self.stage}.md",
                          self._handoff(status, error))
        self.ctx.close()
        return status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.finish(exc)
        return False  # never suppress


def init_status(ctx: RunContext) -> None:
    """Create STATUS.json for a new run, archiving the previous run's status."""
    root = project_root()
    spath = root / "STATUS.json"
    if spath.exists():
        try:
            old = json.loads(spath.read_text())
            old_id = old.get("run_id")
            if old_id and old_id != ctx.run_id:
                archive = ctx.paths.runs_dir / old_id / "STATUS.final.json"
                if not archive.exists():
                    atomic_write_json(archive, old)
        except Exception:
            pass
    st = {
        "schema_version": "1.0",
        "run_id": ctx.run_id,
        "mode": ctx.mode,
        "updated_at": utc_now(),
        "current_stage": None,
        "next_stage": "S00",
        "stages": {},
        "blocked": None,
        "notes": "new run initialised",
        "run_status": "running",
    }
    atomic_write_json(spath, st)


def load_status() -> dict | None:
    p = project_root() / "STATUS.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None
