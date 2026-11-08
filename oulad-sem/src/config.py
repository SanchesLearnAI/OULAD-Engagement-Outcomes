"""Frozen-config loading and mode-aware paths."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import yaml


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def project_root() -> Path:
    return Path(os.environ.get("OULAD_ROOT", Path(__file__).resolve().parents[1])).resolve()


def contracts_dir() -> Path:
    """Directory holding the JSON schemas and the example config."""
    return project_root() / "docs" / "contracts"


@dataclass
class Config:
    data: dict
    path: Path
    sha256: str  # bare hex, no prefix

    @property
    def hash_ref(self) -> str:
        return f"sha256:{self.sha256}"

    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        return self.data.get(key, default)


def load_config(path: str | Path | None = None) -> Config:
    root = project_root()
    path = Path(path) if path else root / "config" / "config.yaml"
    if not path.is_absolute():
        path = root / path
    raw = path.read_bytes()
    data = yaml.safe_load(raw.decode("utf-8"))
    return Config(data=data, path=path, sha256=hashlib.sha256(raw).hexdigest())


@dataclass
class Paths:
    root: Path
    mode: str
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    outputs_dir: Path
    tables_dir: Path
    figures_dir: Path
    runs_dir: Path
    run_dir: Path

    @classmethod
    def build(cls, cfg: Config, run_id: str) -> "Paths":
        root = project_root()
        mode = cfg["mode"]
        raw = root / cfg["paths"]["raw_dir"]
        interim = root / cfg["paths"]["interim_dir"]
        processed = root / cfg["paths"]["processed_dir"]
        outputs = root / cfg["paths"]["outputs_dir"]
        # Simulation must not overwrite full-run artifacts, otherwise the frozen
        # run.jsonl of the simulation would no longer hash-check after full runs.
        if mode == "simulation":
            interim = interim / "sim"
            processed = processed / "sim"
            outputs = outputs / "sim"
        p = cls(
            root=root,
            mode=mode,
            raw_dir=raw,
            interim_dir=interim,
            processed_dir=processed,
            outputs_dir=outputs,
            tables_dir=outputs / "tables",
            figures_dir=outputs / "figures",
            runs_dir=root / cfg["paths"]["runs_dir"],
            run_dir=root / cfg["paths"]["runs_dir"] / run_id,
        )
        for d in (p.interim_dir, p.processed_dir, p.outputs_dir, p.tables_dir,
                  p.figures_dir, p.run_dir):
            d.mkdir(parents=True, exist_ok=True)
        for sub in ("stages", "stdout", "stderr", "metrics", "artifacts", "handoff"):
            (p.run_dir / sub).mkdir(parents=True, exist_ok=True)
        return p
