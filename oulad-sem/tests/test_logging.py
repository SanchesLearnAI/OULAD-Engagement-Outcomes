import json
from pathlib import Path
import jsonschema

ROOT = Path(__file__).resolve().parents[1]


def test_event_schema_accepts_valid_event():
    schema = json.loads((ROOT / "02_contracts" / "log_event.schema.json").read_text())
    event = {
        "schema_version": "1.0", "seq": 1,
        "ts": "2026-11-04T00:00:00.000Z",
        "run_id": "sim-20261104T000000Z-abcdef", "mode": "simulation",
        "stage": "S00", "event": "stage_start", "status": "ok",
        "config_hash": "sha256:" + "0" * 64,
    }
    jsonschema.validate(event, schema)


def test_event_schema_rejects_bad_status():
    import pytest
    schema = json.loads((ROOT / "02_contracts" / "log_event.schema.json").read_text())
    event = {
        "schema_version": "1.0", "seq": 1,
        "ts": "2026-11-04T00:00:00.000Z",
        "run_id": "sim-20261104T000000Z-abcdef", "mode": "simulation",
        "stage": "S00", "event": "stage_start", "status": "pass",
        "config_hash": "sha256:" + "0" * 64,
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(event, schema)
