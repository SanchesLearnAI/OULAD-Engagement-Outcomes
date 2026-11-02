#!/bin/bash
#SBATCH --job-name=oulad28
#SBATCH --partition=i64m512u
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/hpc2hdd/home/aiaa3111_l02_013/projects/oulad-engagement-outcomes/slurm-%j.out
#SBATCH --error=/hpc2hdd/home/aiaa3111_l02_013/projects/oulad-engagement-outcomes/slurm-%j.err

set -euo pipefail
PROJECT=/hpc2hdd/home/aiaa3111_l02_013/projects/oulad-engagement-outcomes
DATA_DIR=${DATA_DIR:-$PROJECT/data/raw}
RAW_ZIP=${RAW_ZIP:-$PROJECT/data/oulad.zip}
RUN_DIR=${RUN_DIR:?Set RUN_DIR to a new, verified output directory}
test -d "$PROJECT/.venv"
if test -e "$RUN_DIR"; then
  echo "Output already exists: $RUN_DIR" >&2
  exit 2
fi
cd "$PROJECT"
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python --version
if ! test -f "$DATA_DIR/studentInfo.csv" || ! test -f "$DATA_DIR/studentVle.csv" || ! test -f "$DATA_DIR/vle.csv"; then
  test -f "$RAW_ZIP"
  if test -e "$DATA_DIR"; then
    echo "Incomplete input directory exists: $DATA_DIR" >&2
    exit 2
  fi
  python - "$RAW_ZIP" "$DATA_DIR" <<'PY'
import sys
import zipfile
from pathlib import Path

archive, destination = Path(sys.argv[1]), Path(sys.argv[2])
with zipfile.ZipFile(archive) as source:
    members = ["studentInfo.csv", "studentVle.csv", "vle.csv"]
    if not set(members).issubset(source.namelist()):
        raise ValueError("Raw archive lacks required OULAD files")
    destination.mkdir(parents=True, exist_ok=False)
    source.extractall(destination, members=members)
PY
fi
mkdir -p "$RUN_DIR"
python -m oulad_analysis --data-dir "$DATA_DIR" --output-dir "$RUN_DIR" --stage all --seed 42 > "$RUN_DIR/run.log" 2>&1
