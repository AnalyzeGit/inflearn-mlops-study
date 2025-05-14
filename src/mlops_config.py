# mlops/mlops_config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# 필요한 디렉토리 생성
DATA_DIR.mkdir(parents=True, exist_ok=True)