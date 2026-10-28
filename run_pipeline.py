"""Run the full pipeline: python run_pipeline.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from qcr.pipeline import run  # noqa: E402

if __name__ == "__main__":
    run()
