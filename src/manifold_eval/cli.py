"""Command-line entry points for manifold-eval."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def viewer(argv: list[str] | None = None) -> int:
    """Launch the Streamlit result viewer."""
    parser = argparse.ArgumentParser(description="Open the manifold-eval result viewer.")
    parser.add_argument(
        "--results-dir",
        default="output",
        help="Directory containing saved eval results.",
    )
    args = parser.parse_args(argv)

    viewer_path = Path(__file__).resolve().parent / "presentation" / "result_viewer.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(viewer_path),
        "--",
        "--results-dir",
        str(args.results_dir),
    ]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(viewer())
