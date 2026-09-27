"""Rebuild the human-readable post-P8 report from saved research artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from adaptive_mas.evaluation.post_p8_report import render_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = render_report(args.output_dir)
    print(f"report={report}")


if __name__ == "__main__":
    main()
