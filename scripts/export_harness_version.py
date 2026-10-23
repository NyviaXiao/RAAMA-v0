"""Export the active public-research harness prompt version without run data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaptive_mas.harness_evolution import HarnessVersionStore


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version-dir",
        type=Path,
        default=project_dir / "data" / "processed" / "harness_versions",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export = HarnessVersionStore(args.version_dir).export_active(args.output)
    print(json.dumps({
        "status": "exported",
        "output": str(args.output),
        "version_id": export["version_id"],
        "candidate_sha256": export["candidate_sha256"],
        "included_fields": sorted(export),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
