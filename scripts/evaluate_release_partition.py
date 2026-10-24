"""Predeclare, freeze package identity, and register locally computed release evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaptive_mas.release_governance import HarnessReleaseStore


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path.name}")
    return value


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    default_store = project_root / "data" / "processed" / "harness_versions"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=default_store)
    parser.add_argument("--project-root", type=Path, default=project_root)
    commands = parser.add_subparsers(dest="command", required=True)

    protocol_parser = commands.add_parser("predeclare", help="freeze protocol and partition calendar before evaluation")
    protocol_parser.add_argument("--protocol-id", required=True)
    protocol_parser.add_argument("--protocol-json", type=Path, required=True)

    package_parser = commands.add_parser("register-package", help="register a complete immutable runtime package")
    package_parser.add_argument("--package-json", type=Path, required=True)

    evaluate_parser = commands.add_parser("evaluate", help="compute an immutable partition report from a frozen input artifact")
    evaluate_parser.add_argument("--protocol-id", required=True)
    evaluate_parser.add_argument("--partition", choices=("development", "selection", "independent_acceptance", "forward_shadow"), required=True)
    evaluate_parser.add_argument("--run-id", required=True)
    evaluate_parser.add_argument("--package-id", required=True)
    evaluate_parser.add_argument("--input-artifact", type=Path, required=True)
    args = parser.parse_args()

    store = HarnessReleaseStore(args.store, project_root=args.project_root)
    if args.command == "predeclare":
        result = store.predeclare_protocol(args.protocol_id, _read(args.protocol_json))
    elif args.command == "register-package":
        result = store.register_package(_read(args.package_json))
    else:
        result = store.evaluate_partition(
            protocol_id=args.protocol_id,
            partition=args.partition,
            run_id=args.run_id,
            package_id=args.package_id,
            input_artifact_path=args.input_artifact,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
