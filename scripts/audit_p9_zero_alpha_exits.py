"""Rebuild no-forecast held-position events from frozen signals and P9 ledgers."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


VARIANTS = {
    "market_relative_legacy_rmse_soft8": "market_plus_relative_fusion",
    "market_relative_portfolio_mean_soft8": "market_plus_relative_fusion",
    "market_relative_portfolio_mean_hard20": "market_plus_relative_fusion",
    "market_relative_portfolio_lcb_hard20": "market_plus_relative_fusion",
    "trend_portfolio_mean_hard20": "trend",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--p9-run-dir",
        type=Path,
        default=project_dir / "experiments" / "p9_joint_economics_20260927_full_exit",
    )
    parser.add_argument(
        "--p8-study-dir",
        type=Path,
        default=project_dir / "experiments" / "post_p8_research_20260926" / "three_session_daily",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "experiments" / "p9_return_first_20260927" / "zero_alpha_exit_audit",
    )
    args = parser.parse_args()

    p9_dir = args.p9_run_dir.resolve()
    status = json.loads((p9_dir / "RUN_STATUS.json").read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        raise ValueError(f"P9 full replay must be complete; status is {status.get('status')!r}")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output_dir}")

    p8_dir = args.p8_study_dir.resolve()
    predictions_path = p8_dir / "prediction_decisions.jsonl.gz"
    forecasts = {}
    with gzip.open(predictions_path, "rt", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                forecasts[str(row["prediction_date"])] = row

    output_dir.mkdir(parents=True, exist_ok=False)
    summaries: dict[str, object] = {}
    for variant, method in VARIANTS.items():
        decisions_path = p9_dir / variant / "simulator_decisions.jsonl"
        decisions = _read_jsonl(decisions_path)
        events = []
        cap_violations = []
        for decision in decisions:
            day = str(decision["decision_date"])
            forecast = forecasts.get(day)
            if forecast is None:
                raise ValueError(f"No frozen forecast matches P9 decision {day}")
            method_returns = forecast["expected_returns_by_method"].get(method)
            if not isinstance(method_returns, dict):
                raise ValueError(f"{day} has no {method} return map")
            nav = float(decision["current_nav_cny"])
            for symbol_value, value in decision["current_holdings_cny"].items():
                symbol = str(symbol_value)
                market_value = float(value)
                if market_value <= 0 or symbol in method_returns:
                    continue
                current_weight = market_value / nav
                target_weight = float(decision["target_weights"].get(symbol, 0.0))
                if target_weight > current_weight + 1e-8:
                    cap_violations.append({
                        "decision_date": day,
                        "symbol": symbol,
                        "current_weight": current_weight,
                        "target_weight": target_weight,
                    })
                events.append({
                    "decision_date": day,
                    "execution_date": decision.get("execution_date"),
                    "symbol": symbol,
                    "current_market_value_cny": market_value,
                    "current_weight": current_weight,
                    "target_weight": target_weight,
                    "forecast_alpha_used": 0.0,
                    "treatment": "no selected-method forecast; zero alpha; hold or reduce only, capped at current weight",
                })
        if cap_violations:
            raise ValueError(f"{variant} increased a no-forecast holding: {cap_violations[:5]}")
        by_symbol = Counter(str(event["symbol"]) for event in events)
        event_payload = {
            "variant": variant,
            "return_method": method,
            "event_definition": "one event per decision date and currently held symbol when the selected method has no forecast for that symbol",
            "rule": "zero forecast alpha; the existing position remains in the account and may be held or reduced; target weight cannot exceed its current weight",
            "event_count": len(events),
            "distinct_symbol_count": len(by_symbol),
            "by_symbol": dict(sorted(by_symbol.items())),
            "target_weight_cap_violations": len(cap_violations),
            "events": events,
        }
        (output_dir / f"{variant}.json").write_text(
            json.dumps(event_payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        summaries[variant] = {
            key: event_payload[key]
            for key in (
                "return_method",
                "event_count",
                "distinct_symbol_count",
                "by_symbol",
                "target_weight_cap_violations",
            )
        }

    result = {
        "p9_run_dir": str(p9_dir),
        "p9_run_status": status.get("status"),
        "p9_python_pid": status.get("pid"),
        "p8_prediction_file": str(predictions_path),
        "p8_prediction_sha256": _sha256(predictions_path),
        "variant_decision_file_sha256": {
            variant: _sha256(p9_dir / variant / "simulator_decisions.jsonl")
            for variant in VARIANTS
        },
        "variants": summaries,
    }
    (output_dir / "zero_alpha_exit_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
