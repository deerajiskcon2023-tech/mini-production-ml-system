"""Compare current transaction features with a reference CSV using PSI and KS tests.

Example:
  python monitoring/drift_monitor.py --reference data/reference.csv --current data/current.csv
Both files must contain the same numeric feature columns. The target column, if
present, is excluded. This script reports drift; it does not retrain a model.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def population_stability_index(reference, current, bins=10):
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
    if len(ref) == 0 or len(cur) == 0:
        return None
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_pct = np.histogram(cur, bins=edges)[0] / len(cur)
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="Training/reference feature CSV")
    parser.add_argument("--current", required=True, help="Recent production feature CSV")
    parser.add_argument("--target", default="Class", help="Target column to exclude (default: Class)")
    parser.add_argument("--psi-warning", type=float, default=0.20)
    parser.add_argument("--psi-critical", type=float, default=0.30)
    parser.add_argument("--ks-alpha", type=float, default=0.01)
    parser.add_argument("--output", default="artifacts/drift_report.json")
    args = parser.parse_args()

    reference = pd.read_csv(args.reference)
    current = pd.read_csv(args.current)
    reference = reference.drop(columns=[args.target], errors="ignore")
    current = current.drop(columns=[args.target], errors="ignore")
    common = [c for c in reference.columns if c in current.columns]
    if not common:
        raise ValueError("No common feature columns found between reference and current data.")

    results = {}
    alerts = []
    for col in common:
        ref = pd.to_numeric(reference[col], errors="coerce").dropna()
        cur = pd.to_numeric(current[col], errors="coerce").dropna()
        if ref.empty or cur.empty:
            results[col] = {"status": "not_evaluated", "reason": "No numeric observations"}
            continue
        psi = population_stability_index(ref, cur)
        ks = ks_2samp(ref, cur)
        status = "critical" if psi is not None and psi >= args.psi_critical else (
            "warning" if (psi is not None and psi >= args.psi_warning) or ks.pvalue < args.ks_alpha else "ok"
        )
        results[col] = {
            "psi": psi,
            "ks_statistic": float(ks.statistic),
            "ks_p_value": float(ks.pvalue),
            "status": status,
            "reference_non_null": int(len(ref)),
            "current_non_null": int(len(cur)),
        }
        if status != "ok":
            alerts.append({"feature": col, "status": status})

    report = {
        "features_compared": len(common),
        "psi_warning_threshold": args.psi_warning,
        "psi_critical_threshold": args.psi_critical,
        "ks_significance_level": args.ks_alpha,
        "alerts": alerts,
        "feature_results": results,
        "recommended_action": "Investigate alerts and assess model performance before retraining; do not retrain on drift alone."
        if alerts else "No configured drift alert triggered.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
