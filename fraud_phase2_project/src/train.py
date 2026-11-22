"""Train and evaluate the fraud model with chronological, leakage-aware splits."""
import argparse, json, os
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    precision_score, recall_score, f1_score, confusion_matrix
)

def choose_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    idx = int(np.nanargmax(f1[:-1]))
    return float(thresholds[idx]), float(precision[idx]), float(recall[idx]), float(f1[idx])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/creditcard.csv")
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()
    os.makedirs(args.artifacts, exist_ok=True)

    df = pd.read_csv(args.data)
    if "Class" not in df.columns:
        raise ValueError("Dataset must contain target column 'Class'.")
    if df.isna().any().any():
        raise ValueError("Missing values detected; define a training-only imputation strategy before training.")
    if not set(df["Class"].unique()).issubset({0, 1}):
        raise ValueError("Target Class must contain only 0 and 1.")

    X, y = df.drop(columns=["Class"]), df["Class"]
    n = len(df)
    i1, i2 = int(n * .60), int(n * .80)
    X_train, y_train = X.iloc[:i1], y.iloc[:i1]
    X_val, y_val = X.iloc[i1:i2], y.iloc[i1:i2]
    X_test, y_test = X.iloc[i2:], y.iloc[i2:]

    candidates = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000,
                                         solver="lbfgs", random_state=42))
        ]),
        "Histogram Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=120, learning_rate=.08, max_leaf_nodes=15,
            min_samples_leaf=30, l2_regularization=1.0,
            class_weight="balanced", random_state=42
        )
    }
    fitted, val_scores, val_ap = {}, {}, {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_val)[:, 1]
        fitted[name], val_scores[name] = model, scores
        val_ap[name] = float(average_precision_score(y_val, scores))

    selected_name = max(val_ap, key=val_ap.get)
    selected_model = fitted[selected_name]
    threshold, vprec, vrec, vf1 = choose_threshold(y_val, val_scores[selected_name])

    test_scores = selected_model.predict_proba(X_test)[:, 1]
    test_pred = (test_scores >= threshold).astype(int)
    cm = confusion_matrix(y_test, test_pred)
    report = {
        "selected_model": selected_name,
        "validation_pr_auc": val_ap,
        "threshold_from_validation": threshold,
        "validation_precision": vprec, "validation_recall": vrec, "validation_f1": vf1,
        "test_pr_auc": float(average_precision_score(y_test, test_scores)),
        "test_roc_auc": float(roc_auc_score(y_test, test_scores)),
        "test_precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "test_confusion_matrix_tn_fp_fn_tp": cm.tolist(),
        "split": {
            "train_rows": len(X_train), "train_fraud": int(y_train.sum()),
            "validation_rows": len(X_val), "validation_fraud": int(y_val.sum()),
            "test_rows": len(X_test), "test_fraud": int(y_test.sum())
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": int(df.isna().sum().sum())
    }
    joblib.dump(selected_model, os.path.join(args.artifacts, "fraud_model.joblib"))
    with open(os.path.join(args.artifacts, "model_metadata.json"), "w") as f:
        json.dump({"model_name": selected_name, "threshold": threshold,
                   "features": list(X.columns)}, f, indent=2)
    with open(os.path.join(args.artifacts, "evaluation_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
