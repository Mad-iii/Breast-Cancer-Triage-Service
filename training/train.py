import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MIN_TEST_RECALL = 0.95


def build_models(p):
    seed = p["seed"]
    m = p["models"]
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=seed, **m["logistic_regression"])),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=seed, **m["random_forest"])),
        ]),
    }


def test_metrics(model, X, y):
    proba = model.predict_proba(X)[:, 1]          # class 1 = malignant
    pred = (proba >= 0.5).astype(int)
    return {
        "malignant_recall": float(recall_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, proba)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-version", default="dev")
    ap.add_argument("--params", default="params.yaml")
    a = ap.parse_args()

    p = yaml.safe_load(open(a.params))
    seed = p["seed"]

    df = pd.read_csv(a.data_path)
    features = [c for c in df.columns if c != "target"]
    X = df[features].to_numpy()
    y = (df["target"] == 0).astype(int).to_numpy()   # 1 = malignant (positive class)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=p["test_size"], stratify=y, random_state=seed)

    cv = StratifiedKFold(n_splits=p["cv_folds"], shuffle=True, random_state=seed)
    scoring = {"recall": "recall", "accuracy": "accuracy", "roc_auc": "roc_auc"}

    results = {}
    models = build_models(p)
    for name, model in models.items():
        cvr = cross_validate(model, X_tr, y_tr, cv=cv, scoring=scoring)
        cv_m = {
            "malignant_recall": float(np.mean(cvr["test_recall"])),
            "accuracy": float(np.mean(cvr["test_accuracy"])),
            "roc_auc": float(np.mean(cvr["test_roc_auc"])),
        }
        model.fit(X_tr, y_tr)
        results[name] = {"cv": cv_m, "test": test_metrics(model, X_te, y_te)}

    best = max(results, key=lambda n: (results[n]["cv"]["malignant_recall"],
                                       results[n]["cv"]["roc_auc"]))

    print(f"{'model':22s} {'cv_recall':>9s} {'cv_acc':>7s} {'cv_auc':>7s} | "
          f"{'te_recall':>9s} {'te_acc':>7s} {'te_auc':>7s}")
    for n, r in results.items():
        star = " <== selected" if n == best else ""
        print(f"{n:22s} {r['cv']['malignant_recall']:9.4f} {r['cv']['accuracy']:7.4f} "
              f"{r['cv']['roc_auc']:7.4f} | {r['test']['malignant_recall']:9.4f} "
              f"{r['test']['accuracy']:7.4f} {r['test']['roc_auc']:7.4f}{star}")

    if results[best]["test"]["malignant_recall"] < MIN_TEST_RECALL:
        print(f"FAIL: test malignant recall below {MIN_TEST_RECALL}", file=sys.stderr)
        sys.exit(1)

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(models[best], out / "model.joblib")
    card = {
        "model_version": a.model_version,
        "model_family": best,
        "feature_names": features,
        "positive_class": "malignant",
        "metrics": {"cv": results[best]["cv"], "test": results[best]["test"]},
        "seed": seed,
        "sklearn_version": sklearn.__version__,
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2))
    (out / "metrics.json").write_text(json.dumps(
        {"selected": best, "candidates": results}, indent=2))
    print(f"Saved model + card to {out}")


if __name__ == "__main__":
    main()