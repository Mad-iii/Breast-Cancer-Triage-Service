import argparse
from pathlib import Path

from sklearn.datasets import load_breast_cancer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/breast_cancer.csv")
    out = Path(ap.parse_args().out)

    df = load_breast_cancer(as_frame=True).frame.copy()
    df.columns = [c.replace(" ", "_") for c in df.columns]

    n_features = df.shape[1] - 1
    if df.isnull().any().any():
        raise SystemExit("ERROR: dataset contains null values")
    if n_features != 30:
        raise SystemExit(f"ERROR: expected 30 features, got {n_features}")

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out} rows={len(df)} features={n_features}")


if __name__ == "__main__":
    main()