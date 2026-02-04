# service/train_local.py
from __future__ import annotations

from pathlib import Path
import json
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    artifacts_dir = repo_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "iris_model.joblib"
    meta_path = artifacts_dir / "iris_model_meta.json"

    # Deterministic dataset split
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Simple, stable baseline model
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, random_state=42)),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))

    joblib.dump(model, model_path)

    meta = {
        "model_path": str(model_path),
        "test_accuracy": acc,
        "random_state": 42,
        "features": list(X.columns),
        "classes": [str(c) for c in iris.target_names],
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved model to: {model_path}")
    print(f"Saved metadata to: {meta_path}")
    print(f"Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
