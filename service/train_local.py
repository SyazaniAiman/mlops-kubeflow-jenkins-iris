import os
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    out_path = os.getenv("MODEL_OUT", "/app/artifacts/iris_model.joblib")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X, y = load_iris(return_X_y=True)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)

    joblib.dump(model, out_path)
    print(f"Saved model to: {out_path}")


if __name__ == "__main__":
    main()

