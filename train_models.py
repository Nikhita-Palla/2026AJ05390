"""
Train and evaluate five classifiers on the UCI Online Shoppers Purchasing
Intention dataset.

Run:  python train_models.py

Outputs
-------
model/<name>.joblib   full sklearn Pipeline (preprocessing + classifier)
model/metrics.json    evaluation metrics on the held-out test split
test_data.csv         held-out test split (raw features + target) for the app
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from metrics_utils import compute_metrics

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "online_shoppers_intention.csv"
MODEL_DIR = ROOT / "model"
TEST_CSV = ROOT / "test_data.csv"

TARGET = "Revenue"
CATEGORICAL = ["Month", "VisitorType"]
NUMERIC = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
    "OperatingSystems", "Browser", "Region", "TrafficType", "Weekend",
]
RANDOM_STATE = 42


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Boolean columns are stored as TRUE/FALSE strings or bools -> 0/1
    for col in ("Weekend", TARGET):
        df[col] = df[col].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0})
    return df.drop_duplicates().reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )


def build_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=20, random_state=RANDOM_STATE
        ),
        "KNN": KNeighborsClassifier(n_neighbors=15, weights="distance"),
        "Naive Bayes": GaussianNB(),
        # SVC has no native probabilities; calibration provides them for AUC
        "Support Vector Machine": CalibratedClassifierCV(
            SVC(kernel="rbf", C=1.0, random_state=RANDOM_STATE), ensemble=False
        ),
    }


def file_name(model_name: str) -> str:
    return model_name.lower().replace(" ", "_") + ".joblib"


def main() -> None:
    df = load_dataset(DATA_PATH)
    print(f"Dataset shape after removing duplicates: {df.shape}")
    print(f"Class balance:\n{df[TARGET].value_counts(normalize=True).round(3)}\n")

    X = df[NUMERIC + CATEGORICAL]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    MODEL_DIR.mkdir(exist_ok=True)
    pd.concat([X_test, y_test], axis=1).to_csv(TEST_CSV, index=False)

    results = {}
    for name, clf in build_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        results[name] = compute_metrics(y_test, y_pred, y_prob)

        joblib.dump(pipe, MODEL_DIR / file_name(name))
        print(f"Saved {name:<24} -> model/{file_name(name)}")

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    table = pd.DataFrame(results).T.round(4)
    print("\nTest-set comparison:\n")
    print(table.to_string())


if __name__ == "__main__":
    main()
