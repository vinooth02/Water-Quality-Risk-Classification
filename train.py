"""
Water Quality Risk Classification - EDA, training, evaluation.
Usage:  python train.py data/water_potability.csv
Dataset: Kaggle "Water Potability" (Aditya Kadiwal), file water_potability.csv
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, balanced_accuracy_score, confusion_matrix,
                             ConfusionMatrixDisplay, RocCurveDisplay, classification_report)

SEED = 42
DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data/water_potability.csv")
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
MODEL_DIR = Path(".")


# ---------- 1. Load & audit ----------
df = pd.read_csv(DATA)
df.columns = [c.strip().lower() for c in df.columns]
TARGET = next(c for c in df.columns if c.startswith("potab"))
report = {
    "rows": int(len(df)), "columns": list(df.columns),
    "missing_per_column": df.isna().sum().to_dict(),
    "duplicate_rows": int(df.duplicated().sum()),
    "class_counts": df[TARGET].value_counts().to_dict(),
}
print(json.dumps(report, indent=2, default=str))

# Positive class = UNSAFE (1), because missing an unsafe sample is the costly error.
df["Unsafe"] = 1 - df[TARGET]
FEATURES = [c for c in df.columns if c not in (TARGET, "Unsafe")]

# ---------- 2. EDA plots ----------
df[FEATURES].hist(bins=30, figsize=(14, 9)); plt.tight_layout()
plt.savefig(OUT / "eda_distributions.png", dpi=120); plt.close()
plt.figure(figsize=(9, 7)); sns.heatmap(df[FEATURES + ["Unsafe"]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.tight_layout(); plt.savefig(OUT / "eda_correlation.png", dpi=120); plt.close()
df[FEATURES].plot(kind="box", subplots=True, layout=(3, 3), figsize=(12, 9), sharex=False, sharey=False)
plt.tight_layout(); plt.savefig(OUT / "eda_outliers.png", dpi=120); plt.close()
df["Unsafe"].value_counts().sort_index().plot(kind="bar", title="Class balance (0=Safe, 1=Unsafe)")
plt.tight_layout(); plt.savefig(OUT / "eda_class_balance.png", dpi=120); plt.close()

# ---------- 3. Split FIRST (avoid leakage), preprocessing lives inside the pipeline ----------
X, y = df[FEATURES], df["Unsafe"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

def pipe(clf):
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler()), ("clf", clf)])

candidates = {
    "Logistic Regression": (pipe(LogisticRegression(class_weight="balanced", max_iter=1000)),
                            {"clf__C": [0.01, 0.1, 1, 10]}),
    "KNN": (pipe(KNeighborsClassifier()),
            {"clf__n_neighbors": [5, 11, 21, 31], "clf__weights": ["uniform", "distance"]}),
    "Decision Tree": (pipe(DecisionTreeClassifier(class_weight="balanced", random_state=SEED)),
                      {"clf__max_depth": [3, 5, 7, 10], "clf__min_samples_leaf": [5, 20, 50]}),
}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

# ---------- 4. Compare models by cross-validated ROC-AUC (train set only) ----------
results, fitted = {}, {}
for name, (p, grid) in candidates.items():
    gs = GridSearchCV(p, grid, scoring="roc_auc", cv=cv, n_jobs=-1).fit(X_tr, y_tr)
    fitted[name] = gs.best_estimator_
    results[name] = {"cv_roc_auc": round(gs.best_score_, 4), "best_params": gs.best_params_}
    print(name, results[name])
best_name = max(results, key=lambda k: results[k]["cv_roc_auc"])
best = fitted[best_name]
print("Selected model:", best_name)

# ---------- 5. Final evaluation on untouched test set ----------
def evaluate(model):
    pred, prob = model.predict(X_te), model.predict_proba(X_te)[:, 1]
    return {"accuracy": accuracy_score(y_te, pred), "balanced_accuracy": balanced_accuracy_score(y_te, pred),
            "precision_unsafe": precision_score(y_te, pred, zero_division=0),
            "recall_unsafe": recall_score(y_te, pred), "f1_unsafe": f1_score(y_te, pred),
            "roc_auc": roc_auc_score(y_te, prob)}
test_metrics = {n: {k: round(v, 4) for k, v in evaluate(m).items()} for n, m in fitted.items()}
baseline = round(max(y_te.mean(), 1 - y_te.mean()), 4)
print(pd.DataFrame(test_metrics).T)
print("Majority-class baseline accuracy:", baseline)
print(classification_report(y_te, best.predict(X_te), target_names=["Safe", "Unsafe"]))

ConfusionMatrixDisplay(confusion_matrix(y_te, best.predict(X_te)), display_labels=["Safe", "Unsafe"]).plot()
plt.title(f"Confusion matrix - {best_name}"); plt.savefig(OUT / "confusion_matrix.png", dpi=120); plt.close()
fig, ax = plt.subplots()
for n, m in fitted.items(): RocCurveDisplay.from_estimator(m, X_te, y_te, name=n, ax=ax)
ax.plot([0, 1], [0, 1], "k--"); plt.savefig(OUT / "roc_curves.png", dpi=120); plt.close()

# ---------- 6. Feature relevance ----------
pi = permutation_importance(best, X_te, y_te, scoring="roc_auc", n_repeats=20, random_state=SEED)
imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values()
imp.plot(kind="barh", title="Permutation importance (drop in ROC-AUC)"); plt.tight_layout()
plt.savefig(OUT / "feature_importance.png", dpi=120); plt.close()

# ---------- 7. Save artefacts ----------
joblib.dump(best, MODEL_DIR / "model.joblib")
meta = {"features": FEATURES, "model": best_name, "class_labels": {"0": "Safe", "1": "Unsafe"},
        "ranges": {f: [float(X_tr[f].quantile(0.01)), float(X_tr[f].quantile(0.99)), float(X_tr[f].median())] for f in FEATURES},
        "test_metrics": test_metrics[best_name]}
json.dump(meta, open(MODEL_DIR / "meta.json", "w"), indent=2)
json.dump({"audit": report, "cv": results, "test": test_metrics, "baseline_accuracy": baseline,
           "importance": imp.to_dict()}, open(OUT / "results.json", "w"), indent=2, default=str)
print("Saved model + results.")
