"""Reproducibility check, leakage check, baselines and feature group ablation.

    python -m research.reproduce_and_ablate --out results/research

Uses the same data loader, features and model code as run_evaluation.py but never
touches the thesis document. The hybrid LSTM model is left out: it is slow and
TensorFlow training is not bit for bit repeatable on CPU, so it is reported as such.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold, StratifiedKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from src.config import RESULTS_DIR, SELECTED_FEATURES  # noqa: E402
from src.data import load_dataset, split_countries, tabular_split, temporal_split  # noqa: E402
from src.models import fit_logistic, fit_probit, fit_xgboost  # noqa: E402

FEATURE_GROUPS = {
    "debt": ["ExtDebt_GDP", "DebtServ_XGDP"],
    "domestic_macro": ["RealGDP_growth", "Inflation", "fiscal_balance", "Trade_openness"],
    "global": ["World_GDP_growth", "Oil_price_index", "US_FedFundsRate", "VIX"],
    "political": ["ICRG_political", "ElectionYear"],
}
assert sorted(sum(FEATURE_GROUPS.values(), [])) == sorted(SELECTED_FEATURES)


def logit_model():
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000))


def xgb_model(y):
    spw = max(1.0, (len(y) - y.sum()) / max(1, y.sum()))
    return XGBClassifier(max_depth=4, learning_rate=0.05, n_estimators=200, subsample=0.8,
                         colsample_bytree=0.8, scale_pos_weight=spw, random_state=42, n_jobs=2,
                         eval_metric="logloss")


def cv_auc(df, features, scheme: str, model: str, seed: int) -> list[float]:
    X = df[features].to_numpy(float)
    y = df["Default"].to_numpy(int)
    if scheme == "random_rows":
        splits = StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)
    elif scheme == "country_groups":
        countries = df["Country"].unique()
        rng = np.random.default_rng(seed)
        perm = {c: i for i, c in enumerate(rng.permutation(countries))}
        groups = df["Country"].map(perm).to_numpy() % 5
        splits = GroupKFold(5).split(X, y, groups)
    elif scheme == "expanding_time":
        splits = []
        for cut in [1995, 2000, 2005, 2010, 2015]:
            tr = np.where(df["Year"] <= cut)[0]
            te = np.where((df["Year"] > cut) & (df["Year"] <= cut + 5))[0]
            splits.append((tr, te))
    else:
        raise ValueError(scheme)
    out = []
    for tr, te in splits:
        if y[te].min() == y[te].max():
            continue
        m = logit_model() if model == "logistic" else xgb_model(y[tr])
        m.fit(X[tr], y[tr])
        out.append(float(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])))
    return out


def reproduce(df) -> dict:
    """Run the committed tabular models twice and compare with results/*.csv."""
    committed = {
        "country": pd.read_csv(RESULTS_DIR / "model_comparison_country_split.csv").set_index("model"),
        "temporal": pd.read_csv(RESULTS_DIR / "model_comparison_temporal_split.csv").set_index("model"),
    }
    tr_c, te_c = split_countries(df)
    splits = {
        "country": tabular_split(df, SELECTED_FEATURES, tr_c, te_c)[:4],
        "temporal": temporal_split(df, SELECTED_FEATURES, cutoff_year=2010)[:4],
    }
    rows = []
    for split, (Xtr, Xte, ytr, yte) in splits.items():
        runs = []
        for _ in range(2):
            res = {
                "Probit": fit_probit(Xtr, ytr, Xte, yte, SELECTED_FEATURES)["metrics"],
                "Logistic Regression": fit_logistic(Xtr, ytr, Xte, yte)["metrics"],
                "XGBoost": fit_xgboost(Xtr, ytr, Xte, yte, use_smote=True)["metrics"],
            }
            runs.append(res)
        for model in runs[0]:
            a, b = runs[0][model], runs[1][model]
            c = committed[split].loc[model]
            rows.append({
                "split": split, "model": model,
                "committed_roc_auc": float(c["roc_auc"]), "rerun1_roc_auc": round(float(a["roc_auc"]), 4),
                "rerun2_roc_auc": round(float(b["roc_auc"]), 4),
                "committed_f1": float(c["f1_default"]), "rerun1_f1": round(float(a["f1_default"]), 4),
                "rerun2_f1": round(float(b["f1_default"]), 4),
                "reruns_identical": bool(a["roc_auc"] == b["roc_auc"] and a["f1_default"] == b["f1_default"]),
                "matches_committed_auc_4dp": bool(round(float(a["roc_auc"]), 4) == round(float(c["roc_auc"]), 4)),
            })
    return {"rows": rows}


def persistence_baseline(df) -> dict:
    """Predict default this year from default status last year, on the temporal test window."""
    d = df.sort_values(["Country", "Year"]).copy()
    d["prev_default"] = d.groupby("Country")["Default"].shift(1).fillna(0)
    test = d[d["Year"] > 2010]
    train = d[d["Year"] <= 2010]
    X_cols = SELECTED_FEATURES
    lr = logit_model().fit(train[X_cols], train["Default"])
    p_lr = lr.predict_proba(test[X_cols])[:, 1]
    y = test["Default"].to_numpy()
    prev = test["prev_default"].to_numpy()
    out = {
        "test_rows": int(len(test)), "test_defaults": int(y.sum()),
        "share_of_defaults_preceded_by_default": float(prev[y == 1].mean()) if y.sum() else None,
        "persistence_roc_auc": float(roc_auc_score(y, prev)),
        "persistence_pr_auc": float(average_precision_score(y, prev)),
        "logistic_roc_auc": float(roc_auc_score(y, p_lr)),
        "logistic_pr_auc": float(average_precision_score(y, p_lr)),
    }
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            diffs.append(roc_auc_score(y[i], p_lr[i]) - roc_auc_score(y[i], prev[i]))
    out["logistic_minus_persistence_auc_95ci"] = [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]
    # Is the logistic model better than a coin? Bootstrap CI of its AUC.
    aucs = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            aucs.append(roc_auc_score(y[i], p_lr[i]))
    out["logistic_auc_95ci"] = [float(np.quantile(aucs, 0.025)), float(np.quantile(aucs, 0.975))]
    # McNemar at the 0.5 threshold between logistic and an always-no-default rule
    a = (p_lr >= 0.5) == y
    b = np.zeros_like(y) == y
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    out["mcnemar_logistic_vs_always_no_default"] = {
        "logistic_right_rule_wrong": n01, "logistic_wrong_rule_right": n10,
        "p_value": float(binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0}
    return out


def decade_slices(df) -> list[dict]:
    """Out of fold logistic predictions by country groups, then AUC per decade."""
    X = df[SELECTED_FEATURES].to_numpy(float)
    y = df["Default"].to_numpy(int)
    oof = np.zeros(len(df))
    countries = df["Country"].unique()
    perm = {c: i for i, c in enumerate(np.random.default_rng(0).permutation(countries))}
    groups = df["Country"].map(perm).to_numpy() % 5
    for tr, te in GroupKFold(5).split(X, y, groups):
        oof[te] = logit_model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    rows = []
    for dec, g in df.assign(p=oof, decade=(df["Year"] // 10) * 10).groupby("decade"):
        rows.append({"decade": int(dec), "rows": len(g), "defaults": int(g.Default.sum()),
                     "default_rate": float(g.Default.mean()),
                     "roc_auc": float(roc_auc_score(g.Default, g.p)) if g.Default.nunique() == 2 else None})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/research"))
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = load_dataset().dropna(subset=SELECTED_FEATURES + ["Default"]).reset_index(drop=True)
    report = {"rows": len(df), "countries": int(df.Country.nunique()), "default_rate": float(df.Default.mean()),
              "environment": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "xgboost",
                                                        "imbalanced-learn", "statsmodels"]}
              | {"python": platform.python_version(), "device": "cpu"}}

    report["reproduction"] = reproduce(df)

    leak = []
    for scheme in ["random_rows", "country_groups", "expanding_time"]:
        for model in ["logistic", "xgboost"]:
            seeds = range(args.seeds) if scheme != "expanding_time" else [0]
            aucs = [a for s in seeds for a in cv_auc(df, SELECTED_FEATURES, scheme, model, s)]
            leak.append({"scheme": scheme, "model": model, "folds": len(aucs),
                         "mean_auc": float(np.mean(aucs)), "std_auc": float(np.std(aucs, ddof=1))})
    report["split_schemes"] = leak

    abl = []
    for model in ["logistic", "xgboost"]:
        base = [a for s in range(args.seeds) for a in cv_auc(df, SELECTED_FEATURES, "country_groups", model, s)]
        abl.append({"model": model, "variant": "all features", "mean_auc": float(np.mean(base)),
                    "std_auc": float(np.std(base, ddof=1)), "delta_vs_all": 0.0})
        for g, cols in FEATURE_GROUPS.items():
            drop = [f for f in SELECTED_FEATURES if f not in cols]
            a = [x for s in range(args.seeds) for x in cv_auc(df, drop, "country_groups", model, s)]
            only = [x for s in range(args.seeds) for x in cv_auc(df, cols, "country_groups", model, s)]
            abl.append({"model": model, "variant": f"drop {g}", "mean_auc": float(np.mean(a)),
                        "std_auc": float(np.std(a, ddof=1)), "delta_vs_all": float(np.mean(a) - np.mean(base))})
            abl.append({"model": model, "variant": f"only {g}", "mean_auc": float(np.mean(only)),
                        "std_auc": float(np.std(only, ddof=1)), "delta_vs_all": float(np.mean(only) - np.mean(base))})
    pd.DataFrame(abl).to_csv(out / "feature_group_ablation.csv", index=False)
    report["feature_group_ablation"] = {"cv": f"5 fold country grouped CV x {args.seeds} country shuffles",
                                        "groups": FEATURE_GROUPS, "rows": abl}

    report["persistence_baseline_temporal"] = persistence_baseline(df)
    report["decade_slices"] = decade_slices(df)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    labels = [f"{r['scheme']}\n{r['model']}" for r in leak]
    ax.bar(range(len(leak)), [r["mean_auc"] for r in leak], yerr=[r["std_auc"] for r in leak], capsize=3)
    ax.axhline(0.5, color="grey", ls="--")
    ax.set_xticks(range(len(leak)), labels, fontsize=7)
    ax.set_ylabel("ROC AUC")
    ax.set_title("Same models, different split schemes")
    fig.tight_layout()
    fig.savefig(out / "split_schemes.png", dpi=120)
    plt.close(fig)

    report["runtime_seconds"] = round(time.time() - t0, 1)
    (out / "research_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ["reproduction", "split_schemes", "persistence_baseline_temporal"]}, indent=1))


if __name__ == "__main__":
    main()
