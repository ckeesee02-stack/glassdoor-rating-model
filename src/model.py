"""
Glassdoor Employee Sentiment Model

Fits the 3-variable model (Organizational Climate, Career Opportunities,
Compensation & Benefits) predicting a company's Overall Glassdoor rating,
and writes every table used in the README to results/.

Usage:
    python src/model.py --data data/glassdoor_sp500_medians.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.model_selection import RepeatedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
TARGET = "Overall"
RAW_FEATURES = ["Comp", "WLB", "Culture", "SeniorMgmt", "Career", "DI"]
CLIMATE_ITEMS = ["WLB", "Culture", "SeniorMgmt", "DI"]
FEATURES = ["Climate", "Career", "Comp"]

LABELS = {
    "Overall": "Overall Rating",
    "Comp": "Compensation & Benefits",
    "WLB": "Work/Life Balance",
    "Culture": "Culture & Values",
    "SeniorMgmt": "Senior Management",
    "Career": "Career Opportunities",
    "DI": "Diversity & Inclusion",
    "Climate": "Organizational Climate",
}


# ---------------------------------------------------------------- helpers
def load_data(path):
    df = pd.read_csv(path)
    missing = [c for c in [TARGET] + RAW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. See data/README.md.")
    df = df.dropna(subset=[TARGET] + RAW_FEATURES).reset_index(drop=True)
    df["Climate"] = df[CLIMATE_ITEMS].mean(axis=1)
    return df


def cronbach_alpha(items):
    k = items.shape[1]
    return k / (k - 1) * (1 - items.var(ddof=1).sum() / items.sum(axis=1).var(ddof=1))


def vif_table(df, cols):
    X = sm.add_constant(df[cols])
    rows = []
    for i, c in enumerate(cols, start=1):
        r2 = sm.OLS(X.iloc[:, i], X.drop(columns=c)).fit().rsquared
        rows.append({"variable": LABELS[c], "VIF": 1 / (1 - r2)})
    return pd.DataFrame(rows)


def standardize(frame):
    return (frame - frame.mean()) / frame.std(ddof=1)


def relative_weights(X, y):
    """Johnson (2000) relative weights. Returns raw weights that sum to R²."""
    R = np.corrcoef(np.column_stack([X, y]).T)
    Rxx, rxy = R[:-1, :-1], R[:-1, -1]
    evals, evecs = np.linalg.eigh(Rxx)
    Lam = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
    beta = np.linalg.solve(Lam, rxy)
    return (Lam**2) @ (beta**2)


def bootstrap_rwa(X, y, n_boot=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = np.empty((n_boot, X.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        w = relative_weights(X[idx], y[idx])
        draws[b] = w / w.sum()
    return np.percentile(draws, [2.5, 97.5], axis=0)


# ---------------------------------------------------------------- analysis
def main(data_path, out_dir, n_boot):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_data(data_path)
    n = len(df)
    print(f"Loaded {n} companies")

    # 1. EDA
    desc = df[[TARGET] + RAW_FEATURES].describe().T
    desc.index = [LABELS[c] for c in desc.index]
    desc.round(3).to_csv(out / "descriptives.csv")
    corr = df[[TARGET] + RAW_FEATURES].corr()
    corr.index = corr.columns = [LABELS[c] for c in corr.columns]
    corr.round(3).to_csv(out / "correlations.csv")

    # 2. Collinearity: 6 raw features vs. 3-variable model
    vif_before = vif_table(df, RAW_FEATURES).assign(model="6 original features")
    vif_after = vif_table(df, FEATURES).assign(model="3-variable model")
    vif = pd.concat([vif_before, vif_after])
    vif.round(2).to_csv(out / "vif.csv", index=False)
    alpha = cronbach_alpha(df[CLIMATE_ITEMS])

    # 3. Standardized OLS on the full sample
    Z, zy = standardize(df[FEATURES]), standardize(df[TARGET])
    ols = sm.OLS(zy, sm.add_constant(Z)).fit()
    ci = ols.conf_int()
    coefs = pd.DataFrame(
        {
            "variable": [LABELS[c] for c in FEATURES],
            "std_beta": ols.params[FEATURES].values,
            "ci_low": ci.loc[FEATURES, 0].values,
            "ci_high": ci.loc[FEATURES, 1].values,
            "t": ols.tvalues[FEATURES].values,
            "p": ols.pvalues[FEATURES].values,
        }
    )
    raw = LinearRegression().fit(df[FEATURES], df[TARGET])
    coefs["raw_coef_stars"] = raw.coef_
    coefs.round(4).to_csv(out / "coefficients.csv", index=False)

    # 4. Relative weights with bootstrap CIs
    X, y = df[FEATURES].to_numpy(), df[TARGET].to_numpy()
    rw = relative_weights(X, y)
    lo, hi = bootstrap_rwa(X, y, n_boot=n_boot)
    rwa = pd.DataFrame(
        {
            "variable": [LABELS[c] for c in FEATURES],
            "raw_weight": rw,
            "pct_of_R2": rw / rw.sum(),
            "ci_low": lo,
            "ci_high": hi,
        }
    )
    rwa.round(4).to_csv(out / "relative_weights.csv", index=False)

    # 5. 80/20 train/test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=SEED
    )
    lm = LinearRegression().fit(X_tr, y_tr)
    train_r2, test_r2 = lm.score(X_tr, y_tr), lm.score(X_te, y_te)

    # 6. Elastic net check (scaled inside a pipeline, tuned on train only)
    enet = make_pipeline(
        StandardScaler(),
        ElasticNetCV(l1_ratio=[0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0], cv=10, random_state=SEED),
    ).fit(X_tr, y_tr)
    en = enet[-1]
    en_test_r2 = enet.score(X_te, y_te)

    # 7. Repeated k-fold CV (5 folds x 10 repeats = 50 fits)
    cv = cross_validate(
        LinearRegression(),
        df[FEATURES],
        df[TARGET],
        cv=RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEED),
        scoring=("r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"),
    )
    folds = pd.DataFrame(
        {
            "fold": range(1, 51),
            "r2": cv["test_r2"],
            "mae_stars": -cv["test_neg_mean_absolute_error"],
            "rmse_stars": -cv["test_neg_root_mean_squared_error"],
        }
    )
    folds.round(4).to_csv(out / "cv_folds.csv", index=False)

    # 8. Summary
    summary = {
        "n_companies": n,
        "cronbach_alpha_climate": alpha,
        "max_vif_6_features": vif_before["VIF"].max(),
        "max_vif_3_features": vif_after["VIF"].max(),
        "r2_full_sample": ols.rsquared,
        "adj_r2_full_sample": ols.rsquared_adj,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "cv_r2_mean": folds["r2"].mean(),
        "cv_r2_min": folds["r2"].min(),
        "cv_r2_max": folds["r2"].max(),
        "cv_mae_stars": folds["mae_stars"].mean(),
        "enet_alpha": en.alpha_,
        "enet_l1_ratio": en.l1_ratio_,
        "enet_test_r2": en_test_r2,
    }
    pd.Series(summary).round(4).to_csv(out / "model_summary.csv", header=["value"])

    print("\nStandardized coefficients")
    print(coefs[["variable", "std_beta", "ci_low", "ci_high"]].round(3).to_string(index=False))
    print("\nRelative weights (share of R²)")
    print(rwa[["variable", "pct_of_R2", "ci_low", "ci_high"]].round(3).to_string(index=False))
    print("\nSummary")
    print(pd.Series(summary).round(3).to_string())
    print(f"\nTables written to {out}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/glassdoor_sp500_medians.csv")
    p.add_argument("--out", default="results")
    p.add_argument("--n-boot", type=int, default=2000)
    a = p.parse_args()
    main(a.data, a.out, a.n_boot)
