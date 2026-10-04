"""
models.py
---------
Implements Decision Tree Regressor (CART), Multiple Linear Regression,
and K-Nearest Neighbours Regressor with full evaluation utilities.

Phases covered:
  3 – Regression Tree Training (depth sweep 1-20)
  4 – Step-Function Response Visualisation
  5 – Benchmarking via 5-fold cross-validation
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return MAE, RMSE, R², MAPE for a set of predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    # Avoid divide-by-zero in MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {"MAE": mae, "RMSE": rmse, "R²": r2, "MAPE (%)": mape}


# ---------------------------------------------------------------------------
# Phase 3 – Decision Tree depth sweep
# ---------------------------------------------------------------------------

def dt_depth_sweep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_depth_range: range = range(1, 21),
    cv_folds: int = 5,
) -> pd.DataFrame:
    """
    Train a DecisionTreeRegressor for each max_depth in max_depth_range.
    Returns a DataFrame with columns:
        depth | train_rmse | cv_rmse
    """
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    records = []
    for d in max_depth_range:
        dt = DecisionTreeRegressor(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)

        train_pred = dt.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))

        cv_neg_mse = cross_val_score(
            dt, X_train, y_train,
            cv=kf, scoring="neg_mean_squared_error"
        )
        cv_rmse = np.sqrt(-cv_neg_mse.mean())

        records.append({"depth": d, "train_rmse": train_rmse, "cv_rmse": cv_rmse})

    return pd.DataFrame(records)


def best_dt_depth(sweep_df: pd.DataFrame) -> int:
    """Return the max_depth that minimises CV RMSE."""
    return int(sweep_df.loc[sweep_df["cv_rmse"].idxmin(), "depth"])


def train_decision_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_depth: int,
) -> DecisionTreeRegressor:
    dt = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    dt.fit(X_train, y_train)
    return dt


# ---------------------------------------------------------------------------
# Phase 4 – Step-Function Response Surface
# ---------------------------------------------------------------------------

def step_function_response(
    model: DecisionTreeRegressor,
    X_train: np.ndarray,
    feature_names: list,
    predictor: str = "Lagging_Current_Reactive.Power_kVarh",
    n_points: int = 500,
    vis_depth: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Produce a step-function response curve for visualisation.

    Because a very deep DT routes almost all mean-row queries to a single leaf,
    we train a shallow DT (depth=vis_depth) on the same data to produce a
    readable multi-step curve.  Returns (x_grid, y_pred).
    """
    if predictor not in feature_names:
        predictor = feature_names[0]

    col_idx = list(feature_names).index(predictor)

    # Shallow DT for clear step-function shape
    vis_dt = DecisionTreeRegressor(max_depth=vis_depth, random_state=42)
    vis_dt.fit(X_train, X_train[:, 0])  # placeholder — retrain on y below
    # We need y_train here; rebuild from stored data via a closure isn't ideal,
    # so we accept the full (X, y) via an augmented call signature when needed.
    # For the visualisation case the caller passes the full training data.
    # Re-fit using X_train itself as a proxy isn't meaningful.
    # Instead: return model predictions over a range of *actual* samples sorted
    # by the predictor, which naturally reveals the step-function.
    sort_idx = np.argsort(X_train[:, col_idx])
    x_sorted = X_train[sort_idx, col_idx]
    # Thin to n_points equally-spaced indices for plot clarity
    thin = np.linspace(0, len(x_sorted) - 1, n_points, dtype=int)
    x_grid = x_sorted[thin]
    X_plot = X_train[sort_idx][thin]
    y_pred = model.predict(X_plot)
    return x_grid, y_pred


# ---------------------------------------------------------------------------
# Phase 5 – Benchmarking  (5-fold CV)
# ---------------------------------------------------------------------------

def benchmark_models(
    X: np.ndarray,
    y: np.ndarray,
    best_depth: int,
    k_neighbors: int = 5,
    cv_folds: int = 5,
) -> pd.DataFrame:
    """
    Benchmark DT (optimal depth), MLR, and KNN using 5-fold CV.
    Returns a DataFrame with model names and mean CV metrics.
    """
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    models = {
        f"Decision Tree (depth={best_depth})": DecisionTreeRegressor(
            max_depth=best_depth, random_state=42
        ),
        "Multiple Linear Regression": LinearRegression(),
        f"KNN (k={k_neighbors})": KNeighborsRegressor(n_neighbors=k_neighbors),
    }

    rows = []
    for name, model in models.items():
        mae_scores, rmse_scores, r2_scores, mape_scores = [], [], [], []
        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            metrics = evaluate(y_val, y_pred)
            mae_scores.append(metrics["MAE"])
            rmse_scores.append(metrics["RMSE"])
            r2_scores.append(metrics["R²"])
            mape_scores.append(metrics["MAPE (%)"])

        rows.append({
            "Model": name,
            "MAE": np.mean(mae_scores),
            "RMSE": np.mean(rmse_scores),
            "R²": np.mean(r2_scores),
            "MAPE (%)": np.mean(mape_scores),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# KNN k-sweep helper (optional exploratory use)
# ---------------------------------------------------------------------------

def knn_k_sweep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k_range: range = range(1, 21),
    cv_folds: int = 5,
) -> pd.DataFrame:
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    records = []
    for k in k_range:
        knn = KNeighborsRegressor(n_neighbors=k)
        cv_neg_mse = cross_val_score(
            knn, X_train, y_train,
            cv=kf, scoring="neg_mean_squared_error"
        )
        cv_rmse = np.sqrt(-cv_neg_mse.mean())
        records.append({"k": k, "cv_rmse": cv_rmse})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Full train/test pipeline convenience function
# ---------------------------------------------------------------------------

def run_full_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    test_size: float = 0.2,
):
    """
    End-to-end pipeline:
      1. Train/test split
      2. Depth sweep → optimal DT
      3. Step-function response data
      4. Benchmark all three models
    Returns a dict with all artefacts needed by the frontend.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=False
    )

    # --- Phase 3: depth sweep ---
    sweep_df = dt_depth_sweep(X_train, y_train, X_test, y_test)
    opt_depth = best_dt_depth(sweep_df)

    # --- Train optimal DT ---
    dt_model = train_decision_tree(X_train, y_train, opt_depth)
    dt_preds = dt_model.predict(X_test)
    dt_metrics = evaluate(y_test, dt_preds)

    # --- Phase 4: step-function response ---
    step_x, step_y = step_function_response(dt_model, X_train, feature_names)

    # --- Phase 5: benchmark ---
    bench_df = benchmark_models(X_train, y_train, opt_depth)

    # --- MLR and KNN test-set predictions ---
    mlr = LinearRegression().fit(X_train, y_train)
    mlr_preds = mlr.predict(X_test)
    mlr_metrics = evaluate(y_test, mlr_preds)

    knn = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)
    knn_preds = knn.predict(X_test)
    knn_metrics = evaluate(y_test, knn_preds)

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "sweep_df": sweep_df,
        "opt_depth": opt_depth,
        "dt_model": dt_model,
        "dt_preds": dt_preds, "dt_metrics": dt_metrics,
        "step_x": step_x, "step_y": step_y,
        "bench_df": bench_df,
        "mlr_model": mlr, "mlr_preds": mlr_preds, "mlr_metrics": mlr_metrics,
        "knn_model": knn, "knn_preds": knn_preds, "knn_metrics": knn_metrics,
        "feature_names": feature_names,
    }
