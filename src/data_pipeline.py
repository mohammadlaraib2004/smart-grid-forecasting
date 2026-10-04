"""
data_pipeline.py
----------------
Handles all data loading, temporal feature extraction,
feature scaling, and correlation utilities.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DEFAULT_CSV = os.path.join(DATA_DIR, "steel_industry_data.csv")

# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------
TARGET = "Usage_kWh"

NUMERIC_FEATURES = [
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM",
]

CATEGORICAL_FEATURES = ["WeekStatus", "Day_of_week", "Load_Type"]

TEMPORAL_FEATURES = ["hour", "minute", "day_of_week_num", "shift"]

DISPLAY_NAMES = {
    "Lagging_Current_Reactive.Power_kVarh": "Lagging Reactive Power (kVArh)",
    "Leading_Current_Reactive_Power_kVarh": "Leading Reactive Power (kVArh)",
    "CO2(tCO2)": "CO₂ Emissions (tCO₂)",
    "Lagging_Current_Power_Factor": "Lagging Power Factor",
    "Leading_Current_Power_Factor": "Leading Power Factor",
    "NSM": "NSM (seconds since midnight)",
}


# ---------------------------------------------------------------------------
# Loading & cleaning
# ---------------------------------------------------------------------------

def load_data(path: str = DEFAULT_CSV) -> pd.DataFrame:
    """Load the Steel Industry CSV and parse timestamps."""
    df = pd.read_csv(path)
    # Strip BOM / whitespace from column names
    df.columns = df.columns.str.strip().str.lstrip("\ufeff")
    # Parse date column robustly
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Temporal Feature Extraction  (Phase 1 of workflow)
# ---------------------------------------------------------------------------

def _assign_shift(hour: int) -> str:
    """Map hour → industrial operating shift label."""
    if 6 <= hour < 14:
        return "Morning"
    elif 14 <= hour < 22:
        return "Afternoon"
    else:
        return "Night"


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-of-day, day-of-week, and industrial shift features
    from the 15-minute timestamp column.
    """
    df = df.copy()
    df["hour"] = df["date"].dt.hour
    df["minute"] = df["date"].dt.minute
    df["day_of_week_num"] = df["date"].dt.dayofweek          # 0=Mon … 6=Sun
    df["month"] = df["date"].dt.month
    df["shift"] = df["hour"].apply(_assign_shift)

    # Encode categorical columns with one-hot encoding
    df = pd.get_dummies(
        df,
        columns=["WeekStatus", "Day_of_week", "Load_Type", "shift"],
        drop_first=False,
    )
    return df


# ---------------------------------------------------------------------------
# Feature list after encoding
# ---------------------------------------------------------------------------

def get_feature_columns(df: pd.DataFrame) -> list:
    """Return all feature columns (numeric + encoded categoricals)."""
    exclude = {"date", TARGET}
    return [c for c in df.columns if c not in exclude]


# ---------------------------------------------------------------------------
# Feature Scaling  (Phase 2 of workflow)
# ---------------------------------------------------------------------------

def scale_features(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit a StandardScaler on training data and transform both splits.
    Returns (X_train_scaled, X_test_scaled, fitted_scaler).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# ---------------------------------------------------------------------------
# Correlation helper  (for Phase 2 analysis)
# ---------------------------------------------------------------------------

def compute_correlations(df: pd.DataFrame) -> pd.Series:
    """
    Return Pearson correlations of NUMERIC_FEATURES + temporal columns
    against Usage_kWh, sorted by absolute value descending.
    """
    cols = [c for c in NUMERIC_FEATURES + TEMPORAL_FEATURES if c in df.columns]
    corr = df[cols + [TARGET]].corr()[TARGET].drop(TARGET)
    return corr.reindex(corr.abs().sort_values(ascending=False).index)


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame):
    """
    Run the complete preprocessing pipeline:
      1. Temporal feature extraction
      2. Identify feature & target columns
    Returns (X, y, processed_df, feature_names).
    """
    processed = extract_temporal_features(df)
    feature_cols = get_feature_columns(processed)
    X = processed[feature_cols].values.astype(float)
    y = processed[TARGET].values
    return X, y, processed, feature_cols
