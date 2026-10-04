"""
app.py  –  Smart Grid Electrical Energy Consumption Demand Forecasting
Streamlit multi-page dashboard.

Pages
─────
1. 🏠  Overview          – dataset summary & EDA statistics
2. 📊  EDA              – distributions, time-series, correlations
3. ⚙️  Preprocessing    – temporal features, correlation heatmap, scaling
4. 🌳  Decision Tree    – depth sweep, optimal DT metrics
5. 📈  Step-Function    – DT step-function response surface
6. 🏆  Benchmarking     – 5-fold CV comparison of all three models
7. 🔮  Predict          – live single-record prediction
"""

import sys, os
# Works locally and on Streamlit Cloud (both place app.py at repo root)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from data_pipeline import (
    load_data, build_feature_matrix, scale_features,
    compute_correlations, NUMERIC_FEATURES, TARGET, DISPLAY_NAMES,
)
from models import step_function_response


# ============================================================
#  Page config
# ============================================================
st.set_page_config(
    page_title="Smart Grid Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
#  Custom CSS
# ============================================================
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0f172a; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
.metric-card {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 1rem 1.4rem;
    text-align: center;
}
.metric-card h2 { margin: 0; font-size: 1.8rem; color: #1e40af; }
.metric-card p  { margin: 0; color: #64748b; font-size: 0.85rem; }
.section-header {
    font-size: 1.3rem; font-weight: 700; color: #1e293b;
    border-left: 4px solid #3b82f6; padding-left: 10px;
    margin-top: 1.5rem; margin-bottom: 0.8rem;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
#  Data + model cache
# ============================================================
DATA_PATH = os.path.join(_HERE, "data", "steel_industry_data.csv")

@st.cache_data(show_spinner="Loading dataset …")
def get_raw_data():
    return load_data(DATA_PATH)

@st.cache_data(show_spinner="Building feature matrix …")
def get_features(_df):
    return build_feature_matrix(_df)

@st.cache_data(show_spinner="Running ML pipeline (this takes ~1 min on first run) …")
def get_pipeline(_X, _y, _feature_names):
    from sklearn.model_selection import train_test_split
    from models import (
        dt_depth_sweep, best_dt_depth, train_decision_tree,
        step_function_response, benchmark_models, evaluate,
    )
    from sklearn.linear_model import LinearRegression
    from sklearn.neighbors import KNeighborsRegressor

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        _X, _y, test_size=0.2, random_state=42, shuffle=False
    )
    X_train_s, X_test_s, scaler = scale_features(X_train_raw, X_test_raw)
    feat_list = list(_feature_names)

    # Depth sweep & optimal DT
    sweep_df = dt_depth_sweep(X_train_s, y_train, X_test_s, y_test)
    opt_depth = best_dt_depth(sweep_df)
    dt_model  = train_decision_tree(X_train_s, y_train, opt_depth)
    dt_preds  = dt_model.predict(X_test_s)
    dt_metrics = evaluate(y_test, dt_preds)

    # Step-function response
    step_x, step_y = step_function_response(dt_model, X_train_s, feat_list)

    # Benchmark
    bench_df = benchmark_models(X_train_s, y_train, opt_depth)

    # MLR
    mlr = LinearRegression().fit(X_train_s, y_train)
    mlr_preds = mlr.predict(X_test_s)
    mlr_metrics = evaluate(y_test, mlr_preds)

    # KNN
    knn = KNeighborsRegressor(n_neighbors=5).fit(X_train_s, y_train)
    knn_preds = knn.predict(X_test_s)
    knn_metrics = evaluate(y_test, knn_preds)

    return {
        "X_train": X_train_s, "X_test": X_test_s,
        "X_train_raw": X_train_raw,
        "y_train": y_train, "y_test": y_test,
        "sweep_df": sweep_df, "opt_depth": opt_depth,
        "dt_model": dt_model, "dt_preds": dt_preds, "dt_metrics": dt_metrics,
        "step_x": step_x, "step_y": step_y,
        "bench_df": bench_df,
        "mlr_model": mlr, "mlr_preds": mlr_preds, "mlr_metrics": mlr_metrics,
        "knn_model": knn, "knn_preds": knn_preds, "knn_metrics": knn_metrics,
        "feature_names": feat_list,
        "scaler": scaler,
    }

df_raw = get_raw_data()
X, y, df_proc, feat_names = get_features(df_raw)
pipe = get_pipeline(X, y, tuple(feat_names))


# ============================================================
#  Sidebar navigation
# ============================================================
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Electric_lightning_bolt.svg/120px-Electric_lightning_bolt.svg.png",
    width=50,
)
st.sidebar.title("⚡ Smart Grid\nForecasting")
st.sidebar.markdown("---")

PAGES = {
    "🏠  Overview":         "overview",
    "📊  EDA":              "eda",
    "⚙️  Preprocessing":   "preprocessing",
    "🌳  Decision Tree":    "decision_tree",
    "📈  Step-Function":    "step_function",
    "🏆  Benchmarking":     "benchmarking",
    "🔮  Predict":          "predict",
}

selected = st.sidebar.radio("Navigate to", list(PAGES.keys()))
page = PAGES[selected]

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small>Dataset: Steel Industry Energy Consumption<br>"
    "UCI / Sathishkumar et al. · 35,040 records</small>",
    unsafe_allow_html=True,
)


# ============================================================
#  Helper: metric cards row
# ============================================================
def metric_row(items: list[tuple[str, str]]):
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.markdown(
            f'<div class="metric-card"><h2>{value}</h2><p>{label}</p></div>',
            unsafe_allow_html=True,
        )

def section(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


# ============================================================
#  PAGE 1 – Overview
# ============================================================
if page == "overview":
    st.title("⚡ Smart Grid Electrical Energy Consumption Demand Forecasting")
    st.markdown(
        """
        **Problem Domain:** Power Systems & Industrial Energy Management  
        **Target:** Forecast `Usage_kWh` (active energy consumption) from sensor readings  
        **Algorithms:** Decision Tree Regressor (CART) · Multiple Linear Regression · KNN Regressor
        """
    )

    section("Dataset at a Glance")
    metric_row([
        ("Total Records",     f"{len(df_raw):,}"),
        ("Features",          str(len(df_raw.columns) - 1)),
        ("Sampling Interval", "15 min"),
        ("Date Range",        f"{df_raw['date'].dt.year.min()}–{df_raw['date'].dt.year.max()}"),
        ("Load Types",        "3"),
    ])

    st.markdown("#### First 10 rows")
    st.dataframe(df_raw.head(10), use_container_width=True)

    section("Descriptive Statistics")
    st.dataframe(df_raw.describe().round(3), use_container_width=True)

    section("Missing Values")
    missing = df_raw.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Count"]
    st.dataframe(missing[missing["Missing Count"] > 0]
                 if missing["Missing Count"].sum() > 0
                 else pd.DataFrame({"Status": ["✅ No missing values"]}),
                 use_container_width=True)

    section("Load Type Distribution")
    lt_counts = df_raw["Load_Type"].value_counts().reset_index()
    lt_counts.columns = ["Load Type", "Count"]
    fig = px.pie(lt_counts, names="Load Type", values="Count",
                 color_discrete_sequence=px.colors.qualitative.Set2,
                 hole=0.4)
    fig.update_layout(margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
#  PAGE 2 – EDA
# ============================================================
elif page == "eda":
    st.title("📊 Exploratory Data Analysis")

    section("Energy Consumption Time-Series (first 2,016 records ≈ 3 weeks)")
    ts_df = df_raw.head(2016).copy()
    fig = px.line(ts_df, x="date", y=TARGET,
                  labels={"Usage_kWh": "Usage (kWh)", "date": ""},
                  color_discrete_sequence=["#3b82f6"])
    fig.update_layout(hovermode="x unified", margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    section("Distribution of Usage_kWh")
    fig = px.histogram(df_raw, x=TARGET, nbins=80,
                       color_discrete_sequence=["#6366f1"],
                       labels={"Usage_kWh": "Usage (kWh)"})
    fig.update_layout(bargap=0.05, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    section("Usage by Load Type")
    fig = px.box(df_raw, x="Load_Type", y=TARGET,
                 color="Load_Type",
                 color_discrete_sequence=px.colors.qualitative.Pastel,
                 labels={"Usage_kWh": "Usage (kWh)", "Load_Type": "Load Type"})
    fig.update_layout(showlegend=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    section("Usage by Day of Week")
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    fig = px.box(df_raw, x="Day_of_week", y=TARGET,
                 category_orders={"Day_of_week": day_order},
                 color="Day_of_week",
                 color_discrete_sequence=px.colors.qualitative.Set3,
                 labels={"Usage_kWh": "Usage (kWh)", "Day_of_week": "Day"})
    fig.update_layout(showlegend=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    section("Average Hourly Consumption Profile")
    df_raw2 = df_raw.copy()
    df_raw2["hour"] = df_raw2["date"].dt.hour
    hourly = df_raw2.groupby("hour")[TARGET].mean().reset_index()
    fig = px.bar(hourly, x="hour", y=TARGET,
                 color=TARGET, color_continuous_scale="Blues",
                 labels={"Usage_kWh": "Avg Usage (kWh)", "hour": "Hour of Day"})
    fig.update_layout(margin=dict(t=20, b=20), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    section("Scatter: Lagging Reactive Power vs Usage")
    fig = px.scatter(df_raw.sample(3000, random_state=1), 
                     x="Lagging_Current_Reactive.Power_kVarh", y=TARGET,
                     color="Load_Type", opacity=0.6,
                     color_discrete_sequence=px.colors.qualitative.Set1,
                     labels={"Lagging_Current_Reactive.Power_kVarh": "Lagging Reactive Power",
                             "Usage_kWh": "Usage (kWh)"})
    fig.update_layout(margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
#  PAGE 3 – Preprocessing
# ============================================================
elif page == "preprocessing":
    st.title("⚙️ Preprocessing & Feature Engineering")

    section("Phase 1 – Temporal Feature Extraction")
    temporal_sample = df_proc[["date", "hour", "minute", "day_of_week_num", "month"]].head(8) \
        if "date" in df_proc.columns else df_proc[["hour", "minute", "day_of_week_num", "month"]].head(8)
    st.markdown("Sample of extracted temporal features:")
    st.dataframe(temporal_sample, use_container_width=True)

    shift_dist = df_raw.copy()
    shift_dist["hour"] = pd.to_datetime(shift_dist["date"]).dt.hour
    def assign_shift(h):
        if 6 <= h < 14: return "Morning (06-14)"
        elif 14 <= h < 22: return "Afternoon (14-22)"
        else: return "Night (22-06)"
    shift_dist["Shift"] = shift_dist["hour"].apply(assign_shift)
    sc = shift_dist["Shift"].value_counts().reset_index()
    sc.columns = ["Shift", "Count"]
    fig = px.bar(sc, x="Shift", y="Count",
                 color="Shift", color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(showlegend=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    section("Phase 2 – Correlation Analysis")
    corr = compute_correlations(df_proc)
    corr_df = corr.reset_index()
    corr_df.columns = ["Feature", "Pearson r"]
    fig = px.bar(corr_df, x="Feature", y="Pearson r",
                 color="Pearson r", color_continuous_scale="RdBu",
                 color_continuous_midpoint=0,
                 labels={"Pearson r": "Correlation with Usage_kWh"})
    fig.update_layout(xaxis_tickangle=-45, margin=dict(t=20, b=80))
    st.plotly_chart(fig, use_container_width=True)

    section("Pearson Correlation Table")
    st.dataframe(corr_df.round(4), use_container_width=True)

    section("Correlation Heatmap (numeric features)")
    num_cols = [c for c in NUMERIC_FEATURES if c in df_raw.columns] + [TARGET]
    corr_mat = df_raw[num_cols].corr().round(3)
    renamed = {c: DISPLAY_NAMES.get(c, c) for c in num_cols}
    corr_mat = corr_mat.rename(index=renamed, columns=renamed)
    fig = px.imshow(corr_mat, text_auto=True, color_continuous_scale="RdBu",
                    color_continuous_midpoint=0, aspect="auto")
    fig.update_layout(margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    section("Feature Matrix Shape after Encoding")
    st.info(f"**X shape:** {X.shape[0]:,} rows × {X.shape[1]} features  |  **y shape:** {y.shape[0]:,}")


# ============================================================
#  PAGE 4 – Decision Tree
# ============================================================
elif page == "decision_tree":
    st.title("🌳 Decision Tree Regressor (CART)")

    sweep_df = pipe["sweep_df"]
    opt_depth = pipe["opt_depth"]

    section("Phase 3 – Depth Sweep (max_depth 1 → 20)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sweep_df["depth"], y=sweep_df["train_rmse"],
        mode="lines+markers", name="Train RMSE",
        line=dict(color="#ef4444", width=2),
        marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=sweep_df["depth"], y=sweep_df["cv_rmse"],
        mode="lines+markers", name="CV RMSE (5-fold)",
        line=dict(color="#3b82f6", width=2),
        marker=dict(size=6),
    ))
    fig.add_vline(x=opt_depth, line_dash="dash", line_color="#16a34a",
                  annotation_text=f"Optimal depth = {opt_depth}",
                  annotation_position="top right")
    fig.update_layout(
        xaxis_title="max_depth", yaxis_title="RMSE",
        hovermode="x unified", margin=dict(t=30, b=20),
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    **Interpretation:**
    - As `max_depth` increases, training RMSE monotonically decreases → the tree memorises more.
    - CV RMSE reaches its minimum at **depth = {opt_depth}** then starts to increase (overfitting).
    - The gap between train and CV RMSE widens with depth — classic bias–variance trade-off.
    """)

    section(f"Optimal Decision Tree Metrics (depth = {opt_depth})")
    m = pipe["dt_metrics"]
    metric_row([
        ("MAE",       f"{m['MAE']:.4f}"),
        ("RMSE",      f"{m['RMSE']:.4f}"),
        ("R²",        f"{m['R²']:.4f}"),
        ("MAPE (%)",  f"{m['MAPE (%)']:.2f}%"),
    ])

    section("Actual vs Predicted (test set sample – 500 points)")
    n = min(500, len(pipe["y_test"]))
    idx = np.arange(n)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=pipe["y_test"][:n],
                             mode="lines", name="Actual",
                             line=dict(color="#64748b", width=1)))
    fig.add_trace(go.Scatter(x=idx, y=pipe["dt_preds"][:n],
                             mode="lines", name="DT Predicted",
                             line=dict(color="#f97316", width=1.5)))
    fig.update_layout(xaxis_title="Sample Index", yaxis_title="Usage (kWh)",
                      hovermode="x unified", margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    section("Depth Sweep Table")
    st.dataframe(sweep_df.round(4), use_container_width=True, height=320)


# ============================================================
#  PAGE 5 – Step-Function Response
# ============================================================
elif page == "step_function":
    st.title("📈 Step-Function Response Surface")
    st.markdown("""
    A Decision Tree produces **piecewise-constant (step-function) predictions** — 
    each leaf returns the mean target of its training samples.  
    Below we vary the dominant predictor **Lagging Reactive Power** while 
    holding all other features at their mean to reveal the non-linear response surface.
    """)

    predictor = "Lagging_Current_Reactive.Power_kVarh"
    feat_list = list(pipe["feature_names"])

    section("DT Step-Function vs Lagging Reactive Power")
    # step_x / step_y are computed on real sorted training samples
    step_x, step_y = step_function_response(
        pipe["dt_model"], pipe["X_train"], feat_list, predictor
    )

    col_idx = feat_list.index(predictor)
    X_tr = pipe["X_train"]

    # MLR predictions on the SAME sorted sample rows as DT
    sort_idx = np.argsort(X_tr[:, col_idx])
    n_pts = len(step_x)
    thin = np.linspace(0, len(sort_idx) - 1, n_pts, dtype=int)
    X_plot_rows = X_tr[sort_idx][thin]
    mlr_step_y = pipe["mlr_model"].predict(X_plot_rows)

    # Actual data scatter (sample)
    actual_x = X_tr[:, col_idx]
    sample_idx = np.random.default_rng(0).choice(len(actual_x), size=1500, replace=False)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actual_x[sample_idx],
        y=pipe["y_train"][sample_idx],
        mode="markers", name="Actual (train sample)",
        marker=dict(color="#94a3b8", size=3, opacity=0.5),
    ))
    fig.add_trace(go.Scatter(
        x=step_x, y=mlr_step_y, mode="lines", name="MLR (linear)",
        line=dict(color="#3b82f6", width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=step_x, y=step_y, mode="lines", name="DT Step-Function",
        line=dict(color="#ef4444", width=2.5),
    ))
    fig.update_layout(
        xaxis_title="Lagging Reactive Power (kVArh)",
        yaxis_title="Predicted Usage (kWh)",
        hovermode="x unified",
        margin=dict(t=20, b=20),
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Key Observations:**
    - The **red step-function** shows the DT's piecewise-constant behaviour — each horizontal 
      segment corresponds to a leaf node.
    - The **blue dashed line** is the MLR response — smooth and linear.
    - The DT can capture non-linear relationships that MLR misses, especially the sharp 
      increase in consumption as reactive power rises.
    - Narrow steps at high reactive-power values indicate fewer training samples → potential 
      overfitting at the extremes.
    """)


# ============================================================
#  PAGE 6 – Benchmarking
# ============================================================
elif page == "benchmarking":
    st.title("🏆 Model Benchmarking — 5-Fold Cross-Validation")
    st.markdown("""
    All three models are evaluated with **5-fold cross-validation** on the training set.
    Metrics: **MAE**, **RMSE**, **R²**, **MAPE**.
    """)

    bench = pipe["bench_df"]

    section("Cross-Validation Metrics Summary")
    st.dataframe(bench.set_index("Model").round(4), use_container_width=True)

    section("RMSE Comparison")
    fig = px.bar(bench, x="Model", y="RMSE",
                 color="Model", text_auto=".4f",
                 color_discrete_sequence=["#3b82f6", "#f97316", "#10b981"])
    fig.update_layout(showlegend=False, margin=dict(t=20, b=20),
                      xaxis_title="", yaxis_title="RMSE (kWh)")
    st.plotly_chart(fig, use_container_width=True)

    section("R² Comparison")
    fig = px.bar(bench, x="Model", y="R²",
                 color="Model", text_auto=".4f",
                 color_discrete_sequence=["#3b82f6", "#f97316", "#10b981"])
    fig.update_layout(showlegend=False, margin=dict(t=20, b=20),
                      xaxis_title="", yaxis_title="R²")
    st.plotly_chart(fig, use_container_width=True)

    section("MAE Comparison")
    fig = px.bar(bench, x="Model", y="MAE",
                 color="Model", text_auto=".4f",
                 color_discrete_sequence=["#3b82f6", "#f97316", "#10b981"])
    fig.update_layout(showlegend=False, margin=dict(t=20, b=20),
                      xaxis_title="", yaxis_title="MAE (kWh)")
    st.plotly_chart(fig, use_container_width=True)

    section("MAPE Comparison")
    fig = px.bar(bench, x="Model", y="MAPE (%)",
                 color="Model", text_auto=".2f",
                 color_discrete_sequence=["#3b82f6", "#f97316", "#10b981"])
    fig.update_layout(showlegend=False, margin=dict(t=20, b=20),
                      xaxis_title="", yaxis_title="MAPE (%)")
    st.plotly_chart(fig, use_container_width=True)

    section("Test-Set: Actual vs Predicted (all 3 models – 300 samples)")
    n = min(300, len(pipe["y_test"]))
    idx = np.arange(n)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=pipe["y_test"][:n], mode="lines",
                             name="Actual", line=dict(color="#64748b", width=1.5)))
    fig.add_trace(go.Scatter(x=idx, y=pipe["dt_preds"][:n], mode="lines",
                             name="Decision Tree", line=dict(color="#ef4444", width=1.5)))
    fig.add_trace(go.Scatter(x=idx, y=pipe["mlr_preds"][:n], mode="lines",
                             name="MLR", line=dict(color="#3b82f6", width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=idx, y=pipe["knn_preds"][:n], mode="lines",
                             name="KNN", line=dict(color="#10b981", width=1.5, dash="dash")))
    fig.update_layout(xaxis_title="Sample", yaxis_title="Usage (kWh)",
                      hovermode="x unified", margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
#  PAGE 7 – Predict
# ============================================================
elif page == "predict":
    st.title("🔮 Live Single-Record Prediction")
    st.markdown(
        "Enter sensor readings below to get an instant energy-consumption forecast "
        "from all three models."
    )

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        lag_rp  = c1.number_input("Lagging Reactive Power (kVArh)", 0.0, 200.0, 5.0, 0.1)
        lead_rp = c1.number_input("Leading Reactive Power (kVArh)", 0.0, 200.0, 0.0, 0.1)
        co2     = c1.number_input("CO₂ Emissions (tCO₂)", 0.0, 10.0, 0.0, 0.01)
        lag_pf  = c2.number_input("Lagging Power Factor", 0.0, 100.0, 70.0, 0.1)
        lead_pf = c2.number_input("Leading Power Factor", 0.0, 100.0, 100.0, 0.1)
        nsm     = c2.number_input("NSM (seconds since midnight)", 0, 86400, 900, 900)
        hour    = c3.slider("Hour of Day", 0, 23, 8)
        minute  = c3.selectbox("Minute", [0, 15, 30, 45])
        dow     = c3.selectbox("Day of Week", list(range(7)),
                               format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
        submitted = st.form_submit_button("⚡ Forecast Energy Consumption", type="primary")

    if submitted:
        scaler = pipe["scaler"]
        feat_names_list = list(pipe["feature_names"])
        n_features = len(feat_names_list)

        # Build a row of zeros then fill known features
        row = np.zeros(n_features)
        feature_map = {
            "Lagging_Current_Reactive.Power_kVarh": lag_rp,
            "Leading_Current_Reactive_Power_kVarh": lead_rp,
            "CO2(tCO2)": co2,
            "Lagging_Current_Power_Factor": lag_pf,
            "Leading_Current_Power_Factor": lead_pf,
            "NSM": nsm,
            "hour": hour,
            "minute": minute,
            "day_of_week_num": dow,
        }
        for fname, val in feature_map.items():
            if fname in feat_names_list:
                row[feat_names_list.index(fname)] = val

        X_input = scaler.transform(row.reshape(1, -1))

        dt_pred  = pipe["dt_model"].predict(X_input)[0]
        mlr_pred = pipe["mlr_model"].predict(X_input)[0]
        knn_pred = pipe["knn_model"].predict(X_input)[0]

        st.success("Predictions generated successfully!")
        metric_row([
            ("Decision Tree (CART)", f"{dt_pred:.3f} kWh"),
            ("Multiple Linear Regression", f"{mlr_pred:.3f} kWh"),
            ("KNN Regressor", f"{knn_pred:.3f} kWh"),
            ("Ensemble Average", f"{np.mean([dt_pred, mlr_pred, knn_pred]):.3f} kWh"),
        ])

        fig = px.bar(
            x=["Decision Tree", "MLR", "KNN", "Ensemble Avg"],
            y=[dt_pred, mlr_pred, knn_pred, np.mean([dt_pred, mlr_pred, knn_pred])],
            color=["Decision Tree", "MLR", "KNN", "Ensemble"],
            color_discrete_sequence=["#ef4444", "#3b82f6", "#10b981", "#f59e0b"],
            labels={"x": "Model", "y": "Predicted Usage (kWh)"},
            text_auto=".3f",
        )
        fig.update_layout(showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
