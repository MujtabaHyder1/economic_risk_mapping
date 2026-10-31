"""
Economic Risk Mapping of Horticulture Under Shifting Climatic Zones
Interactive Streamlit Dashboard
Focus: Kashmir Valley (Shopian, Baramulla, Pulwama)
Crops: Apples, Walnuts, Saffron
"""

import sys
import os
from pathlib import Path
import json

# Setup paths
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from model_training import XGBoostTrainer

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Economic Risk Mapping | Kashmir Horticulture",
    page_icon="🍏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .risk-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .risk-low { background-color: #DCFCE7; color: #166534; }
    .risk-moderate { background-color: #FEF9C3; color: #854D0E; }
    .risk-high { background-color: #FED7AA; color: #9A3412; }
    .risk-very-high { background-color: #FEE2E2; color: #991B1B; }
    .risk-critical { background-color: #7F1D1D; color: #FFFFFF; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# CACHED DATA & MODEL LOADERS
# ==============================================================================
@st.cache_resource
def load_trained_model():
    """Load cached trained XGBoost model."""
    model_path = PROJECT_ROOT / "models" / "xgboost_model.pkl"
    if model_path.exists():
        return joblib.load(model_path)
    return None

@st.cache_data
def load_project_data():
    """Load processed features, evaluation metrics, and scenario data."""
    data = {}
    proc_path = PROJECT_ROOT / "data" / "processed" / "merged_features.csv"
    if proc_path.exists():
        data["processed"] = pd.read_csv(proc_path)
    else:
        data["processed"] = None

    metrics_path = PROJECT_ROOT / "results" / "evaluation_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            data["metrics"] = json.load(f)
    else:
        data["metrics"] = None

    v15_path = PROJECT_ROOT / "results" / "vulnerability_index_1_5C.csv"
    if v15_path.exists():
        data["v15"] = pd.read_csv(v15_path)
    else:
        data["v15"] = None

    v20_path = PROJECT_ROOT / "results" / "vulnerability_index_2_0C.csv"
    if v20_path.exists():
        data["v20"] = pd.read_csv(v20_path)
    else:
        data["v20"] = None

    return data

data_bundle = load_project_data()
model = load_trained_model()
df_processed = data_bundle["processed"]

def get_risk_badge(score: float) -> str:
    """Return colored risk badge based on vulnerability index."""
    if score < 20:
        return f'<span class="risk-badge risk-low">Low ({score:.1f})</span>'
    elif score < 40:
        return f'<span class="risk-badge risk-moderate">Moderate ({score:.1f})</span>'
    elif score < 60:
        return f'<span class="risk-badge risk-high">High ({score:.1f})</span>'
    elif score < 80:
        return f'<span class="risk-badge risk-very-high">Very High ({score:.1f})</span>'
    else:
        return f'<span class="risk-badge risk-critical">Critical ({score:.1f})</span>'


# ==============================================================================
# SIDEBAR CONTROLS & SCENARIO SIMULATOR
# ==============================================================================
st.sidebar.image("https://images.unsplash.com/photo-1560493676-04071c5f467b?w=600&auto=format&fit=crop&q=80", use_container_width=True)
st.sidebar.title("Climate Stress Lab")
st.sidebar.caption("Simulate micro-climate shocks across Kashmir Valley")

st.sidebar.markdown("### 🌡️ Temperature Forcing")
temp_delta = st.sidebar.slider(
    "Mean Temperature Anomaly (+°C)",
    min_value=0.0,
    max_value=3.5,
    value=1.5,
    step=0.25,
    help="Increase in seasonal mean and pre-harvest temperatures"
)

st.sidebar.markdown("### ❄️ Frost & Heat Extremes")
frost_change = st.sidebar.slider(
    "March Frost Days Change (days)",
    min_value=-10,
    max_value=5,
    value=-5,
    help="Shift in frost events during critical March blooming window"
)

heat_days_added = st.sidebar.slider(
    "Pre-Harvest Heatwave Days (>30°C)",
    min_value=0,
    max_value=15,
    value=3,
    help="Consecutive heat stress days during July-August pre-harvest"
)

st.sidebar.markdown("### 🌿 Vegetation & Moisture")
precip_pct = st.sidebar.slider(
    "Growing Season Rainfall Delta (%)",
    min_value=-40,
    max_value=40,
    value=-10,
    step=5,
    help="Percentage shift in growing season precipitation (Apr-Oct)"
)

ndvi_stress = st.sidebar.slider(
    "NDVI Stress Deviation",
    min_value=-0.20,
    max_value=0.10,
    value=-0.05,
    step=0.01,
    help="Vegetation index deviation from historical 10-year rolling average"
)

st.sidebar.markdown("---")
districts_selected = st.sidebar.multiselect(
    "Filter Districts",
    options=["Shopian", "Baramulla", "Pulwama"],
    default=["Shopian", "Baramulla", "Pulwama"]
)

crops_selected = st.sidebar.multiselect(
    "Filter Crops",
    options=["Apples", "Walnuts", "Saffron"],
    default=["Apples", "Walnuts", "Saffron"]
)

# ==============================================================================
# HEADER
# ==============================================================================
st.markdown('<div class="main-header">🍏 Economic Risk Mapping of Kashmir Horticulture</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Spatial Machine Learning Platform for Quantifying Climate Shocks, Per-Hectare Financial Loss & District Vulnerability</div>', unsafe_allow_html=True)

if df_processed is None or model is None:
    st.error("Model or processed data not found. Please run `python main.py` in the terminal to initialize the data and model.")
    st.stop()

# ==============================================================================
# DYNAMIC SIMULATION CALCULATION
# ==============================================================================
trainer = XGBoostTrainer()
X_full, _ = trainer.prepare_features(df_processed)
feature_names = trainer.feature_names

sim_rows = []
for district in districts_selected:
    for crop in crops_selected:
        subset = df_processed[(df_processed["District"] == district) & (df_processed["Crop_Type"] == crop)].copy()
        if len(subset) == 0:
            continue

        X_base = subset[feature_names].copy()
        pred_base = float(model.predict(X_base).mean())

        # Construct perturbed scenario
        X_scen = X_base.copy()
        temp_features = [c for c in feature_names if "Temp" in c and "PreHarvest" not in c]
        X_scen[temp_features] = X_scen[temp_features] + temp_delta

        if "Max_Temp_PreHarvest" in X_scen.columns:
            X_scen["Max_Temp_PreHarvest"] = X_scen["Max_Temp_PreHarvest"] + temp_delta * 1.2
        if "Hot_Days_JulAug" in X_scen.columns:
            X_scen["Hot_Days_JulAug"] = np.maximum(0, X_scen["Hot_Days_JulAug"] + heat_days_added)
        if "Frost_Days_March" in X_scen.columns:
            X_scen["Frost_Days_March"] = np.maximum(0, X_scen["Frost_Days_March"] + frost_change)
        if "Precip_Growing_Season" in X_scen.columns:
            X_scen["Precip_Growing_Season"] = X_scen["Precip_Growing_Season"] * (1.0 + precip_pct / 100.0)
        if "NDVI_Deviation" in X_scen.columns:
            X_scen["NDVI_Deviation"] = X_scen["NDVI_Deviation"] + ndvi_stress

        pred_scen = float(model.predict(X_scen).mean())
        loss_pct = max(0.0, (pred_base - pred_scen) / (pred_base if pred_base > 0 else 1.0) * 100.0)
        abs_loss_inr = max(0.0, pred_base - pred_scen)

        sim_rows.append({
            "District": district,
            "Crop_Type": crop,
            "Baseline_Revenue_INR": pred_base,
            "Scenario_Revenue_INR": pred_scen,
            "Loss_INR_per_Ha": abs_loss_inr,
            "Revenue_Loss_Pct": loss_pct
        })

df_sim = pd.DataFrame(sim_rows)
if not df_sim.empty:
    max_loss = df_sim["Revenue_Loss_Pct"].max()
    if max_loss > 0:
        df_sim["Vulnerability_Index"] = (df_sim["Revenue_Loss_Pct"] / max_loss * 100.0).round(1)
    else:
        df_sim["Vulnerability_Index"] = 0.0
else:
    df_sim["Vulnerability_Index"] = 0.0

# ==============================================================================
# TOP-LEVEL KPI ROW
# ==============================================================================
c1, c2, c3, c4 = st.columns(4)

with c1:
    avg_vuln = df_sim["Vulnerability_Index"].mean() if not df_sim.empty else 0.0
    st.metric("Avg Vulnerability Index", f"{avg_vuln:.1f} / 100", delta=f"{temp_delta:+.1f}°C Scenario")

with c2:
    mean_pct_loss = df_sim["Revenue_Loss_Pct"].mean() if not df_sim.empty else 0.0
    st.metric("Expected Revenue Loss", f"{mean_pct_loss:.1f}%", delta=f"{precip_pct:+d}% Rainfall", delta_color="inverse")

with c3:
    if not df_sim.empty:
        highest_risk_row = df_sim.loc[df_sim["Vulnerability_Index"].idxmax()]
        st.metric("Highest Vulnerability", f"{highest_risk_row['District']} - {highest_risk_row['Crop_Type']}", f"{highest_risk_row['Vulnerability_Index']:.1f} Index")
    else:
        st.metric("Highest Vulnerability", "N/A")

with c4:
    mean_abs_loss = df_sim["Loss_INR_per_Ha"].mean() if not df_sim.empty else 0.0
    st.metric("Mean Loss per Hectare", f"₹{mean_abs_loss:,.0f} / ha")

st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# MAIN DASHBOARD TABS
# ==============================================================================
tab_map, tab_sim, tab_econ, tab_diag, tab_data = st.tabs([
    "🗺️ Vulnerability & District Profiles",
    "🧪 Real-Time Scenario Breakdown",
    "🌾 Crop Economics & Loss Dynamics",
    "🤖 Model Diagnostics & Explainability",
    "📁 Raw Data & Scenario Export"
])

# ------------------------------------------------------------------------------
# TAB 1: Vulnerability Map & District Profiles
# ------------------------------------------------------------------------------
with tab_map:
    st.markdown("### District Financial Vulnerability Under Current Scenario")
    col_cards, col_chart = st.columns([1, 1.2])

    with col_cards:
        st.markdown("#### District Risk Ratings")
        district_agg = df_sim.groupby("District").agg({
            "Vulnerability_Index": "mean",
            "Revenue_Loss_Pct": "mean",
            "Loss_INR_per_Ha": "mean"
        }).reset_index().sort_values("Vulnerability_Index", ascending=False)

        for _, d_row in district_agg.iterrows():
            d_name = d_row["District"]
            v_score = d_row["Vulnerability_Index"]
            badge = get_risk_badge(v_score)
            
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom: 0.8rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 1.15rem; font-weight: 700; color: #0F172A;">{d_name}</span>
                    {badge}
                </div>
                <div style="margin-top: 0.5rem; font-size: 0.95rem; color: #475569;">
                    <b>Expected Revenue Loss:</b> {d_row['Revenue_Loss_Pct']:.1f}% &nbsp;|&nbsp; 
                    <b>Avg Financial Deficit:</b> ₹{d_row['Loss_INR_per_Ha']:,.0f}/ha
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_chart:
        st.markdown("#### Vulnerability Comparison by District & Crop")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.barplot(
            data=df_sim,
            x="District",
            y="Vulnerability_Index",
            hue="Crop_Type",
            palette="crest",
            ax=ax
        )
        ax.set_title("Vulnerability Index by Crop Type (0-100)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Index Score")
        ax.set_ylim(0, 105)
        ax.legend(title="Crop Type", loc="upper right")
        st.pyplot(fig)
        plt.close()

# ------------------------------------------------------------------------------
# TAB 2: Real-Time Scenario Breakdown
# ------------------------------------------------------------------------------
with tab_sim:
    st.markdown("### Active Micro-Climate Simulation Results")
    st.caption(f"Configured Anomaly: **{temp_delta:+.2f}°C Temp** | **{precip_pct:+d}% Rain** | **{frost_change:+d} Frost Days** | **{heat_days_added:+d} Heatwave Days**")

    st.dataframe(
        df_sim.style.format({
            "Baseline_Revenue_INR": "₹{:,.0f}",
            "Scenario_Revenue_INR": "₹{:,.0f}",
            "Loss_INR_per_Ha": "₹{:,.0f}",
            "Revenue_Loss_Pct": "{:.2f}%",
            "Vulnerability_Index": "{:.1f}"
        }).background_gradient(subset=["Vulnerability_Index"], cmap="YlOrRd"),
        use_container_width=True
    )

    st.markdown("#### Projected Revenue Shift (Baseline vs. Shocked Scenario)")
    melted_rev = pd.melt(
        df_sim,
        id_vars=["District", "Crop_Type"],
        value_vars=["Baseline_Revenue_INR", "Scenario_Revenue_INR"],
        var_name="Scenario",
        value_name="Revenue_INR"
    )
    melted_rev["Scenario"] = melted_rev["Scenario"].replace({
        "Baseline_Revenue_INR": "Historical Baseline",
        "Scenario_Revenue_INR": f"Scenario (+{temp_delta}°C)"
    })

    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.barplot(
        data=melted_rev,
        x="Crop_Type",
        y="Revenue_INR",
        hue="Scenario",
        palette=["#3B82F6", "#EF4444"],
        ax=ax
    )
    ax.set_title("Revenue Impact: Baseline vs. Climate Scenario", fontweight="bold")
    ax.set_ylabel("Revenue (INR per Hectare)")
    ax.yaxis.set_major_formatter('₹{x:,.0f}')
    st.pyplot(fig)
    plt.close()

# ------------------------------------------------------------------------------
# TAB 3: Crop Economics & Loss Dynamics
# ------------------------------------------------------------------------------
with tab_econ:
    st.markdown("### Crop Economic Characteristics in Kashmir Valley")
    c_e1, c_e2 = st.columns(2)

    with c_e1:
        st.markdown("#### Historical Farm-Gate Price Distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=df_processed, x="Crop_Type", y="Farm_Gate_Price", palette="Set2", ax=ax)
        ax.set_title("Farm-Gate Prices (INR / kg)", fontweight="bold")
        ax.set_ylabel("Price (INR/kg)")
        st.pyplot(fig)
        plt.close()

    with c_e2:
        st.markdown("#### Yield per Hectare by District")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(data=df_processed, x="Crop_Type", y="Yield_per_Hectare", hue="District", palette="viridis", ax=ax)
        ax.set_title("Mean Yield (Metric Tons / Hectare)", fontweight="bold")
        ax.set_ylabel("Yield (MT/ha)")
        st.pyplot(fig)
        plt.close()

# ------------------------------------------------------------------------------
# TAB 4: Model Diagnostics & Explainability
# ------------------------------------------------------------------------------
with tab_diag:
    st.markdown("### XGBoost Machine Learning Model Diagnostics")
    m_col1, m_col2 = st.columns([1, 1.2])

    with m_col1:
        st.markdown("#### Holdout Test Set Performance")
        metrics = data_bundle["metrics"]
        if metrics:
            st.metric("Test R² Score", f"{metrics.get('test_r2', 0):.4f}")
            st.metric("Test Mean Absolute Error (MAE)", f"₹{metrics.get('test_mae', 0):,.2f} / ha")
            st.metric("Root Mean Squared Error (RMSE)", f"₹{metrics.get('test_rmse', 0):,.2f}")
            st.metric("Mean Absolute Percentage Error", f"{metrics.get('test_mape', 0):.2f}%")
        else:
            st.info("Metrics file not found. Run main.py to compute metrics.")

    with m_col2:
        st.markdown("#### Feature Importance (XGBoost)")
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": imp}).sort_values("Importance", ascending=True).tail(12)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(feat_imp["Feature"], feat_imp["Importance"], color="teal")
            ax.set_title("Top Predictors for Horticulture Economic Loss", fontweight="bold")
            ax.set_xlabel("Relative Importance Score")
            st.pyplot(fig)
            plt.close()

# ------------------------------------------------------------------------------
# TAB 5: Raw Data & Scenario Export
# ------------------------------------------------------------------------------
with tab_data:
    st.markdown("### Download Simulation Scenario Report")
    st.markdown("Export the current simulation outputs for policy planning, agricultural extension, or risk assessment.")

    csv_data = df_sim.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Active Scenario CSV",
        data=csv_data,
        file_name=f"kashmir_horticulture_risk_{temp_delta}C.csv",
        mime="text/csv",
    )

    st.markdown("#### Processed Dataset Preview (`data/processed/merged_features.csv`)")
    st.dataframe(df_processed.head(20), use_container_width=True)

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.caption("Economic Risk Mapping Platform | Shifting Agro-Climatic Zones Analysis | Kashmir Valley Horticulture Initiative")

