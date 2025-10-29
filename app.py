"""
NexGen Predictive Delivery Optimizer — Pro Edition (Streamlit)
Upgraded UI/UX: dark mode toggle, top navigation tabs, toasts, drill‑down modal,
optional SHAP explainability, and gentle fallbacks.

NOTE: This file assumes the same `src/` module interfaces you already have.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import project modules
try:
    from data import load_and_prepare_data
    from features import engineer_features
    from model import (
        train_and_evaluate_model,
        predict_new_orders,
        DelayPredictor,
    )
    from rules import generate_prescriptions
    from utils import (
        plot_feature_importance,
        plot_roc_curve,
        plot_risk_distribution,
        plot_confusion_matrix,
        plot_actions_by_type,
        plot_risk_by_priority,
        calculate_business_impact,
        create_executive_summary,
        format_currency,
        export_action_plan_csv,
    )
except Exception as e:
    st.error(f"Import Error: {e}")
    st.stop()

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="NexGen Delivery Optimizer — Pro",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Theme toggle (Light / Dark)
# ----------------------------
if 'theme' not in st.session_state:
    st.session_state.theme = 'Light'

theme = st.session_state.theme

LIGHT_CSS = """
<style>
:root {
  --bg: #ffffff; --text: #2c3e50; --muted: #6b7280; --card: #f8fafc; --brandA: #667eea; --brandB: #764ba2;
  --success: #28a745; --warn: #ffc107; --danger: #dc3545; --info: #17a2b8;
}
body { background: var(--bg); }
.main-header { font-size: 3rem; font-weight: 800; background: linear-gradient(135deg, var(--brandA) 0%, var(--brandB) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align:center; padding:1.5rem 0; margin-bottom:1rem; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, var(--brandA) 0%, var(--brandB) 100%); }
[data-testid="stSidebar"] * { color: white !important; }
.metric-card { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding:1.25rem; border-radius:1rem; border-left:5px solid var(--brandA); box-shadow:0 4px 6px rgba(0,0,0,0.08); }
.success-box { background: linear-gradient(135deg, #d4edda 0%, #a8d5ba 100%); padding:1.25rem; border-radius:1rem; border-left:5px solid var(--success); }
.warning-box { background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); padding:1.25rem; border-radius:1rem; border-left:5px solid var(--warn); }
.danger-box { background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); padding:1.25rem; border-radius:1rem; border-left:5px solid var(--danger); }
.info-box { background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%); padding:1.25rem; border-radius:1rem; border-left:5px solid var(--info); }
.feature-card { background:white; padding:2rem; border-radius:1rem; box-shadow:0 8px 16px rgba(0,0,0,0.08); border-top:4px solid var(--brandA); }
.stats-card { background: linear-gradient(135deg, var(--brandA) 0%, var(--brandB) 100%); color:white; padding:2rem; border-radius:1rem; text-align:center; }
.stButton > button { background:linear-gradient(135deg, var(--brandA) 0%, var(--brandB) 100%); color:white; font-weight:600; border:none; border-radius:.5rem; padding:.75rem 1.25rem; box-shadow:0 4px 6px rgba(102,126,234,0.3); }
[data-testid="stMetricValue"] { color: var(--brandA); }
</style>
"""

DARK_CSS = """
<style>
:root {
  --bg: #0b1020; --text: #ecf2ff; --muted: #9aa4c7; --card: #111735; --brandA: #7aa2ff; --brandB: #a78bfa;
  --success: #36d399; --warn: #fbbd23; --danger: #f87272; --info: #22d3ee;
}
body, .stApp { background: var(--bg); color: var(--text); }
.block-container h1, h2, h3, h4, h5, h6, p, label, span { color: var(--text) !important; }
.main-header { font-size: 3rem; font-weight: 800; background: linear-gradient(135deg, var(--brandA) 0%, var(--brandB) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align:center; padding:1.5rem 0; margin-bottom:1rem; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a 0%, #1f2937 100%); }
[data-testid="stSidebar"] * { color: #e5e7eb !important; }
.metric-card { background: #0f1633; padding:1.25rem; border-radius:1rem; border-left:5px solid var(--brandA); box-shadow: inset 0 0 0 1px rgba(255,255,255,0.04); }
.success-box { background: rgba(54,211,153,0.12); padding:1.25rem; border-radius:1rem; border-left:5px solid var(--success); }
.warning-box { background: rgba(251,189,35,0.12); padding:1.25rem; border-radius:1rem; border-left:5px solid var(--warn); }
.danger-box { background: rgba(248,114,114,0.12); padding:1.25rem; border-radius:1rem; border-left:5px solid var(--danger); }
.info-box { background: rgba(34,211,238,0.12); padding:1.25rem; border-radius:1rem; border-left:5px solid var(--info); }
.feature-card { background:#0f1633; padding:2rem; border-radius:1rem; box-shadow: inset 0 0 0 1px rgba(255,255,255,0.06); border-top:4px solid var(--brandA); }
.stats-card { background: linear-gradient(135deg, var(--brandA) 0%, var(--brandB) 100%); color:white; padding:2rem; border-radius:1rem; text-align:center; }
.stButton > button { background:linear-gradient(135deg, var(--brandA) 0%, var(--brandB) 100%); color:white; font-weight:600; border:none; border-radius:.5rem; padding:.75rem 1.25rem; box-shadow:0 4px 6px rgba(16,24,64,0.6); }
[data-testid="stMetricValue"] { color: var(--brandA); }
</style>
"""

st.markdown(LIGHT_CSS if theme == 'Light' else DARK_CSS, unsafe_allow_html=True)

# ----------------------------
# Sidebar — Theme + Quick Stats
# ----------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 1rem; background: rgba(255,255,255,0.08); border-radius: 1rem; margin-bottom: 1rem;">
            <h1 style="margin:0">📦</h1>
            <h3 style="margin:.25rem 0 0 0">NexGen</h3>
            <p style="opacity:.8; margin:.25rem 0 0 0; font-size:.9rem">Delivery Optimizer — Pro</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.selectbox(
        "Theme",
        options=["Light", "Dark"],
        index=0 if theme == 'Light' else 1,
        key="theme",
        help="Switch between Light and Dark themes",
    )

    st.markdown("---")
    st.markdown("#### ⚡ Quick Stats")
    if st.session_state.get('datasets'):
        st.metric("Datasets Loaded", len(st.session_state.datasets))
    if st.session_state.get('predictor') and getattr(st.session_state.predictor, 'trained', False):
        st.success("Model: Trained")
    else:
        st.warning("Model: Not trained")
    if st.session_state.get('action_plan') is not None:
        st.metric("Actions Generated", len(st.session_state.action_plan))
    if st.session_state.get('predictions_df') is not None:
        dfp = st.session_state.predictions_df
        if 'delay_risk_score' in dfp.columns:
            at_risk = int((dfp['delay_risk_score'] >= 0.6).sum())
            st.metric("High-Risk Orders", at_risk)

# ----------------------------
# Session state boot
# ----------------------------
for key, default in [
    ('datasets', None), ('features_df', None), ('predictor', None), ('metrics', None),
    ('predictions_df', None), ('action_plan', None), ('action_summary', None)
]:
    if key not in st.session_state: st.session_state[key] = default

# ----------------------------
# Header + Tabs Navigation
# ----------------------------
st.markdown('<div class="main-header">🚀 NexGen Predictive Delivery Optimizer</div>', unsafe_allow_html=True)

home_tab, data_tab, train_tab, predict_tab, actions_tab, impact_tab = st.tabs([
    "🏠 Home", "📊 Data Overview", "🤖 Model Training", "🔮 Predictions", "📋 Action Plan", "📈 Business Impact"
])

# ----------------------------
# Helper: Toasts
# ----------------------------

def toast_ok(msg: str):
    st.toast(f"✅ {msg}")

def toast_warn(msg: str):
    st.toast(f"⚠️ {msg}")

def toast_err(msg: str):
    st.toast(f"❌ {msg}")

# ----------------------------
# HOME
# ----------------------------
with home_tab:
    st.markdown(
        f"""
        <div style="text-align:center; padding: 1.5rem; background: linear-gradient(135deg, var(--brandA) 0%, var(--brandB) 100%); border-radius: 1rem; margin-bottom: 1.5rem;">
            <h2 style="color:white; margin:0">Predict delays • Act early • Protect margin</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    for icon, title, text, col in [
        ("📉", "Reduce Delays", "Predict high-risk shipments and intervene proactively", col1),
        ("💰", "Cut Costs", "Save ₹60K–₹90K monthly via smarter routing & carriers", col2),
        ("😊", "Boost CSAT", "Prevent churn with proactive comms & SLAs", col3),
    ]:
        with col:
            st.markdown(f"""
                <div class='feature-card'>
                    <div style='font-size:2rem; text-align:center'>{icon}</div>
                    <h4 style='text-align:center; color: var(--brandA); margin:.25rem 0'>{title}</h4>
                    <p style='text-align:center; opacity:.9'>{text}</p>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("### 📁 Required Data Files")
    required = {
        'orders.csv': ('📦','Order IDs, dates, distances, priorities'),
        'customers.csv': ('👥','Segments & LTV'),
        'warehouses.csv': ('🏭','Capacity & utilization'),
        'carriers.csv': ('🚚','On-time %, coverage'),
        'fleet.csv': ('🚗','Vehicle capabilities'),
        'tracking.csv': ('📍','Scan events & delays'),
        'costs.csv': ('💵','Linehaul / last-mile costs'),
    }
    c1, c2 = st.columns(2)
    for i, (name, (icon, desc)) in enumerate(required.items()):
        with (c1 if i % 2 == 0 else c2):
            exists = Path(f"data/{name}").exists()
            klass = 'success-box' if exists else 'warning-box'
            mark = '✅' if exists else '❌'
            st.markdown(f"""
            <div class='{klass}' style='padding:1rem; margin:.35rem 0'>
                <div style='font-size:1.25rem'>{icon} {mark} <b>{name}</b></div>
                <div style='opacity:.85; font-size:.95rem'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### 📂 Load / Reload Data")
    _, mid, _ = st.columns([1,2,1])
    with mid:
        if st.button("🔄 Load/Reload Data", use_container_width=True):
            try:
                data_dir = Path(__file__).parent / 'data'
                if not data_dir.exists():
                    st.markdown(
                        f"""
                        <div class='danger-box'>
                        <h4>Data directory not found</h4>
                        <code>{data_dir}</code>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    toast_err("Create a data/ folder and add CSVs")
                else:
                    datasets = load_and_prepare_data(str(data_dir))
                    if not datasets or 'orders' not in datasets:
                        st.markdown("<div class='danger-box'><b>orders.csv</b> is required.</div>", unsafe_allow_html=True)
                        toast_err("orders.csv missing")
                    else:
                        st.session_state.datasets = datasets
                        toast_ok(f"Loaded {len(datasets)} datasets")
                        st.balloons()
            except Exception as e:
                toast_err(str(e))

    if st.session_state.datasets is not None:
        st.info(f"✓ {len(st.session_state.datasets)} datasets loaded this session")
        cols = st.columns(len(st.session_state.datasets))
        icons = {'orders':'📦','customers':'👥','warehouses':'🏭','carriers':'🚚','fleet':'🚗','tracking':'📍','costs':'💵'}
        for i, (name, df) in enumerate(st.session_state.datasets.items()):
            with cols[i]:
                st.markdown(
                    f"""
                    <div class='stats-card'>
                        <div style='font-size:2rem'>{icons.get(name,'📄')}</div>
                        <h2>{len(df):,}</h2>
                        <p>{name.upper()}</p>
                        <p style='opacity:.9; font-size:.9rem'>{len(df.columns)} columns</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# ----------------------------
# DATA OVERVIEW
# ----------------------------
with data_tab:
    if st.session_state.datasets is None:
        st.warning("Load data on Home tab first.")
    else:
        datasets = st.session_state.datasets
        icons = {'orders':'📦','customers':'👥','warehouses':'🏭','carriers':'🚚','fleet':'🚗','tracking':'📍','costs':'💵'}
        options = [f"{icons.get(k,'📄')} {k.capitalize()}" for k in datasets.keys()]
        choice = st.selectbox("Choose a dataset", options)
        dataset_name = choice.split(' ')[1].lower()
        df = datasets[dataset_name]

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Rows", f"{len(df):,}")
        with c2: st.metric("Columns", len(df.columns))
        with c3: st.metric("Memory (MB)", f"{df.memory_usage(deep=True).sum()/1024**2:.1f}")
        with c4: st.metric("Missing %", f"{(df.isnull().sum().sum()/(len(df)*len(df.columns))*100):.1f}%")

        st.subheader("📋 Preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("📝 Column Info")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.values,
            'Non-Null': df.count().values,
            'Null %': ((1 - df.count()/len(df))*100).round(2).values,
        })
        st.dataframe(col_info, use_container_width=True)

        if len(df.select_dtypes(include=[np.number]).columns) > 0:
            st.subheader("📈 Summary Stats")
            st.dataframe(df.describe(), use_container_width=True)

        if dataset_name == 'orders' and 'is_delayed' in df.columns:
            st.subheader("⏱️ Delay Analysis")
            delay_rate = df['is_delayed'].mean()
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Delay Rate", f"{delay_rate:.1%}")
            with c2: st.metric("Delayed", int(df['is_delayed'].sum()))
            with c3: st.metric("On-Time", int((1-df['is_delayed']).sum()))
            if 'priority' in df.columns:
                st.markdown("#### By Priority")
                d = df.groupby('priority')['is_delayed'].agg(['mean','count'])
                d.columns = ['Delay Rate','Orders']
                d['Delay Rate'] = (d['Delay Rate']*100).round(2)
                st.dataframe(d, use_container_width=True)

# ----------------------------
# MODEL TRAINING
# ----------------------------
with train_tab:
    if st.session_state.datasets is None:
        st.warning("Load data on Home tab first.")
    else:
        # Step cards
        s1 = st.session_state.features_df is not None
        s2 = st.session_state.predictor is not None and getattr(st.session_state.predictor,'trained',False)
        s3 = st.session_state.metrics is not None
        cols = st.columns(3)
        for i, (done, label) in enumerate([(s1,'Feature Engineering'),(s2,'Model Training'),(s3,'Performance Metrics')]):
            with cols[i]:
                st.markdown(f"<div class='{'success-box' if done else 'warning-box'}'><b>{'✅' if done else '⏳'} {label}</b></div>", unsafe_allow_html=True)

        # Step 1: Build features
        st.markdown("### 1️⃣ Feature Engineering")
        center = st.container()
        with center:
            if st.button("🔧 Build Features", disabled=s1, use_container_width=True):
                try:
                    feats = engineer_features(st.session_state.datasets)
                    st.session_state.features_df = feats
                    toast_ok(f"Features ready: {len(feats):,} rows × {len(feats.columns)} cols")
                except Exception as e:
                    toast_err(f"Feature engineering failed: {e}")
        if st.session_state.features_df is None:
            st.stop()

        st.markdown("---")

        # Step 2: Train
        st.markdown("### 2️⃣ Model Training Configuration")
        c1, c2 = st.columns(2)
        with c1:
            model_type = st.radio(
                "Model",
                ["logistic", "random_forest"],
                format_func=lambda x: "⚡ Logistic Regression (fast, interpretable)" if x=="logistic" else "🌲 Random Forest (robust, accurate)",
            )
            if model_type == 'logistic':
                st.info("Best when you want speed and clarity of coefficients.")
            else:
                st.info("Recommended for best baseline performance + feature importance.")
        with c2:
            test_size = st.slider("Test size %", 10, 40, 20, 5)
            train_pct = 100 - test_size
            st.markdown(
                f"""
                <div style='display:flex;height:28px;border-radius:.5rem;overflow:hidden;margin-top:.35rem'>
                    <div style='width:{train_pct}%;background:linear-gradient(135deg,var(--brandA) 0%, var(--brandB) 100%);display:flex;align-items:center;justify-content:center;color:white;font-weight:700'>Train {train_pct}%</div>
                    <div style='width:{test_size}%;background:var(--warn);display:flex;align-items:center;justify-content:center;color:black;font-weight:700'>Test {test_size}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        _, mid, _ = st.columns([1,2,1])
        with mid:
            if st.button("🚀 Train Model", use_container_width=True):
                try:
                    feats = st.session_state.features_df
                    if 'is_delayed' not in feats.columns:
                        toast_err("Target 'is_delayed' not found in features")
                        st.stop()
                    predictor, metrics = train_and_evaluate_model(
                        feats, model_type=model_type, test_size=test_size/100
                    )
                    st.session_state.predictor = predictor
                    st.session_state.metrics = metrics
                    toast_ok(f"AUC {metrics['auc']:.3f} | Acc {metrics['accuracy']:.3f}")
                    st.balloons()
                except Exception as e:
                    toast_err(f"Training error: {e}")

        # Step 3: Metrics
        if st.session_state.metrics:
            st.markdown("---")
            st.markdown("### 3️⃣ Model Performance")
            m = st.session_state.metrics
            auc = m['auc']
            mood = '#36d399' if auc>=0.85 else ('#fbbd23' if auc>=0.75 else '#f87272')
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg,{mood}33 0%, {mood}11 100%); border:2px solid {mood}; border-radius:1rem; padding:1.25rem; text-align:center;">
                    <h3 style='margin:.25rem 0'>AUC-ROC</h3>
                    <h1 style='margin:.25rem 0; color:{mood}'>{auc:.1%}</h1>
                    <div style='opacity:.85'>{'🌟 Excellent' if auc>=0.85 else ('✅ Good' if auc>=0.75 else '⚠️ Needs improvement')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            c1, c2, c3, c4 = st.columns(4)
            with c1: st.markdown(f"""<div class='stats-card'><h2>{m['precision']:.1%}</h2><p>Precision</p></div>""", unsafe_allow_html=True)
            with c2: st.markdown(f"""<div class='stats-card'><h2>{m['recall']:.1%}</h2><p>Recall</p></div>""", unsafe_allow_html=True)
            with c3: st.markdown(f"""<div class='stats-card'><h2>{m['f1']:.1%}</h2><p>F1-Score</p></div>""", unsafe_allow_html=True)
            with c4: st.markdown(f"""<div class='stats-card'><h2>{m['accuracy']:.1%}</h2><p>Accuracy</p></div>""", unsafe_allow_html=True)

            left, right = st.columns(2)
            with left:
                st.markdown("#### Confusion Matrix")
                fig_cm = plot_confusion_matrix(m)
                st.plotly_chart(fig_cm, use_container_width=True)
            with right:
                if st.session_state.predictor is not None:
                    st.markdown("#### Top Features")
                    try:
                        imp_df = st.session_state.predictor.get_feature_importance(top_n=15)
                        fig_imp = plot_feature_importance(imp_df, top_n=15)
                        st.plotly_chart(fig_imp, use_container_width=True)
                    except Exception:
                        st.info("Feature importance not available for this model.")

            # Optional: SHAP Explainability
            with st.expander("🧠 Why did the model predict that? (SHAP)"):
                try:
                    import shap  # optional dependency
                    feats = st.session_state.features_df
                    model = getattr(st.session_state.predictor, 'model', None)
                    if model is None:
                        raise RuntimeError("Underlying model not exposed as 'predictor.model'")

                    # Use a small background sample for speed
                    numeric_cols = feats.select_dtypes(include=[np.number]).columns
                    X = feats[numeric_cols].fillna(0)
                    bg = shap.utils.sample(X, 200)
                    explainer = shap.Explainer(model, bg)
                    sv = explainer(X.sample(n=min(500, len(X)), random_state=42))
                    st.write("Global feature impact (sampled):")
                    shap.plots.bar(sv, max_display=12, show=False)
                    st.pyplot(bbox_inches='tight')
                except Exception as e:
                    st.info("Install `shap` and ensure predictor exposes `.model` for explainability.\n"+str(e))

            st.markdown("### Save Model")
            if st.button("💾 Save Model for Production"):
                try:
                    st.session_state.predictor.save()
                    st.success("Model saved to models/model.joblib")
                    toast_ok("Model saved")
                except Exception as e:
                    toast_err(str(e))

# ----------------------------
# PREDICTIONS
# ----------------------------
with predict_tab:
    if st.session_state.predictor is None or not getattr(st.session_state.predictor,'trained',False):
        st.warning("Train a model first (Model Training tab)")
    elif st.session_state.features_df is None:
        st.warning("Build features first (Model Training tab)")
    else:
        st.markdown("### 🎯 Generate Risk Scores")
        btn_text = "🔄 Regenerate Risk Scores" if st.session_state.predictions_df is not None else "🎯 Generate Risk Scores"
        if st.button(btn_text, use_container_width=True):
            try:
                preds = predict_new_orders(st.session_state.predictor, st.session_state.features_df)
                st.session_state.predictions_df = preds
                hi = int((preds['delay_risk_score'] >= 0.6).sum()) if 'delay_risk_score' in preds.columns else 0
                toast_ok(f"Scored {len(preds):,} orders • {hi} high-risk")
                st.balloons()
            except Exception as e:
                toast_err(f"Prediction error: {e}")

        if st.session_state.predictions_df is not None:
            predictions_df = st.session_state.predictions_df
            st.markdown("---")
            st.markdown("### 📊 Risk Summary Dashboard")

            if 'delay_risk_score' not in predictions_df.columns:
                st.error("Column 'delay_risk_score' missing from predictions.")
            else:
                high = int((predictions_df['delay_risk_score'] >= 0.6).sum())
                med = int(((predictions_df['delay_risk_score'] >= 0.3) & (predictions_df['delay_risk_score'] < 0.6)).sum())
                low = int((predictions_df['delay_risk_score'] < 0.3).sum())
                avg = float(predictions_df['delay_risk_score'].mean())

                c1, c2, c3, c4 = st.columns(4)
                with c1: st.markdown(f"""<div class='metric-card' style='text-align:center'><div style='font-size:2rem'>📦</div><div style='font-size:2.25rem;font-weight:800;color:var(--brandA)'>{len(predictions_df):,}</div><div>Total Scored</div></div>""", unsafe_allow_html=True)
                with c2: st.markdown(f"""<div class='danger-box' style='text-align:center'><div style='font-size:2rem'>🔴</div><div style='font-size:2.25rem;font-weight:800;color:var(--danger)'>{high}</div><div>High Risk ({(high/len(predictions_df))*100:.1f}%)</div></div>""", unsafe_allow_html=True)
                with c3: st.markdown(f"""<div class='warning-box' style='text-align:center'><div style='font-size:2rem'>🟡</div><div style='font-size:2.25rem;font-weight:800'>{med}</div><div>Medium Risk ({(med/len(predictions_df))*100:.1f}%)</div></div>""", unsafe_allow_html=True)
                with c4: st.markdown(f"""<div class='success-box' style='text-align:center'><div style='font-size:2rem'>🟢</div><div style='font-size:2.25rem;font-weight:800;color:var(--success)'>{low}</div><div>Low Risk ({(low/len(predictions_df))*100:.1f}%)</div></div>""", unsafe_allow_html=True)

                st.markdown(f"""<div class='metric-card' style='text-align:center'><h4>Average Risk</h4><h1 style='margin:.25rem 0; color:var(--brandA)'>{avg:.1%}</h1></div>""", unsafe_allow_html=True)

                st.markdown("#### Distribution")
                fig = plot_risk_distribution(predictions_df)
                st.plotly_chart(fig, use_container_width=True)

                if 'priority' in predictions_df.columns:
                    st.markdown("#### Risk by Priority")
                    figp = plot_risk_by_priority(predictions_df)
                    if figp: st.plotly_chart(figp, use_container_width=True)

                # High-risk table + Drill-down modal
                st.markdown("#### High-Risk Orders")
                high_df = predictions_df[predictions_df['delay_risk_score'] >= 0.6].copy()
                if len(high_df) == 0:
                    st.info("No high-risk orders found.")
                else:
                    display_cols = [c for c in ['order_id','delay_risk_score','risk_category','priority','distance_km','carrier_id'] if c in high_df.columns]
                    st.dataframe(high_df[display_cols].head(50), use_container_width=True)

                    # Drill-down selector
                    sel_col1, sel_col2 = st.columns([2,1])
                    with sel_col1:
                        oid = st.selectbox("Inspect order_id", options=high_df['order_id'].astype(str).unique())
                    with sel_col2:
                        if st.button("🔎 View Details"):
                            with st.modal(f"Order Details — {oid}"):
                                try:
                                    oid_mask = predictions_df['order_id'].astype(str) == str(oid)
                                    order_row = predictions_df[oid_mask].iloc[0]
                                    st.subheader("Risk & Prediction")
                                    st.json({
                                        'order_id': order_row.get('order_id'),
                                        'risk_score': float(order_row.get('delay_risk_score', np.nan)),
                                        'risk_category': order_row.get('risk_category'),
                                        'priority': order_row.get('priority'),
                                        'carrier_id': order_row.get('carrier_id'),
                                        'distance_km': float(order_row.get('distance_km', np.nan)) if 'distance_km' in predictions_df.columns else None,
                                    })

                                    # Join raw info if available
                                    if st.session_state.datasets and 'orders' in st.session_state.datasets:
                                        raw = st.session_state.datasets['orders']
                                        more = raw[raw['order_id'].astype(str) == str(oid)]
                                        if len(more) > 0:
                                            st.subheader("Raw Order Snapshot")
                                            st.dataframe(more.head(1), use_container_width=True)

                                except Exception as e:
                                    st.error(str(e))

# ----------------------------
# ACTION PLAN
# ----------------------------
with actions_tab:
    if st.session_state.predictions_df is None:
        st.warning("Generate predictions first.")
    else:
        st.header("📋 Action Plan & Prescriptions")
        min_risk = st.slider("Minimum Risk Threshold", 0.0, 1.0, 0.5, 0.05)
        if st.button("📝 Generate Action Plan", use_container_width=True):
            try:
                ap, summ = generate_prescriptions(
                    st.session_state.predictions_df,
                    datasets=st.session_state.datasets,
                    min_risk=min_risk,
                )
                st.session_state.action_plan = ap
                st.session_state.action_summary = summ
                toast_ok(f"Generated {len(ap)} actions")
            except Exception as e:
                toast_err(str(e))

        if st.session_state.action_plan is not None and len(st.session_state.action_plan) > 0:
            ap = st.session_state.action_plan
            summ = st.session_state.action_summary
            st.subheader("📊 Summary")
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("At-Risk Orders", summ['total_at_risk_orders'])
            with c2: st.metric("Total Actions", summ['total_actions'])
            with c3: st.metric("High Priority", summ['high_priority_actions'])

            st.markdown("#### Actions by Type")
            fig = plot_actions_by_type(ap)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Detailed Action Plan")
            prio = st.multiselect("Filter by Priority", ["High","Medium","Low"], default=["High","Medium","Low"])
            filtered = ap[ap['action_priority'].isin(prio)] if 'action_priority' in ap.columns else ap
            st.dataframe(filtered, use_container_width=True, height=420)

            st.markdown("#### Export")
            csv_data = export_action_plan_csv(filtered)
            st.download_button("📥 Download CSV", data=csv_data, file_name="nexgen_action_plan.csv", mime="text/csv")

# ----------------------------
# BUSINESS IMPACT
# ----------------------------
with impact_tab:
    if st.session_state.action_plan is None or len(st.session_state.action_plan) == 0:
        st.warning("Generate an action plan first.")
    else:
        st.header("📈 Business Impact Analysis")
        summ = st.session_state.action_summary
        metrics = st.session_state.metrics or {}
        impact = calculate_business_impact(summ)

        st.subheader("💰 Financial Impact (Monthly)")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Baseline Delay Cost", format_currency(impact['baseline_delay_cost']))
        with c2:
            st.metric("Estimated Savings", format_currency(impact['estimated_savings']), delta=f"{impact['estimated_savings']/impact['baseline_delay_cost']:.0%}")
        with c3:
            st.metric("Total Business Value", format_currency(impact['total_business_value']))

        st.subheader("🎯 Operational Impact")
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Delays Prevented", impact['prevented_delays'])
        with c2: st.metric("Churn Prevented", f"{impact['prevented_churn_customers']} customers")
        with c3: st.metric("ROI Multiplier", f"{impact['roi_multiplier']:.1f}x")

        st.markdown("---")
        st.subheader("📄 Executive Summary")
        summary_text = create_executive_summary(metrics, summ, impact)
        st.markdown(summary_text)
        st.download_button("📥 Download Summary", data=summary_text, file_name="nexgen_executive_summary.md", mime="text/markdown")

        with st.expander("ℹ️ Assumptions"):
            st.markdown(
                """
                - Average delay cost: ₹500 per order
                - Average order value: ₹2,000
                - Intervention recovery rate: 60%
                - Customer churn rate: 5% of delayed orders
                - Churn cost: ₹10,000 per lost customer
                - ROI multiplier: 5.0×
                (Tune these in `src/utils.py` for your business.)
                """
            )

# EOF
