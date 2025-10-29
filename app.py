"""
NexGen Predictive Delivery Optimizer — Pro (Dark, Self-Contained)
Tailored to: orders.csv, delivery_performance.csv, routes_distance.csv

What it does
- Loads your datasets (auto-detect join key, delay flag, and distance column)
- Cleans + engineers features
- Trains an ML model (Logistic / RandomForest) with metrics + charts
- Scores orders and groups risk (Low/Medium/High)
- Generates a simple, practical action plan
- Estimates business impact and exports a summary

Run:
  streamlit run app.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib


# ---------------------------------------------------------------------------
# Page config + Dark Theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NexGen Delivery Optimizer — Pro",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
:root {
  --bg: #0b1020; --text: #ecf2ff; --muted: #9aa4c7; --card: #111735;
  --brandA: #7aa2ff; --brandB: #a78bfa;
  --success: #36d399; --warn: #fbbd23; --danger: #f87272; --info: #22d3ee;
}
body, .stApp { background: var(--bg); color: var(--text); }
.block-container { padding-top: 1rem; }
.block-container h1, h2, h3, h4, h5, h6, p, label, span { color: var(--text) !important; }
.main-header {
  font-size: 3rem; font-weight: 800;
  background: linear-gradient(135deg, var(--brandA) 0%, var(--brandB) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  text-align:center; padding: 1.1rem 0; margin-bottom: .6rem;
}
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a 0%, #1f2937 100%); }
[data-testid="stSidebar"] * { color: #e5e7eb !important; }
.metric-card { background:#0f1633; padding:1rem; border-radius:1rem; border-left:5px solid var(--brandA); box-shadow:inset 0 0 0 1px rgba(255,255,255,.05); }
.success-box { background: rgba(54,211,153,.12); padding:1rem; border-radius:1rem; border-left:5px solid var(--success); }
.warning-box { background: rgba(251,189,35,.12); padding:1rem; border-radius:1rem; border-left:5px solid var(--warn); }
.danger-box  { background: rgba(248,114,114,.12); padding:1rem; border-radius:1rem; border-left:5px solid var(--danger); }
.feature-card { background:#0f1633; padding:1.1rem; border-radius:1rem; box-shadow:inset 0 0 0 1px rgba(255,255,255,.06); border-top:4px solid var(--brandA); }
.stats-card { background: linear-gradient(135deg, var(--brandA) 0%, var(--brandB) 100%); color:white; padding:1.1rem; border-radius:1rem; text-align:center; }
.stButton > button {
  background: linear-gradient(135deg, var(--brandA) 0%, var(--brandB) 100%);
  color: white; font-weight: 600; border: none; border-radius: .55rem; padding: .65rem 1rem;
  box-shadow: 0 4px 10px rgba(16,24,64,.5);
}
[data-testid="stMetricValue"] { color: var(--brandA); }
hr { border:none; height:1px; background: #1e2a4b; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers: toasts + detection
# ---------------------------------------------------------------------------
def toast_ok(msg: str): st.toast(f"✅ {msg}")
def toast_warn(msg: str): st.toast(f"⚠️ {msg}")
def toast_err(msg: str): st.toast(f"❌ {msg}")

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

@st.cache_data(show_spinner=False)
def read_csv_smart(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _normalize_cols(df)

def detect_common_key(*dfs: pd.DataFrame) -> Optional[str]:
    # candidates by preference
    cands = ["order_id", "shipment_id", "tracking_id", "consignment_id", "delivery_id"]
    cols = set.intersection(*[set(d.columns) for d in dfs if d is not None and len(d) > 0])
    for key in cands:
        if key in cols:
            return key
    # heuristic: first column ending with _id
    for c in cols:
        if c.endswith("_id"):
            return c
    return None

def detect_delay_column(df: pd.DataFrame) -> Tuple[str, pd.Series]:
    # try boolean/0-1 style
    for cand in ["is_delayed", "delayed", "delay_flag", "late", "delayed_flag"]:
        if cand in df.columns:
            s = df[cand]
            # map yes/no or strings to 0/1 if needed
            if s.dtype == object:
                s1 = s.astype(str).str.lower()
                return cand, s1.isin(["1", "true", "yes", "y", "delayed", "late"]).astype(int)
            return cand, s.fillna(0).astype(int)
    # try status column
    for cand in ["status", "delivery_status", "current_status"]:
        if cand in df.columns:
            s = df[cand].astype(str).str.lower()
            return cand, s.isin(["delayed", "late", "fail", "failed", "in_transit_late"]).astype(int)
    raise ValueError("Could not detect a delay label column in delivery_performance.csv")

def detect_distance_column(df: pd.DataFrame) -> Tuple[str, pd.Series]:
    # prefer *_km
    km_cols = [c for c in df.columns if re.search(r"(distance|route|km)", c)]
    if not km_cols:
        raise ValueError("No distance-like column found in routes_distance.csv")
    # choose best candidate
    for c in ["distance_km", "route_distance_km", "km", "distance", "route_distance"]:
        if c in df.columns:
            return c, pd.to_numeric(df[c], errors="coerce")
    # fallback: first numeric among km_cols
    for c in km_cols:
        try:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                return c, s
        except Exception:
            continue
    raise ValueError("Could not parse any numeric distance column")


# ---------------------------------------------------------------------------
# Data loading tailored to your files
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_datasets(data_dir: Path) -> Dict[str, pd.DataFrame]:
    files = {
        "orders": "orders.csv",
        "delivery_performance": "delivery_performance.csv",
        "routes_distance": "routes_distance.csv",
        # Optional extras will be shown in Data tab if present:
        "vehicle_fleet": "vehicle_fleet.csv",
        "warehouse_inventory": "warehouse_inventory.csv",
        "cost_breakdown": "cost_breakdown.csv",
        "customer_feedback": "customer_feedback.csv",
    }
    ds = {}
    for key, fname in files.items():
        p = data_dir / fname
        if p.exists():
            try:
                ds[key] = read_csv_smart(p)
            except Exception as e:
                st.warning(f"Couldn't read {fname}: {e}")
    if "orders" not in ds or "delivery_performance" not in ds or "routes_distance" not in ds:
        missing = [k for k in ["orders", "delivery_performance", "routes_distance"] if k not in ds]
        raise FileNotFoundError(f"Required file(s) missing: {', '.join(missing)} in {data_dir}")
    return ds


# ---------------------------------------------------------------------------
# Feature engineering (self-contained)
# ---------------------------------------------------------------------------
def build_feature_table(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    orders = datasets["orders"].copy()
    perf = datasets["delivery_performance"].copy()
    routes = datasets["routes_distance"].copy()

    # normalize date
    for cand in ["order_date", "date", "created_date", "placed_at"]:
        if cand in orders.columns:
            orders["order_date"] = pd.to_datetime(orders[cand], errors="coerce")
            break
    if "order_date" not in orders.columns:
        orders["order_date"] = pd.NaT

    # detect target from performance
    delay_col, delay_series = detect_delay_column(perf)
    perf["_is_delayed"] = delay_series

    # detect distance from routes
    dist_col, dist_series = detect_distance_column(routes)
    routes["_distance_km"] = dist_series

    # detect join key
    key = detect_common_key(orders, perf, routes)
    if key is None:
        raise ValueError("Couldn't detect a common key (e.g., order_id). Add a column like 'order_id' to all files.")

    # reduce perf/routes to needed columns
    perf_keep = [key, "_is_delayed"]
    routes_keep = [key, "_distance_km"]
    perf_small = perf[[c for c in perf_keep if c in perf.columns]].drop_duplicates(subset=[key])
    routes_small = routes[[c for c in routes_keep if c in routes.columns]].drop_duplicates(subset=[key])

    # merge
    base = orders.copy()
    if key not in base.columns:
        raise ValueError(f"Join key '{key}' not found in orders.csv")
    base = base.merge(perf_small, on=key, how="left")
    base = base.merge(routes_small, on=key, how="left")

    # derive priority if exists or infer
    priority_col = None
    for c in ["priority", "order_priority", "sla_priority"]:
        if c in base.columns:
            priority_col = c
            break

    # engineer temporal features
    if "order_date" in base.columns:
        base["order_dow"] = base["order_date"].dt.dayofweek
        base["order_month"] = base["order_date"].dt.month
        base["order_week"] = base["order_date"].dt.isocalendar().week.astype("Int64")

    # finalize columns
    rename_map = {
        key: "order_id",
        "_is_delayed": "is_delayed",
        "_distance_km": "distance_km",
    }
    base = base.rename(columns=rename_map)

    # ensure target presence
    if "is_delayed" not in base.columns:
        raise ValueError("Target 'is_delayed' could not be created from delivery_performance.csv")

    # sanity types
    base["is_delayed"] = base["is_delayed"].fillna(0).astype(int)
    if "distance_km" in base.columns:
        base["distance_km"] = pd.to_numeric(base["distance_km"], errors="coerce")

    # optional derived metrics
    if "distance_km" in base.columns:
        base["distance_bin"] = pd.cut(
            base["distance_km"],
            bins=[-0.01, 50, 200, 500, 2000, 1e9],
            labels=["0-50", "50-200", "200-500", "500-2000", "2000+"],
        )

    # keep a tidy set
    keep_cols = ["order_id", "is_delayed", "distance_km", "distance_bin", "order_dow", "order_month", "order_week"]
    if priority_col and priority_col in base.columns:
        keep_cols.append(priority_col)
        base = base.rename(columns={priority_col: "priority"})
    base = base[[c for c in keep_cols if c in base.columns]].copy()

    return base


# ---------------------------------------------------------------------------
# Modeling (self-contained)
# ---------------------------------------------------------------------------
class ModelBundle:
    def __init__(self, model_type: str, pipeline: Pipeline, feature_names: List[str]):
        self.model_type = model_type
        self.pipeline = pipeline
        self.feature_names = feature_names
        self.trained = True  # for UI compatibility

    def save(self, path: Path = Path("models/model.joblib")):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        clf = self.pipeline.named_steps["clf"]
        pre: ColumnTransformer = self.pipeline.named_steps["pre"]
        # feature names after preprocessing
        num_cols = pre.transformers_[0][2] if pre.transformers_ else []
        ohe = pre.named_transformers_.get("cat", None)
        cat_cols = []
        if ohe is not None and hasattr(ohe.named_steps["oh"], "get_feature_names_out"):
            cat_cols = ohe.named_steps["oh"].get_feature_names_out().tolist()
        names = list(num_cols) + list(cat_cols)

        if hasattr(clf, "feature_importances_"):
            vals = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            vals = np.abs(clf.coef_).ravel()
        else:
            return pd.DataFrame(columns=["feature", "importance"])

        n = min(len(vals), len(names))
        imp = pd.DataFrame({"feature": names[:n], "importance": vals[:n]}).sort_values(
            "importance", ascending=False
        )
        return imp.head(top_n)

def build_pipeline(model_type: str, X: pd.DataFrame) -> Pipeline:
    nums = X.select_dtypes(include=[np.number]).columns.tolist()
    cats = [c for c in X.columns if c not in nums]

    num_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler(with_mean=False))]
    )
    cat_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", sparse=False))]
    )
    pre = ColumnTransformer([("num", num_pipe, nums), ("cat", cat_pipe, cats)], remainder="drop")

    clf = LogisticRegression(max_iter=300) if model_type == "logistic" else RandomForestClassifier(
        n_estimators=250, random_state=42, n_jobs=-1
    )
    return Pipeline([("pre", pre), ("clf", clf)])

def train_and_evaluate(features_df: pd.DataFrame, model_type: str, test_size: float) -> Tuple[ModelBundle, Dict[str, float]]:
    if "is_delayed" not in features_df.columns:
        raise ValueError("Target 'is_delayed' missing in features")

    y = features_df["is_delayed"].astype(int).values
    X = features_df.drop(columns=["is_delayed", "order_id"], errors="ignore")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    pipe = build_pipeline(model_type, X)
    pipe.fit(Xtr, ytr)

    proba = pipe.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "auc": float(roc_auc_score(yte, proba)) if len(np.unique(yte)) > 1 else np.nan,
        "precision": float(precision_score(yte, pred, zero_division=0)),
        "recall": float(recall_score(yte, pred, zero_division=0)),
        "f1": float(f1_score(yte, pred, zero_division=0)),
        "accuracy": float(accuracy_score(yte, pred)),
    }
    bundle = ModelBundle(model_type, pipe, X.columns.tolist())
    return bundle, metrics

def score_orders(bundle: ModelBundle, features_df: pd.DataFrame) -> pd.DataFrame:
    X = features_df.drop(columns=["is_delayed"], errors="ignore")
    scores = bundle.pipeline.predict_proba(X)[:, 1]
    out = features_df.copy()
    out["delay_risk_score"] = scores
    out["risk_category"] = pd.cut(out["delay_risk_score"], [-0.01, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"])
    return out


# ---------------------------------------------------------------------------
# Simple prescriptions + impact (self-contained)
# ---------------------------------------------------------------------------
def generate_prescriptions(pred_df: pd.DataFrame, min_risk: float = 0.5) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = pred_df[pred_df["delay_risk_score"] >= min_risk].copy()
    if df.empty:
        return pd.DataFrame(), {"total_at_risk_orders": 0, "total_actions": 0, "high_priority_actions": 0}
    rows = []
    for _, r in df.iterrows():
        action = "Expedite dispatch" if r.get("priority", "Normal") in ("High", "Urgent") else "Notify customer w/ new ETA"
        if r.get("distance_km", np.nan) and r.get("distance_km", 0) > 500:
            action = "Reroute via nearest hub"
        rows.append(
            {
                "order_id": r.get("order_id", _),
                "risk_score": float(r["delay_risk_score"]),
                "risk_category": r.get("risk_category", None),
                "action": action,
                "action_priority": "High" if r["delay_risk_score"] >= 0.7 else "Medium",
                "owner": "Ops",
            }
        )
    ap = pd.DataFrame(rows)
    summary = {
        "total_at_risk_orders": int(len(df)),
        "total_actions": int(len(ap)),
        "high_priority_actions": int((ap["action_priority"] == "High").sum()),
    }
    return ap, summary

def calculate_business_impact(summary: Dict[str, int]) -> Dict[str, float]:
    avg_delay_cost = 500.0
    recovery_rate = 0.6
    churn_rate = 0.05
    churn_cost = 10000.0

    n = summary.get("total_at_risk_orders", 0)
    baseline_delay_cost = n * avg_delay_cost
    prevented_delays = int(n * recovery_rate)
    prevented_churn = int(n * churn_rate * recovery_rate)
    estimated_savings = prevented_delays * avg_delay_cost
    total_business_value = estimated_savings + prevented_churn * churn_cost

    return {
        "baseline_delay_cost": baseline_delay_cost,
        "estimated_savings": estimated_savings,
        "total_business_value": total_business_value,
        "prevented_delays": prevented_delays,
        "prevented_churn_customers": prevented_churn,
        "roi_multiplier": 5.0,
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: .9rem; background: rgba(255,255,255,0.06); border-radius: 1rem; margin-bottom: 1rem;">
            <div style="font-size:2rem">📦</div>
            <div style="font-weight:700; letter-spacing:.2px">NexGen Optimizer</div>
            <div style="opacity:.75; font-size:.9rem">Pro • Dark</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "datasets" in st.session_state and st.session_state.get("datasets"):
        st.metric("Datasets", len(st.session_state["datasets"]))
    if st.session_state.get("model") is not None:
        st.success("Model: Trained")
    else:
        st.warning("Model: Not trained")
    if st.session_state.get("pred") is not None:
        hi = int((st.session_state["pred"]["delay_risk_score"] >= 0.6).sum())
        st.metric("High-risk", hi)


# ---------------------------------------------------------------------------
# Header + Tabs
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">🚀 NexGen Predictive Delivery Optimizer</div>', unsafe_allow_html=True)
home_tab, data_tab, train_tab, predict_tab, actions_tab, impact_tab = st.tabs(
    ["🏠 Home", "📊 Data", "🤖 Train", "🔮 Predict", "📋 Actions", "📈 Impact"]
)


# ---------------------------------------------------------------------------
# HOME
# ---------------------------------------------------------------------------
with home_tab:
    st.markdown(
        """
        <div style="text-align:center; padding: 1.1rem; background: linear-gradient(135deg, var(--brandA) 0%, var(--brandB) 100%); border-radius: 1rem; margin-bottom: 1rem;">
            <h3 style="margin:0;color:white">Predict delays • Act early • Protect margin</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3, gap="large")
    for col, (icon, title, text) in zip(
        [c1, c2, c3],
        [
            ("📉", "Reduce Delays", "Predict high-risk shipments; intervene proactively"),
            ("💰", "Cut Costs", "Optimize distance + priority tradeoffs"),
            ("😊", "Boost CSAT", "Proactive comms and SLA recovery"),
        ],
    ):
        with col:
            st.markdown(
                f"""
                <div class='feature-card'>
                    <div style='font-size:1.8rem; text-align:center'>{icon}</div>
                    <h4 style='text-align:center; color: var(--brandA); margin:.25rem 0'>{title}</h4>
                    <p style='text-align:center; opacity:.9'>{text}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### 📁 Required Data Files")
    req = {
        "orders.csv": ("📦", "Order IDs, dates, optional priority"),
        "delivery_performance.csv": ("📈", "Delivery status / delay labels"),
        "routes_distance.csv": ("🛣️", "Route distance (km) per order"),
    }
    c1, c2 = st.columns(2)
    data_dir = Path(__file__).parent / "data"
    for i, (fname, (icon, desc)) in enumerate(req.items()):
        p = data_dir / fname
        with (c1 if i % 2 == 0 else c2):
            ok = p.exists()
            klass = "success-box" if ok else "danger-box"
            mark = "✅" if ok else "❌"
            st.markdown(
                f"""
                <div class='{klass}' style='padding:.9rem; margin:.35rem 0'>
                    <div style='font-size:1.15rem'>{icon} {mark} <b>{fname}</b></div>
                    <div style='opacity:.85; font-size:.95rem'>{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### 📂 Load / Reload Data")
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        if st.button("🔄 Load Data", use_container_width=True):
            try:
                datasets = load_datasets(data_dir)
                st.session_state["datasets"] = datasets
                toast_ok(f"Loaded {len(datasets)} datasets")
                st.balloons()
            except Exception as e:
                toast_err(str(e))

    if st.session_state.get("datasets"):
        ds = st.session_state["datasets"]
        icons = {
            "orders": "📦",
            "delivery_performance": "📈",
            "routes_distance": "🛣️",
            "vehicle_fleet": "🚗",
            "warehouse_inventory": "🏭",
            "cost_breakdown": "💵",
            "customer_feedback": "🗣️",
        }
        st.info(f"✓ {len(ds)} datasets loaded")
        cols = st.columns(len(ds))
        for (name, df), col in zip(ds.items(), cols):
            with col:
                st.markdown(
                    f"""
                    <div class='stats-card'>
                        <div style='font-size:1.5rem'>{icons.get(name,'📄')}</div>
                        <h2 style='margin:.2rem 0'>{len(df):,}</h2>
                        <p style='margin:0'>{name.replace('_',' ').title()}</p>
                        <p style='opacity:.9; font-size:.85rem; margin:.2rem 0'>{len(df.columns)} columns</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------
with data_tab:
    ds = st.session_state.get("datasets")
    if not ds:
        st.warning("Load data on the Home tab first.")
    else:
        options = list(ds.keys())
        name = st.selectbox("Dataset", options, format_func=lambda s: s.replace("_", " ").title())
        df = ds[name]

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Rows", f"{len(df):,}")
        with c2: st.metric("Columns", len(df.columns))
        with c3: st.metric("Memory (MB)", f"{df.memory_usage(deep=True).sum()/1024**2:.1f}")
        with c4: st.metric("Missing %", f"{(df.isnull().sum().sum()/max(1,len(df)*len(df.columns))*100):.1f}%")

        with st.expander("Preview", expanded=True):
            st.dataframe(df.head(30), use_container_width=True)

        with st.expander("Column Info", expanded=False):
            ci = pd.DataFrame({
                "Column": df.columns,
                "Type": df.dtypes.values,
                "Non-Null": df.count().values,
                "Null %": ((1 - df.count()/len(df))*100).round(2).values
            })
            st.dataframe(ci, use_container_width=True)

        # Delay trend (if status + date exist)
        if name in ("orders", "delivery_performance"):
            tmp = df.copy()
            date_col = None
            for c in ["order_date", "date", "created_date", "placed_at"]:
                if c in tmp.columns:
                    date_col = c
                    break
            if date_col and ("is_delayed" in tmp.columns or "status" in tmp.columns):
                if "is_delayed" not in tmp.columns:
                    _, s = detect_delay_column(tmp)
                    tmp["is_delayed"] = s
                tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
                trend = (
                    tmp.dropna(subset=[date_col]).set_index(date_col)["is_delayed"].resample("W").mean().reset_index()
                )
                fig = px.line(trend, x=date_col, y="is_delayed", markers=True, labels={"is_delayed": "Delay Rate"})
                fig.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------------
with train_tab:
    ds = st.session_state.get("datasets")
    if not ds:
        st.warning("Load data first.")
    else:
        # Build features
        if st.button("🔧 Build Features", use_container_width=True):
            try:
                feats = build_feature_table(ds)
                st.session_state["features"] = feats
                st.success(f"Features ready: {len(feats):,} rows × {len(feats.columns)} cols")
            except Exception as e:
                toast_err(str(e))

        feats = st.session_state.get("features")
        if feats is not None:
            with st.expander("Feature Sample"):
                st.dataframe(feats.head(20), use_container_width=True)

            # Train config
            st.markdown("### Training")
            c1, c2 = st.columns(2)
            with c1:
                algo = st.radio(
                    "Algorithm",
                    ["logistic", "random_forest"],
                    index=1,
                    format_func=lambda x: "⚡ Logistic (fast, interpretable)" if x == "logistic" else "🌲 RandomForest (robust)",
                )
            with c2:
                test_pct = st.slider("Test size %", 10, 40, 20, 5)

            if st.button("🚀 Train Model", use_container_width=True):
                try:
                    model, metrics = train_and_evaluate(feats, algo, test_pct / 100)
                    st.session_state["model"] = model
                    st.session_state["metrics"] = metrics
                    toast_ok(f"AUC {metrics['auc']:.3f} | Acc {metrics['accuracy']:.3f}")
                    st.balloons()
                except Exception as e:
                    toast_err(str(e))

        # Metrics
        metrics = st.session_state.get("metrics")
        model = st.session_state.get("model")
        if metrics and model:
            m = metrics
            auc = m["auc"]
            mood = "#36d399" if (auc is not None and auc >= 0.85) else ("#fbbd23" if auc and auc >= 0.75 else "#f87272")
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg,{mood}33 0%, {mood}11 100%); border:2px solid {mood}; border-radius:1rem; padding:1rem; text-align:center;">
                    <h3 style='margin:.2rem 0'>AUC-ROC</h3>
                    <h1 style='margin:.2rem 0; color:{mood}'>{(auc if auc==auc else 0):.1%}</h1>
                    <div style='opacity:.85'>{'🌟 Excellent' if auc and auc>=0.85 else ('✅ Good' if auc and auc>=0.75 else '⚠️ Needs improvement')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            a, b, c, d = st.columns(4)
            with a: st.markdown(f"""<div class='stats-card'><h2>{m['precision']:.1%}</h2><p>Precision</p></div>""", unsafe_allow_html=True)
            with b: st.markdown(f"""<div class='stats-card'><h2>{m['recall']:.1%}</h2><p>Recall</p></div>""", unsafe_allow_html=True)
            with c: st.markdown(f"""<div class='stats-card'><h2>{m['f1']:.1%}</h2><p>F1</p></div>""", unsafe_allow_html=True)
            with d: st.markdown(f"""<div class='stats-card'><h2>{m['accuracy']:.1%}</h2><p>Accuracy</p></div>""", unsafe_allow_html=True)

            # Confusion matrix (recompute quickly on same split)
            y = feats["is_delayed"].astype(int).values
            X = feats.drop(columns=["is_delayed", "order_id"], errors="ignore")
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_pct / 100, random_state=42, stratify=y)
            proba = model.pipeline.predict_proba(Xte)[:, 1]
            pred = (proba >= 0.5).astype(int)
            cm = confusion_matrix(yte, pred)
            fig_cm = go.Figure(data=go.Heatmap(z=cm, x=["Pred 0", "Pred 1"], y=["True 0", "True 1"],
                                               text=cm, texttemplate="%{text}", colorscale="Blues"))
            fig_cm.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
            st.plotly_chart(fig_cm, use_container_width=True)

            # Feature importance
            st.markdown("#### Top Features")
            imp = model.get_feature_importance(20)
            if len(imp) > 0:
                st.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h"), use_container_width=True)
            else:
                st.info("Importance not available for this model.")

            # Save model
            if st.button("💾 Save model", use_container_width=False):
                try:
                    model.save()
                    st.success("Saved to models/model.joblib")
                except Exception as e:
                    toast_err(str(e))


# ---------------------------------------------------------------------------
# PREDICT
# ---------------------------------------------------------------------------
with predict_tab:
    feats = st.session_state.get("features")
    model = st.session_state.get("model")
    if feats is None or model is None:
        st.warning("Build features and train a model first.")
    else:
        if st.button("🎯 Generate Risk Scores", use_container_width=True):
            try:
                pred = score_orders(model, feats)
                st.session_state["pred"] = pred
                hi = int((pred["delay_risk_score"] >= 0.6).sum())
                toast_ok(f"Scored {len(pred):,} orders • {hi} high-risk")
                st.balloons()
            except Exception as e:
                toast_err(str(e))

        pred = st.session_state.get("pred")
        if pred is not None:
            st.markdown("---")
            st.markdown("### 📊 Risk Dashboard")

            high = int((pred["delay_risk_score"] >= 0.6).sum())
            med = int(((pred["delay_risk_score"] >= 0.3) & (pred["delay_risk_score"] < 0.6)).sum())
            low = int((pred["delay_risk_score"] < 0.3).sum())
            avg = float(pred["delay_risk_score"].mean())

            a, b, c, d = st.columns(4)
            with a: st.markdown(f"""<div class='metric-card' style='text-align:center'><div style='font-size:2rem'>📦</div><div style='font-size:2.1rem;font-weight:800;color:var(--brandA)'>{len(pred):,}</div><div>Total scored</div></div>""", unsafe_allow_html=True)
            with b: st.markdown(f"""<div class='danger-box' style='text-align:center'><div style='font-size:2rem'>🔴</div><div style='font-size:2.1rem;font-weight:800;color:var(--danger)'>{high}</div><div>High ({(high/len(pred))*100:.1f}%)</div></div>""", unsafe_allow_html=True)
            with c: st.markdown(f"""<div class='warning-box' style='text-align:center'><div style='font-size:2rem'>🟡</div><div style='font-size:2.1rem;font-weight:800'>{med}</div><div>Medium ({(med/len(pred))*100:.1f}%)</div></div>""", unsafe_allow_html=True)
            with d: st.markdown(f"""<div class='success-box' style='text-align:center'><div style='font-size:2rem'>🟢</div><div style='font-size:2.1rem;font-weight:800;color:var(--success)'>{low}</div><div>Low ({(low/len(pred))*100:.1f}%)</div></div>""", unsafe_allow_html=True)

            st.markdown(f"""<div class='metric-card' style='text-align:center'><h4>Average Risk</h4><h1 style='margin:.25rem 0; color:var(--brandA)'>{avg:.1%}</h1></div>""", unsafe_allow_html=True)

            st.markdown("#### Distribution")
            st.plotly_chart(px.histogram(pred, x="delay_risk_score", nbins=30, title="Risk score distribution"), use_container_width=True)

            if "priority" in pred.columns:
                st.markdown("#### Risk by Priority")
                r = pred.groupby("priority")["delay_risk_score"].mean().reset_index()
                figp = px.bar(r, x="priority", y="delay_risk_score", title="Avg risk by priority")
                figp.update_yaxes(tickformat=".0%")
                st.plotly_chart(figp, use_container_width=True)

            # High-risk table + modal
            st.markdown("#### High-Risk Orders")
            high_df = pred[pred["delay_risk_score"] >= 0.6].copy()
            show_cols = [c for c in ["order_id", "delay_risk_score", "risk_category", "priority", "distance_km"] if c in pred.columns]
            st.dataframe(high_df[show_cols].head(60), use_container_width=True)

            if len(high_df) > 0:
                left, right = st.columns([2, 1])
                with left:
                    oid = st.selectbox("Inspect order_id", options=high_df["order_id"].astype(str).unique())
                with right:
                    if st.button("🔎 View Details"):
                        with st.modal(f"Order Details — {oid}"):
                            try:
                                row = pred[pred["order_id"].astype(str) == str(oid)].iloc[0]
                                st.json({
                                    "order_id": row.get("order_id"),
                                    "risk_score": float(row.get("delay_risk_score", np.nan)),
                                    "risk_category": row.get("risk_category"),
                                    "priority": row.get("priority"),
                                    "distance_km": float(row.get("distance_km", np.nan)) if "distance_km" in pred.columns else None,
                                })
                                raw = st.session_state["datasets"]["orders"]
                                more = raw[raw[raw.columns[0]].astype(str) == str(oid)] if "order_id" not in raw.columns else raw[raw["order_id"].astype(str) == str(oid)]
                                if len(more) > 0:
                                    st.subheader("Raw Order Snapshot")
                                    st.dataframe(_normalize_cols(more).head(1), use_container_width=True)
                            except Exception as e:
                                st.error(str(e))

            # Download predictions
            st.download_button(
                "📥 Download Predictions CSV",
                data=pred.to_csv(index=False).encode(),
                file_name="nexgen_predictions.csv",
                mime="text/csv",
            )


# ---------------------------------------------------------------------------
# ACTIONS
# ---------------------------------------------------------------------------
with actions_tab:
    pred = st.session_state.get("pred")
    if pred is None:
        st.warning("Generate predictions first.")
    else:
        thr = st.slider("Minimum risk threshold", 0.0, 1.0, 0.5, 0.05)
        if st.button("📝 Create Action Plan"):
            ap, summ = generate_prescriptions(pred, min_risk=thr)
            st.session_state["ap"] = ap
            st.session_state["ap_summ"] = summ
            toast_ok(f"Generated {len(ap)} actions")

        ap = st.session_state.get("ap")
        summ = st.session_state.get("ap_summ")
        if ap is not None and not ap.empty:
            a, b, c = st.columns(3)
            with a: st.markdown(f"""<div class='metric-card kpi'><h3>{summ.get('total_at_risk_orders',0)}</h3><div>at-risk orders</div></div>""", unsafe_allow_html=True)
            with b: st.markdown(f"""<div class='metric-card kpi'><h3>{summ.get('total_actions',0)}</h3><div>actions</div></div>""", unsafe_allow_html=True)
            with c: st.markdown(f"""<div class='metric-card kpi'><h3>{summ.get('high_priority_actions',0)}</h3><div>high priority</div></div>""", unsafe_allow_html=True)

            st.markdown("#### Actions by Type")
            st.plotly_chart(px.histogram(ap, x="action", color="action_priority"), use_container_width=True)
            st.dataframe(ap, use_container_width=True, height=420)

            st.download_button(
                "📥 Download Action Plan",
                data=ap.to_csv(index=False).encode(),
                file_name="nexgen_action_plan.csv",
                mime="text/csv",
            )


# ---------------------------------------------------------------------------
# IMPACT
# ---------------------------------------------------------------------------
with impact_tab:
    ap_summ = st.session_state.get("ap_summ")
    if not ap_summ:
        st.warning("Create an action plan first.")
    else:
        imp = calculate_business_impact(ap_summ)
        st.subheader("💰 Financial Impact (Monthly)")
        x, y, z = st.columns(3)
        with x: st.metric("Baseline Delay Cost", f"₹{imp['baseline_delay_cost']:,.0f}")
        with y: st.metric("Estimated Savings", f"₹{imp['estimated_savings']:,.0f}", delta=f"{(imp['estimated_savings']/imp['baseline_delay_cost']):.0%}" if imp['baseline_delay_cost'] else None)
        with z: st.metric("Total Business Value", f"₹{imp['total_business_value']:,.0f}")

        st.subheader("🎯 Operational Impact")
        x, y, z = st.columns(3)
        with x: st.metric("Delays Prevented", imp["prevented_delays"])
        with y: st.metric("Churn Prevented", f"{imp['prevented_churn_customers']} customers")
        with z: st.metric("ROI Multiplier", f"{imp['roi_multiplier']:.1f}x")

        # Executive summary
        st.markdown("---")
        st.subheader("📄 Executive Summary")
        lines = [
            "# Executive Summary",
            f"- Baseline delay cost: ₹{imp['baseline_delay_cost']:,.0f}",
            f"- Estimated savings: ₹{imp['estimated_savings']:,.0f}",
            f"- Total business value: ₹{imp['total_business_value']:,.0f}",
            f"- Delays prevented: {imp['prevented_delays']}",
            f"- Churn prevented: {imp['prevented_churn_customers']}",
            f"- ROI multiplier: {imp['roi_multiplier']:.1f}x",
        ]
        summary_text = "\n".join(lines)
        st.markdown(summary_text)
        st.download_button(
            "📥 Download Summary",
            data=summary_text.encode(),
            file_name="nexgen_executive_summary.md",
            mime="text/markdown",
        )
