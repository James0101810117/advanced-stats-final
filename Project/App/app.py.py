# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 17:20:43 2025

@author: james
"""

# -*- coding: utf-8 -*-
"""
Combined Streamlit dashboard + ML models for Late_delivery_risk

- 上半部：Delivery Delay Dashboard — Market 延遲率互動分析
- 下半部：Late_delivery_risk 二元分類模型 + Random Forest 特徵重要度

@author: james
"""

import pandas as pd
import numpy as np
import streamlit as st

# ==========================
# Optional libraries
# ==========================
# Plotly for interactive charts
try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# XGBoost for one of the models
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ==========================
# ML related imports
# ==========================
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import matplotlib.pyplot as plt


# =====================================================
# 1. Preprocessor
# =====================================================
def build_preprocessor(X):
    """
    Build preprocessing pipeline:
    - numeric: impute mean + standardize
    - categorical: impute most frequent + one-hot encode
    """
    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    return preprocessor


# =====================================================
# 2. Evaluation Function
# =====================================================
def evaluate_model(model, X_train, X_test, y_train, y_test,
                   model_name, split_name, target_name="Late_delivery_risk"):
    """
    Fit model, compute metrics, and return results as dict.
    """
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    return {
        "Target": target_name,
        "Model": model_name,
        "Split": split_name,
        "Accuracy": acc,
        "F1": f1,
        "AUC": auc,
    }


# =====================================================
# 3. Train models (80/20 only)
# =====================================================
def run_Late_delivery_risk_models(df):
    """
    Run Logistic Regression, XGBoost (if available), Random Forest
    on Late_delivery_risk with an 80/20 stratified split.
    """
    # Target
    y = df["Late_delivery_risk"]

    # Potential leakage columns (根據你原本的設計)
    leak_cols = [
        "Late_delivery_risk",
        "Delivery Status",
        "Days for shipping (real)",
        "Scheduled delivery date",
        # ⚠️ 請注意：這裡欄位名稱要跟你實際 df.columns 一致
        # 例如：'Shipping date (DateOrders)' 或 'shipping date (DateOrders)'
        "Shipping date (DateOrders)",
    ]

    # ID-like columns (對預測沒什麼 generalizable 的結構)
    id_like_cols = [
        "Order Id",
        "Order Item Id",
        "Order Item Cardprod Id",
        "Order Customer Id",
        "Customer Id",
        "Customer Email",
        "Customer Password",
        "Product Card Id",
        "Product Category Id",
        "Product Image",
        "Order Zipcode",
        "Customer Zipcode",
        "Type",
        "Order Item Quantity",
    ]

    # 去掉重複欄位名稱
    drop_cols = list(set(leak_cols + id_like_cols))

    # Drop 真的有在 df 裡的那些欄位
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    preprocessor = build_preprocessor(X)

    results = []

    # Only 80/20 split
    test_size = 0.2
    split_name = "80/20"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # 1️⃣ Logistic Regression (L2)
    log_clf = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, penalty="l2")),
        ]
    )
    results.append(
        evaluate_model(
            log_clf, X_train, X_test, y_train, y_test,
            "Logistic Regression (L2)", split_name
        )
    )

    # 2️⃣ XGBoost (if available)
    if HAS_XGB:
        xgb_clf = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("clf", XGBClassifier(
                    n_estimators=250,
                    learning_rate=0.1,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                )),
            ]
        )
        results.append(
            evaluate_model(
                xgb_clf, X_train, X_test, y_train, y_test,
                "XGBoost", split_name
            )
        )

    # 3️⃣ Random Forest
    rf_clf = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1
            )),
        ]
    )
    results.append(
        evaluate_model(
            rf_clf, X_train, X_test, y_train, y_test,
            "Random Forest", split_name
        )
    )

    results_df = pd.DataFrame(results)
    return results_df, X, y


# =====================================================
# 4. Feature Importance (Random Forest 80/20)
# =====================================================
def Late_delivery_risk_feature_importance(X, y):
    """
    Fit Random Forest (80/20) and output top 5 most important features
    + matplotlib figure (給 Streamlit 顯示).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor(X)

    best_rf = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1
            )),
        ]
    )

    best_rf.fit(X_train, y_train)

    # Get transformed feature names
    prep = best_rf.named_steps["prep"]

    num_cols = X.select_dtypes(include=[np.number]).columns
    num_features = prep.named_transformers_["num"].get_feature_names_out(num_cols)

    cat_cols = X.select_dtypes(include=["object"]).columns
    onehot = prep.named_transformers_["cat"].named_steps["onehot"]
    cat_features = onehot.get_feature_names_out(cat_cols)

    feature_names = np.concatenate([num_features, cat_features])

    # Get importances
    importances = best_rf.named_steps["clf"].feature_importances_

    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    top5 = fi.head(5)

    # Plot top 5 feature importances
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(top5["feature"], top5["importance"])
    ax.invert_yaxis()
    ax.set_xlabel("Feature importance")
    ax.set_title("Top 5 Features for Late_delivery_risk – Random Forest (80/20)")
    fig.tight_layout()

    return top5, fig


# =====================================================
# 5. Streamlit App
# =====================================================

# --------------------
# Basic page config
# --------------------
st.set_page_config(
    page_title="Delivery Delay Dashboard + ML Models",
    layout="wide"
)

st.title("📦 Delivery Delay Dashboard — Market Delay & ML Explorer")

st.markdown(
    """
    This app uses the **DataCoSupplyChain** dataset to:
    - Explore shipment delays across **Markets**
    - Train models to predict **`Late_delivery_risk`**  
      (Logistic Regression, Random Forest, and XGBoost if available)

    **Delay definition**: `Late_delivery_risk` (1 = delayed, 0 = on time)
    """
)

# --------------------
# File path input (blank by default)
# --------------------
data_path = st.sidebar.text_input(
    "📂 CSV file path",
    value="",
    placeholder=r'Paste full file path here, e.g. C:\Users\james\...\DataCoSupplyChainDataset_sample_20000.csv',
    help="Copy the full path of your CSV file from File Explorer and paste it here. Do NOT include quotes."
)

# If no path is provided, stop and show guidance
if not data_path.strip():
    st.info("👈 Please paste the full CSV file path in the box on the left to load the data.")
    st.stop()

# Clean up accidental quotes from Copy as path (e.g. \"C:\\...\\file.csv\")
clean_path = data_path.strip().strip('"').strip("'")

# --------------------
# Load data
# --------------------
try:
    df_all = pd.read_csv(clean_path, encoding="latin-1")
except FileNotFoundError:
    st.error(
        f"❌ Could not find the file at:\n\n`{clean_path}`\n\n"
        "Please check the path (no quotes, correct folder and file name) and try again."
    )
    st.stop()
except Exception as e:
    st.error(f"❌ An error occurred while reading the file:\n\n`{e}`")
    st.stop()
# --------------------
# 修正 _lead_bucket 類別字串
# --------------------
if "_lead_bucket" in df_all.columns:
    # 把壞掉的中文 bucket 換成好懂的英文
    mapping = {
        "<=0": "<=0 days",
        "1¤ë2¤é": "1-2 days",
        "3¤ë5¤é": "3-5 days",
        "6¤ë7¤é": "6-7 days",
    }

    df_all["lead_bucket_clean"] = df_all["_lead_bucket"].map(mapping)

    # 設成有順序的 ordinal 類別（之後 OHE 的欄名會比較好看）
    order = ["<=0 days", "1-2 days", "3-5 days", "6-7 days"]
    df_all["lead_bucket_clean"] = pd.Categorical(
        df_all["lead_bucket_clean"],
        categories=order,
        ordered=True
    )

    # 不要再用原本那個爛編碼欄位了
    df_all.drop(columns=["_lead_bucket"], inplace=True)

# --------------------
# Optional: clean Chinese lead time buckets -> English
# --------------------
if "_lead_bucket" in df_all.columns:
    mapping = {
        "<=0": "<=0 days",
        "1月2日": "1-2 days",
        "3月5日": "3-5 days",
        "6月7日": "6-7 days",
    }

    df_all["lead_bucket_clean"] = df_all["_lead_bucket"].map(mapping)

    order = ["<=0 days", "1-2 days", "3-5 days", "6-7 days"]
    df_all["lead_bucket_clean"] = pd.Categorical(
        df_all["lead_bucket_clean"],
        categories=order,
        ordered=True,
    )

    # 如果你想在畫面上確認，可用 st.write，而不是 print
    # st.write(df_all["lead_bucket_clean"].value_counts().sort_index())
    # st.write(df_all["lead_bucket_clean"].value_counts(normalize=True).sort_index())


# Verify required columns
required_cols = ["Market", "Late_delivery_risk"]
missing = [c for c in required_cols if c not in df_all.columns]

if missing:
    st.error(
        "The dataset is missing required columns.\n\n"
        f"Required columns: {required_cols}\n"
        f"Missing columns: {missing}\n\n"
        f"Available columns: {list(df_all.columns)}"
    )
    st.stop()

# Make sure Late_delivery_risk is numeric
df_all["Late_delivery_risk"] = pd.to_numeric(df_all["Late_delivery_risk"], errors="coerce")

# df for Market dashboard (需要 Market + Late_delivery_risk)
df_valid = df_all.dropna(subset=["Market", "Late_delivery_risk"]).copy()

# df for ML models (只需要 Late_delivery_risk)
df_model = df_all.dropna(subset=["Late_delivery_risk"]).copy()

if df_valid.empty or df_model.empty:
    st.error("After cleaning, there are no valid rows left. Please check `Market` and `Late_delivery_risk` values.")
    st.stop()

# =====================================================
# 5.1 Market-level aggregation (Dashboard part)
# =====================================================
market_summary = (
    df_valid
    .groupby("Market", as_index=False)
    .agg(
        delay_rate=("Late_delivery_risk", "mean"),   # average delay rate
        orders=("Late_delivery_risk", "size")        # number of orders
    )
)

market_summary["delay_rate_pct"] = market_summary["delay_rate"] * 100

# --------------------
# Sidebar filters
# --------------------
st.sidebar.header("🔎 Filter settings")

all_markets = sorted(market_summary["Market"].unique().tolist())
selected_markets = st.sidebar.multiselect(
    "Select markets to display",
    options=all_markets,
    default=all_markets
)

min_orders = st.sidebar.slider(
    "Minimum number of orders (filter out very small markets)",
    min_value=0,
    max_value=int(market_summary["orders"].max()),
    value=0,
    step=100
)

filtered = market_summary[
    market_summary["Market"].isin(selected_markets)
    & (market_summary["orders"] >= min_orders)
].copy()

if filtered.empty:
    st.warning("No markets match the current filters. Please widen your selection.")
    st.stop()

filtered = filtered.sort_values("delay_rate", ascending=True)

# --------------------
# KPI section
# --------------------
total_orders = int(df_model["Late_delivery_risk"].count())
overall_delay_rate = df_model["Late_delivery_risk"].mean() * 100
selected_orders = int(
    df_valid[df_valid["Market"].isin(filtered["Market"])]["Late_delivery_risk"].count()
)

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("📦 Total valid orders", f"{total_orders:,}")

with c2:
    st.metric("⏱ Overall average delay rate", f"{overall_delay_rate:.1f} %")

with c3:
    st.metric("🌍 Orders in selected markets", f"{selected_orders:,}")

st.markdown("---")

# --------------------
# Chart: delay rate by market
# --------------------
st.subheader("📊 Delay rate by market")

if HAS_PLOTLY:
    fig_bar = px.bar(
        filtered,
        x="Market",
        y="delay_rate_pct",
        hover_data=["orders"],
        text="delay_rate_pct"
    )
    fig_bar.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )
    fig_bar.update_layout(
        xaxis_title="Market",
        yaxis_title="Delay rate (%)",
        yaxis_tickformat=".1f",
        margin=dict(t=40, b=40)
    )
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.info("⚠️ Plotly is not installed. Using Streamlit's built-in bar_chart instead. "
            "For interactive charts, install Plotly with `pip install plotly`.")
    chart_data = filtered.set_index("Market")["delay_rate_pct"]
    st.bar_chart(chart_data)

# --------------------
# Detail table
# --------------------
st.subheader("📃 Detail table")

detail_df = (
    filtered[["Market", "orders", "delay_rate_pct"]]
    .rename(columns={
        "Market": "Market",
        "orders": "Orders",
        "delay_rate_pct": "Delay rate (%)"
    })
)

st.dataframe(detail_df, use_container_width=True)

st.caption(
    "📌 Delay rate = mean of `Late_delivery_risk` (1 = delayed, 0 = on time). "
    "When interpreting results, always consider the number of orders to avoid overreacting to tiny markets."
)

# =====================================================
# 5.2 ML Models section
# =====================================================
st.markdown("---")
st.header("🤖 Late_delivery_risk — Predictive models (80/20 split)")

st.markdown(
    """
    The models below are trained on all rows with a valid `Late_delivery_risk` value.  
    Obvious leakage variables (actual shipping dates / realized days for shipping, etc.)  
    and ID-like fields are removed before training.
    """
)

if not HAS_XGB:
    st.info("ℹ️ `xgboost` is not installed, so the XGBoost model will be skipped. "
            "Install it with `pip install xgboost` if you want to include it.")

if st.button("🚀 Train models"):
    with st.spinner("Training models..."):
        results_df, X_L, y_L = run_Late_delivery_risk_models(df_model)

    st.subheader("📈 Model performance on test set (80/20)")
    st.dataframe(
        results_df.style.format({
            "Accuracy": "{:.3f}",
            "F1": "{:.3f}",
            "AUC": "{:.3f}",
        }),
        use_container_width=True
    )

    st.subheader("🌟 Random Forest — Top 5 important features")
    top5_df, fig_imp = Late_delivery_risk_feature_importance(X_L, y_L)
    st.dataframe(top5_df, use_container_width=True)
    st.pyplot(fig_imp)
