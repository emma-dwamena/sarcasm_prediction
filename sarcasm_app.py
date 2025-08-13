
# -*- coding: utf-8 -*-

Sarcasm Detection App (ELMo + Logistic Regression / Random Forest)
Imbalance handling: **Downsampling** (majority class), with adjustable majority:minority ratio.
Pages: Upload → Preprocess → Train → Evaluate → Predict (single + batch).
UI: Professional dark theme.

Install (typical):
pip install streamlit scikit-learn matplotlib pandas numpy tensorflow==2.15.0 tensorflow-hub==0.15.0
# (ELMo is a TF1 module; TF 2.15 + tf.compat.v1 is used.)

Run:
streamlit run sarcasm_downsample_app.py

import os
import io
import re
import json
import base64
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# --- ML / Metrics ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix
)

# --- ELMo / TF Hub (TF1-style) ---
ELMO_URL = "https://tfhub.dev/google/elmo/3"
TF_OK = True
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    tf.get_logger().setLevel("ERROR")
    # We'll use TF1-compat mode because ELMo is a TF1 module.
    tf.compat.v1.disable_eager_execution()
except Exception as _e:
    TF_OK = False

# ----------------------------
# Streamlit Page Config & Theme
# ----------------------------
st.set_page_config(
    page_title="Sarcasm Detection (ELMo + LR/RF)",
    layout="wide",
    page_icon="📰"
)

# Dark professional theme via CSS
st.markdown(
    """
    <style>
    :root{
      --bg:#0b0f19;           /* app background */
      --panel:#121826;        /* panels/cards */
      --border:#1f2937;       /* subtle borders */
      --text:#e5e7eb;         /* primary text */
      --muted:#9ca3af;        /* secondary text */
      --accent:#60a5fa;       /* links / accents */
      --good:#10b981;         /* success */
      --warn:#f59e0b;         /* warning */
      --bad:#ef4444;          /* danger */
    }
    html, body, [data-testid="stAppViewContainer"]{ background:var(--bg); color:var(--text); }
    section[data-testid="stSidebar"]{ background:linear-gradient(180deg,#0b0f19 0%, #0f172a 100%); }
    section[data-testid="stSidebar"] *{ color:#e5e7eb !important; }
    a { color: var(--accent) !important; }
    .card{ background:var(--panel); border:1px solid var(--border); border-radius:16px; padding:16px; box-shadow:0 2px 8px rgba(0,0,0,.15); }
    .stButton>button{ background:var(--panel); color:var(--text); border:1px solid var(--border); border-radius:10px; padding:.6rem 1rem; box-shadow:0 1px 2px rgba(0,0,0,.25); }
    .stButton>button:hover{ border-color:#334155; }
    .stTextInput input, .stTextArea textarea, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
      background: var(--panel) !important; color: var(--text) !important; border:1px solid var(--border) !important;
    }
    div[data-testid="stDataFrame"] { background: var(--panel); border:1px solid var(--border); border-radius:12px; padding:8px; }
    div[data-testid="stMetricValue"]{ color:var(--text); } div[data-testid="stMetricLabel"]{ color:var(--muted); }
    div[data-testid="stTabs"] > div[role="tablist"]{ position:sticky; top:0; z-index:10; background:var(--panel); border-bottom:1px solid var(--border); }
    h1, h2, h3, h4, h5, h6 { color: var(--text); }
    .pill { display:inline-block; padding:.2rem .5rem; border-radius:9999px; border:1px solid var(--border); background:#0d1324; color:#cbd5e1; }
    .ok { color: var(--good); }
    .warn { color: var(--warn); }
    .bad { color: var(--bad); }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Helpers: Session State
# ----------------------------
def _init_state():
    ss = st.session_state
    ss.setdefault("df", None)
    ss.setdefault("text_col", None)
    ss.setdefault("label_col", None)
    ss.setdefault("clean_lower", True)
    ss.setdefault("clean_punct", True)
    ss.setdefault("dedupe", True)
    ss.setdefault("test_size", 0.2)
    ss.setdefault("random_state", 42)
    ss.setdefault("down_maj_mult", 1.0)   # majority:minority target after downsample (>=1.0)
    ss.setdefault("elmo", None)           # ELMo embedder (initialized once)
    ss.setdefault("X_train_emb", None)
    ss.setdefault("X_test_emb", None)
    ss.setdefault("y_train", None)
    ss.setdefault("y_test", None)
    ss.setdefault("scaler", None)
    ss.setdefault("models", {})           # {"lr": model, "rf": model}
    ss.setdefault("threshold", 0.5)
    ss.setdefault("train_done", False)
    ss.setdefault("prep_cache", None)

_init_state()

# ----------------------------
# Text Cleaning
# ----------------------------
_punct_pattern = re.compile(r"[^\w\s]")
def basic_clean(text, lower=True, remove_punct=True):
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if lower:
        t = t.lower()
    if remove_punct:
        t = _punct_pattern.sub(" ", t)
    t = re.sub(r"\s+", " ", t)
    return t

# ----------------------------
# ELMo Embedder (TF1 hub.Module)
# ----------------------------
class ELMoEmbedder:
    def __init__(self, url: str = ELMO_URL):
        if not TF_OK:
            raise RuntimeError("TensorFlow / TF-Hub not available. Please install tensorflow<=2.15 and tensorflow-hub.")
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.text_input = tf.compat.v1.placeholder(tf.string, shape=[None], name="text_input")
            self.module = hub.Module(url, trainable=False, name="elmo_module")
            # 'elmo' output (token-level, 1024); average across time to get sentence embedding.
            elmo_out = self.module(self.text_input, signature="default", as_dict=True)["elmo"]
            self.sentence_emb = tf.reduce_mean(elmo_out, axis=1, name="sentence_embedding")
            self.init_op = tf.group([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
        self.sess = tf.compat.v1.Session(graph=self.graph)
        self.sess.run(self.init_op)

    def embed(self, texts, batch_size=32):
        # Returns numpy array [n_samples, 1024]
        if isinstance(texts, (pd.Series, list, tuple)):
            texts = list(texts)
        elif isinstance(texts, np.ndarray):
            texts = texts.tolist()
        else:
            texts = [str(texts)]
        mats = []
        for i in range(0, len(texts), batch_size):
            batch = [str(t) if t is not None else "" for t in texts[i:i+batch_size]]
            vecs = self.sess.run(self.sentence_emb, feed_dict={self.text_input: batch})
            mats.append(vecs)
        return np.vstack(mats)

# ----------------------------
# Downsampling helper (without imblearn dependency)
# ----------------------------
def random_downsample(X, y, maj_mult=1.0, random_state=42):
    """Randomly downsample the majority class to achieve a majority:minority ratio of `maj_mult`.
    X: np.ndarray [n, d]; y: np.ndarray [n]; binary labels {0,1}.
    Returns: X_ds, y_ds
    """
    rng = np.random.RandomState(random_state)
    y = np.asarray(y).astype(int)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    n0, n1 = len(idx0), len(idx1)
    if n0 == 0 or n1 == 0:
        # Degenerate, no downsampling
        return X, y

    # Identify majority/minority
    if n0 >= n1:
        maj_label, maj_idx, min_label, min_idx = 0, idx0, 1, idx1
        nmaj, nmin = n0, n1
    else:
        maj_label, maj_idx, min_label, min_idx = 1, idx1, 0, idx0
        nmaj, nmin = n1, n0

    desired_maj = int(max(nmin, np.floor(nmin * maj_mult)))  # ensure >= nmin
    desired_maj = min(desired_maj, nmaj)  # don't overshoot

    if nmaj > desired_maj:
        keep_maj = rng.choice(maj_idx, size=desired_maj, replace=False)
    else:
        keep_maj = maj_idx

    keep_idx = np.concatenate([min_idx, keep_maj])
    rng.shuffle(keep_idx)
    return X[keep_idx], y[keep_idx]

# ----------------------------
# Sidebar Navigation
# ----------------------------
st.sidebar.title("📰 Sarcasm Detector (Downsampling)")
page = st.sidebar.radio(
    "Navigate",
    [
        "1) Data Upload",
        "2) Data Preprocessing",
        "3) Model Training",
        "4) Model Evaluation",
        "5) Prediction"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("ELMo → Logistic Regression / Random Forest • Precision / Recall / F1 / ROC-AUC")

# ----------------------------
# Page 1 — Data Upload
# ----------------------------
def page_upload():
    st.title("1) Data Upload")
    st.markdown("Upload the Kaggle **Sarcasm** dataset. Supports **CSV** and **JSON** (array or JSON lines).")
    f = st.file_uploader("Upload dataset", type=["csv", "json", "txt", "jsonl"])

    if f is not None:
        name = f.name.lower()
        try:
            if name.endswith(".csv"):
                df = pd.read_csv(f)
            elif name.endswith(".jsonl") or name.endswith(".txt"):
                df = pd.read_json(f, lines=True)
            elif name.endswith(".json"):
                content = f.read()
                try:
                    data = json.loads(content)
                    df = pd.DataFrame(data)
                except Exception:
                    df = pd.read_json(io.BytesIO(content), lines=True)
            else:
                st.error("Unsupported file type.")
                return
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            return

        st.session_state.df = df.copy()
        st.success(f"Loaded shape: {df.shape[0]} rows × {df.shape[1]} columns")

        with st.expander("Preview (first 10 rows)", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)

        # Column mapping
        cols = list(df.columns)
        st.subheader("Select Columns")
        default_text = "headline" if "headline" in cols else cols[0]
        default_label = "is_sarcastic" if "is_sarcastic" in cols else cols[-1]
        st.session_state.text_col = st.selectbox("Text column", cols, index=cols.index(default_text) if default_text in cols else 0, key="text_col_select")
        st.session_state.label_col = st.selectbox("Label column (0/1)", cols, index=cols.index(default_label) if default_label in cols else len(cols)-1, key="label_col_select")

        st.info("Tip: Kaggle 'News Headlines' uses **headline** for text and **is_sarcastic** (0/1) for labels.")

# ----------------------------
# Page 2 — Data Preprocessing (with Downsampling)
# ----------------------------
def page_preprocess():
    st.title("2) Data Preprocessing — with Downsampling")

    if st.session_state.df is None:
        st.warning("Please upload a dataset in **1) Data Upload**.")
        return

    df = st.session_state.df.copy()
    text_col = st.session_state.text_col
    label_col = st.session_state.label_col

    if text_col is None or label_col is None:
        st.warning("Select text and label columns in **1) Data Upload**.")
        return

    # Cleaning options
    st.subheader("Text Cleaning")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state.clean_lower = st.checkbox("lowercase", value=st.session_state.clean_lower)
    with c2:
        st.session_state.clean_punct = st.checkbox("remove punctuation", value=st.session_state.clean_punct)
    with c3:
        st.session_state.dedupe = st.checkbox("drop duplicate texts", value=st.session_state.dedupe)

    df["__text__"] = df[text_col].astype(str).apply(lambda t: basic_clean(t, st.session_state.clean_lower, st.session_state.clean_punct))
    df["__label__"] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)

    if st.session_state.dedupe:
        df = df.drop_duplicates(subset="__text__")

    st.markdown(f'<div class="pill">Rows after cleaning: {len(df):,}</div>', unsafe_allow_html=True)

    with st.expander("Class balance", expanded=True):
        vc = df["__label__"].value_counts().sort_index()
        n0 = int(vc.get(0, 0)); n1 = int(vc.get(1, 0)); N = n0 + n1 if (n0+n1)>0 else 1
        st.write(pd.DataFrame({
            "class": ["Not Sarcastic (0)", "Sarcastic (1)"],
            "count": [n0, n1],
            "percent": [round(100*n0/N, 2), round(100*n1/N, 2)]
        }))

    st.subheader("Train/Test Split")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.test_size = st.slider("Test size fraction", 0.1, 0.4, float(st.session_state.test_size), 0.05)
    with c2:
        st.session_state.random_state = st.number_input("Random state", 0, 10_000, int(st.session_state.random_state), step=1)

    # Downsampling config
    st.subheader("Imbalance Handling — Downsampling")
    st.caption("Reduce the majority class in the **training set** only. Choose target majority:minority ratio (≥ 1.0). Example: 1.0 → 50/50; 1.5 → majority is 1.5× minority.")
    st.session_state.down_maj_mult = st.slider("Target majority:minority ratio after downsampling", 1.0, 3.0, float(st.session_state.down_maj_mult), 0.1)

    # Split
    X = df["__text__"].values
    y = df["__label__"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=st.session_state.test_size, random_state=st.session_state.random_state, stratify=y
    )

    # ELMo init (once)
    st.subheader("ELMo Embeddings")
    if st.session_state.elmo is None:
        if not TF_OK:
            st.error("TensorFlow / tensorflow-hub not available. Please install:

`pip install tensorflow==2.15.0 tensorflow-hub==0.15.0`")
            return
        with st.spinner("Loading ELMo module from TF Hub… (first run may take a while)"):
            try:
                st.session_state.elmo = ELMoEmbedder(ELMO_URL)
            except Exception as e:
                st.error(f"Failed to load ELMo: {e}")
                return
        st.success("ELMo loaded.")

    # Embed train/test
    bsz = 32
    with st.spinner("Embedding training texts with ELMo…"):
        X_train_emb = st.session_state.elmo.embed(X_train, batch_size=bsz)
    with st.spinner("Embedding test texts with ELMo…"):
        X_test_emb = st.session_state.elmo.embed(X_test, batch_size=bsz)

    # Scale for LR; RF can take raw embeddings
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train_emb)
    X_test_std = scaler.transform(X_test_emb)

    # Downsample (train only), both views (std for LR, raw for RF)
    maj_mult = float(st.session_state.down_maj_mult)
    X_lr_train, y_lr_train = random_downsample(X_train_std, y_train, maj_mult=maj_mult, random_state=st.session_state.random_state)
    X_rf_train, y_rf_train = random_downsample(X_train_emb, y_train, maj_mult=maj_mult, random_state=st.session_state.random_state)

    # Report class counts after downsampling
    def _counts(yv):
        v = pd.Series(yv).value_counts().sort_index()
        return int(v.get(0,0)), int(v.get(1,0))

    n0_before = int((y_train==0).sum()); n1_before = int((y_train==1).sum())
    n0_lr_after, n1_lr_after = _counts(y_lr_train)
    n0_rf_after, n1_rf_after = _counts(y_rf_train)

    with st.expander("Train set class counts (before → after downsampling)", expanded=True):
        st.write(pd.DataFrame({
            "view": ["LR (scaled)", "RF (raw)"],
            "0_before": [n0_before, n0_before],
            "1_before": [n1_before, n1_before],
            "0_after": [n0_lr_after, n0_rf_after],
            "1_after": [n1_lr_after, n1_rf_after],
        }))

    # Save to state
    st.session_state.X_train_emb = X_train_emb
    st.session_state.X_test_emb = X_test_emb
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.session_state.scaler = scaler
    st.session_state.prep_cache = {
        "X_lr_train": X_lr_train, "y_lr_train": y_lr_train,
        "X_rf_train": X_rf_train, "y_rf_train": y_rf_train,
        "X_test_std": X_test_std
    }

    st.success("Preprocessing complete. Proceed to **3) Model Training**.")

# ----------------------------
# Page 3 — Model Training
# ----------------------------
def page_train():
    st.title("3) Model Training")
    required = ["X_train_emb", "X_test_emb", "y_train", "y_test", "scaler", "prep_cache"]
    if not all(k in st.session_state and st.session_state[k] is not None for k in required):
        st.warning("Please finish **2) Data Preprocessing** first.")
        return

    cache = st.session_state.prep_cache
    X_lr_train = cache["X_lr_train"]
    y_lr_train = cache["y_lr_train"]
    X_rf_train = cache["X_rf_train"]
    y_rf_train = cache["y_rf_train"]

    st.subheader("Hyperparameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        C = st.number_input("LogReg C (inverse regularization strength)", 0.01, 100.0, 1.0, step=0.05)
    with c2:
        n_estimators = st.number_input("RandomForest n_estimators", 50, 1000, 300, step=50)
    with c3:
        max_depth = st.number_input("RandomForest max_depth (0 = None)", 0, 100, 0, step=1)
        max_depth = None if max_depth == 0 else int(max_depth)

    st.subheader("Train Models")
    colA, colB = st.columns(2)

    with colA:
        with st.spinner("Training Logistic Regression…"):
            lr = LogisticRegression(
                C=C, solver="liblinear", random_state=st.session_state.random_state
            )
            lr.fit(X_lr_train, y_lr_train)

    with colB:
        with st.spinner("Training Random Forest…"):
            rf = RandomForestClassifier(
                n_estimators=int(n_estimators), max_depth=max_depth,
                random_state=st.session_state.random_state, n_jobs=-1
            )
            rf.fit(X_rf_train, y_rf_train)

    st.session_state.models = {"lr": lr, "rf": rf}
    st.session_state.train_done = True
    st.success("Training complete. Proceed to **4) Model Evaluation**.")

# ----------------------------
# Metrics utilities
# ----------------------------
def metric_table(metrics_dict):
    rows = []
    for model_name, m in metrics_dict.items():
        rows.append([model_name, m["Precision"], m["Recall"], m["F1"], m["ROC-AUC"]])
    return pd.DataFrame(rows, columns=["Model", "Precision", "Recall", "F1", "ROC-AUC"])

# ----------------------------
# Page 4 — Model Evaluation
# ----------------------------
def page_evaluation():
    st.title("4) Model Evaluation")
    req = ["models", "X_test_emb", "y_test", "scaler", "prep_cache"]
    if not all(k in st.session_state and st.session_state[k] is not None for k in req):
        st.warning("Train models in **3) Model Training** first.")
        return

    models = st.session_state.models
    scaler = st.session_state.scaler
    X_test_emb = st.session_state.X_test_emb
    y_test = st.session_state.y_test
    X_test_std = st.session_state.prep_cache["X_test_std"]

    lr = models["lr"]
    rf = models["rf"]

    # Predict probabilities
    lr_proba = lr.predict_proba(X_test_std)[:, 1]
    rf_proba = rf.predict_proba(X_test_emb)[:, 1]

    # Threshold control
    st.session_state.threshold = st.slider("Decision threshold (affects Precision/Recall/F1)", 0.1, 0.9, float(st.session_state.threshold), 0.05)
    thresh = st.session_state.threshold
    lr_pred = (lr_proba >= thresh).astype(int)
    rf_pred = (rf_proba >= thresh).astype(int)

    # Metrics
    metrics = {
        "Logistic Regression": {
            "Precision": precision_score(y_test, lr_pred, zero_division=0),
            "Recall": recall_score(y_test, lr_pred, zero_division=0),
            "F1": f1_score(y_test, lr_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, lr_proba),
        },
        "Random Forest": {
            "Precision": precision_score(y_test, rf_pred, zero_division=0),
            "Recall": recall_score(y_test, rf_pred, zero_division=0),
            "F1": f1_score(y_test, rf_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, rf_proba),
        }
    }

    tab_perf, tab_cm, tab_roc = st.tabs(["Performance", "Confusion Matrices", "ROC Curves"])

    with tab_perf:
        st.subheader("Performance Comparison")
        dfm = metric_table(metrics).round(4)
        st.dataframe(dfm, use_container_width=True)
        better = "Logistic Regression" if dfm.set_index("Model").loc["Logistic Regression", "F1"] >= dfm.set_index("Model").loc["Random Forest", "F1"] else "Random Forest"
        st.markdown(f"**Better F1 (at threshold={thresh:.2f}):** `{better}`")

    with tab_cm:
        c1, c2 = st.columns(2)
        with c1:
            st.write("Confusion Matrix — Logistic Regression")
            cm_lr = confusion_matrix(y_test, lr_pred)
            st.write(pd.DataFrame(cm_lr, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))
        with c2:
            st.write("Confusion Matrix — Random Forest")
            cm_rf = confusion_matrix(y_test, rf_pred)
            st.write(pd.DataFrame(cm_rf, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

    with tab_roc:
        import matplotlib.pyplot as plt
        fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
        fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)

        fig = plt.figure(figsize=(6, 5))
        plt.plot(fpr_lr, tpr_lr, label=f"LogReg (AUC={metrics['Logistic Regression']['ROC-AUC']:.3f})")
        plt.plot(fpr_rf, tpr_rf, label=f"RandForest (AUC={metrics['Random Forest']['ROC-AUC']:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")
        plt.legend(loc="lower right")
        st.pyplot(fig)

# ----------------------------
# Page 5 — Prediction
# ----------------------------
def page_prediction():
    st.title("5) Prediction")

    req = ["models", "scaler", "elmo"]
    if not all(k in st.session_state and st.session_state[k] is not None for k in req):
        st.warning("Please complete **Training** before predicting.")
        return

    models = st.session_state.models
    scaler = st.session_state.scaler
    elmo = st.session_state.elmo
    threshold = st.session_state.get("threshold", 0.5)

    tab_single, tab_batch = st.tabs(["Single Text", "Batch Upload"])

    with tab_single:
        text = st.text_area("Enter headline / text", height=120, placeholder="e.g., 'Local man wins lottery, quits job to pursue full-time napping career'")
        if st.button("Predict"):
            if not text.strip():
                st.warning("Enter some text.")
            else:
                emb = elmo.embed([text])
                x_std = scaler.transform(emb)
                lr_proba = models["lr"].predict_proba(x_std)[:, 1][0]
                rf_proba = models["rf"].predict_proba(emb)[:, 1][0]
                lr_pred = int(lr_proba >= threshold)
                rf_pred = int(rf_proba >= threshold)

                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Logistic Regression", f"{'Sarcastic' if lr_pred else 'Not Sarcastic'}", delta=f"P={lr_proba:.3f}")
                with c2:
                    st.metric("Random Forest", f"{'Sarcastic' if rf_pred else 'Not Sarcastic'}", delta=f"P={rf_proba:.3f}")

    with tab_batch:
        st.write("Upload a CSV for batch predictions.")
        bf = st.file_uploader("Upload CSV", type=["csv"], key="batch_csv")
        text_col_name = st.text_input("Text column name in CSV", value=st.session_state.text_col or "headline")

        if bf is not None:
            try:
                bdf = pd.read_csv(bf)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                return

            if text_col_name not in bdf.columns:
                st.error(f"Column '{text_col_name}' not in CSV.")
                return

            with st.spinner("Embedding and predicting…"):
                texts = bdf[text_col_name].astype(str).tolist()
                emb = elmo.embed(texts, batch_size=32)
                x_std = scaler.transform(emb)
                lr_proba = models["lr"].predict_proba(x_std)[:, 1]
                rf_proba = models["rf"].predict_proba(emb)[:, 1]
                lr_pred = (lr_proba >= threshold).astype(int)
                rf_pred = (rf_proba >= threshold).astype(int)

                out = bdf.copy()
                out["proba_lr"] = lr_proba
                out["pred_lr"] = lr_pred
                out["proba_rf"] = rf_proba
                out["pred_rf"] = rf_pred

            # Save to file
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = f"sarcasm_predictions_{ts}.csv"
            out.to_csv(out_path, index=False)

            st.success(f"Done. Saved to {out_path}")
            st.download_button("Download predictions CSV", data=out.to_csv(index=False).encode(), file_name=out_path, mime="text/csv")

# ----------------------------
# Router
# ----------------------------
st.sidebar.markdown("---")
st.sidebar.write("**Pages**")
page = st.sidebar.radio(
    "Go to",
    ["1) Data Upload", "2) Data Preprocessing", "3) Model Training", "4) Model Evaluation", "5) Prediction"],
    index=["1) Data Upload", "2) Data Preprocessing", "3) Model Training", "4) Model Evaluation", "5) Prediction"].index(page)
)

if page.startswith("1"):
    page_upload()
elif page.startswith("2"):
    page_preprocess()
elif page.startswith("3"):
    page_train()
elif page.startswith("4"):
    page_evaluation()
elif page.startswith("5"):
    page_prediction()
