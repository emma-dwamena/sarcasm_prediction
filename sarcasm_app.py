# -*- coding: utf-8 -*-
"""
Sarcasm Detection App — ELMo Embeddings + Tunable Downsampling + Logistic Regression / Random Forest
---------------------------------------------------------------------------------------------------
This Streamlit application demonstrates an end-to-end text classification pipeline on the
Kaggle "News Headlines with Sarcasm" dataset, with a **tunable downsampling ratio** for
class imbalance handling.

Pages:
1) Data Upload        — Load CSV/JSON/JSONL and map text/label columns.
2) Data Preprocessing — Clean text, split the data, **downsample majority class** to a user-chosen
                        majority:minority ratio (≥ 1.0), compute ELMo embeddings, and standardize features.
3) Model Training     — Train Logistic Regression (std. embeddings) and Random Forest (raw embeddings).
4) Model Evaluation   — Compare Precision, Recall, F1, and ROC-AUC; show confusion matrices & ROC curves.
5) Prediction         — Predict on a single text or a batch CSV and download results.

Notes on ELMo / TensorFlow Hub:
- ELMo v3 is a TF1-style `hub.Module`. Use TF 2.15 in TF1-compat mode + tensorflow-hub==0.12.0.
- If these versions are not present, the app explains what to install.

Run:
    pip install streamlit scikit-learn matplotlib pandas numpy tensorflow==2.15.0 tensorflow-hub==0.12.0
    streamlit run sarcasm_elmo_downsample_ratio_commented_app.py
"""

# ==============================
# Imports
# ==============================
import os, io, re, json, base64
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# Scikit-learn building blocks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
)

# ------------------------------
# ELMo via TensorFlow Hub (TF1)
# ------------------------------
ELMO_URL = "https://tfhub.dev/google/elmo/3"
TF_OK = True
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    # ELMo v3 is a TF1 module, so we use TF1-style graph execution.
    tf.get_logger().setLevel('ERROR')
    tf.compat.v1.disable_eager_execution()
    _HAS_MODULE = hasattr(hub, "Module")  # required for ELMo v3
    if not _HAS_MODULE:
        TF_OK = False
except Exception:
    TF_OK = False

# ==============================
# Streamlit: Page Config & Theme
# ==============================
st.set_page_config(page_title="Sarcasm Detection (ELMo + LR/RF)", page_icon="📰", layout="wide")

# -- Lightweight, professional dark theme via CSS.
st.markdown(
    """
    <style>
    :root{
      --bg:#0b0f19; --panel:#121826; --border:#1f2937; --text:#e5e7eb; --muted:#9ca3af; --accent:#60a5fa;
      --good:#10b981; --warn:#f59e0b; --bad:#ef4444;
    }
    html, body, [data-testid="stAppViewContainer"]{ background:var(--bg); color:var(--text); }
    section[data-testid="stSidebar"]{ background:linear-gradient(180deg,#0b0f19 0%, #0f172a 100%); }
    section[data-testid="stSidebar"] *{ color:#e5e7eb !important; }
    a { color: var(--accent) !important; }
    .card{ background:var(--panel); border:1px solid var(--border); border-radius:16px; padding:16px; box-shadow:0 2px 8px rgba(0,0,0,.15); }
    .stButton>button{ background:var(--panel); color:var(--text); border:1px solid var(--border); border-radius:10px; padding:.6rem 1rem; box-shadow:0 1px 2px rgba(0,0,0,.25); }
    .stButton>button:hover{ border-color:#334155; }
    .stTextInput input, .stTextArea textarea, .stNumberInput input, .stSelectbox div[data-baseweb="select"]{
      background: var(--panel) !important; color: var(--text) !important; border:1px solid var(--border) !important;
    }
    div[data-testid="stDataFrame"]{ background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:8px; }
    div[data-testid="stMetricValue"]{ color:var(--text); } div[data-testid="stMetricLabel"]{ color:var(--muted); }
    div[data-testid="stTabs"] > div[role="tablist"]{ position:sticky; top:0; z-index:10; background:var(--panel); border-bottom:1px solid var(--border); }
    .pill { display:inline-block; padding:.2rem .5rem; border-radius:9999px; border:1px solid var(--border); background:#0d1324; color:#cbd5e1; }
    </style>
    """,
    unsafe_allow_html=True
)

# ==============================
# Session-State Initialization
# ==============================
def _init_state():
    """Define all session-state variables used across pages (safe default values)."""
    ss = st.session_state
    ss.setdefault("df", None)                 # raw dataframe
    ss.setdefault("text_col", None)           # selected text column name
    ss.setdefault("label_col", None)          # selected label column name
    ss.setdefault("clean_lower", True)        # lowercase text
    ss.setdefault("clean_punct", True)        # remove punctuation
    ss.setdefault("dedupe", True)             # drop duplicate texts
    ss.setdefault("test_size", 0.2)           # fraction for test split
    ss.setdefault("random_state", 42)         # RNG seed for reproducibility
    ss.setdefault("down_maj_mult", 1.0)       # target majority:minority ratio (>=1.0)
    ss.setdefault("elmo", None)               # ELMo embedder instance
    ss.setdefault("X_train_emb", None)        # train embeddings (raw ELMo)
    ss.setdefault("X_test_emb", None)         # test embeddings (raw ELMo)
    ss.setdefault("y_train", None)            # train labels
    ss.setdefault("y_test", None)             # test labels
    ss.setdefault("scaler", None)             # StandardScaler (for LR input)
    ss.setdefault("models", {})               # trained models {"lr":..., "rf":...}
    ss.setdefault("threshold", 0.5)           # classification threshold in evaluation
    ss.setdefault("prep_cache", None)         # extra cached arrays (std features etc.)

_init_state()

# ==============================
# Basic Text Cleaning
# ==============================
_punct_pattern = re.compile(r"[^\w\s]")

def basic_clean(text, lower=True, remove_punct=True):
    """
    Simple preprocessing for short headlines:
    - strip whitespace
    - lowercase (optional)
    - remove punctuation (optional)
    - squeeze multiple spaces into one
    """
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if lower: t = t.lower()
    if remove_punct: t = _punct_pattern.sub(" ", t)
    t = re.sub(r"\s+", " ", t)
    return t

# ==============================
# ELMo Embedder (TF1 graph mode)
# ==============================
class ELMoEmbedder:
    """
    Thin wrapper around TF Hub ELMo v3 (TF1-style). Produces a 1024-dim embedding per text
    by averaging token-level 'elmo' outputs.
    """
    def __init__(self, url: str = ELMO_URL):
        if not TF_OK:
            st.error("""TensorFlow / tensorflow-hub not available or incompatible.
Install exact versions:

pip install tensorflow==2.15.0 tensorflow-hub==0.12.0

(ELMo v3 requires TF1 hub.Module.)""")
            raise RuntimeError("TF/Hub unavailable")
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Placeholder for a batch of strings
            self.text_input = tf.compat.v1.placeholder(tf.string, shape=[None], name="text_input")
            # Load TF Hub module (frozen graph)
            self.module = hub.Module(url, trainable=False, name="elmo_module")
            # 'elmo' => (batch, timesteps, 1024); we average over time to get sentence embeddings.
            elmo_out = self.module(self.text_input, signature="default", as_dict=True)["elmo"]
            self.sentence_emb = tf.reduce_mean(elmo_out, axis=1, name="sentence_embedding")
            # Initialize variables and tables
            self.init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer())
        # Create session and run initializers
        self.sess = tf.compat.v1.Session(graph=self.graph)
        self.sess.run(self.init_op)

    def embed(self, texts, batch_size=32):
        """Compute ELMo embeddings for a list/Series/array of strings. Returns np.ndarray [N, 1024]."""
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

# ==============================
# Tunable Downsampling (majority:minority >= 1.0)
# ==============================
def downsample_ratio(X, y, maj_mult=1.0, random_state=42):
    """
    Downsample the majority class to achieve a **majority:minority** ratio close to `maj_mult`.
    - Only removes samples from the majority class (never from the minority).
    - Example: maj_mult=1.0 → 1:1 balance (strict); 1.5 → majority ≈ 1.5× minority.
    """
    rng = np.random.RandomState(random_state)
    y = np.asarray(y).astype(int)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    n0, n1 = len(idx0), len(idx1)
    if n0 == 0 or n1 == 0:
        # Can't balance if a class is missing
        return X, y

    # Identify majority vs minority
    if n0 >= n1:
        maj_idx, min_idx = idx0, idx1
        nmaj, nmin = n0, n1
    else:
        maj_idx, min_idx = idx1, idx0
        nmaj, nmin = n1, n0

    # Desired majority size (cannot exceed current majority size and cannot be < minority size)
    target_maj = int(max(nmin, np.floor(nmin * float(maj_mult))))
    target_maj = min(target_maj, nmaj)

    # Randomly keep `target_maj` samples from the majority and keep all minority samples
    if nmaj > target_maj:
        keep_maj = rng.choice(maj_idx, size=target_maj, replace=False)
    else:
        keep_maj = maj_idx

    keep_idx = np.concatenate([min_idx, keep_maj])
    rng.shuffle(keep_idx)
    return X[keep_idx], y[keep_idx]

# ==============================
# Downsampling Distribution Plot (Streamlit)
# ==============================
def st_plot_dist(y_before, y_after, title):
    """Bar chart: class counts before vs after downsampling (Streamlit)."""
    import numpy as np
    import matplotlib.pyplot as plt
    y_before = np.asarray(y_before).astype(int)
    y_after  = np.asarray(y_after).astype(int)
    # count 0/1 for both arrays
    def _counts(y):
        c = np.bincount(y, minlength=2)[:2]
        return int(c[0]), int(c[1])
    c0b, c1b = _counts(y_before)
    c0a, c1a = _counts(y_after)

    labels = ["Class 0 (Not Sarcastic)", "Class 1 (Sarcastic)"]
    x = np.arange(len(labels))
    width = 0.35

    fig = plt.figure(figsize=(7, 5))
    plt.bar(x - width/2, [c0b, c1b], width, label="Before")
    plt.bar(x + width/2, [c0a, c1a], width, label="After")
    # annotate bars with counts
    for i, v in enumerate([c0b, c1b]):
        plt.text(x[i] - width/2, v, str(v), ha='center', va='bottom')
    for i, v in enumerate([c0a, c1a]):
        plt.text(x[i] + width/2, v, str(v), ha='center', va='bottom')

    plt.xticks(x, labels)
    plt.ylabel("Count")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    st.pyplot(fig)


# ==============================
# Sidebar Navigation
# ==============================
st.sidebar.title("📰 Sarcasm Detector (ELMo + Downsampling)")
page = st.sidebar.radio("Navigate", [
    "1) Data Upload",
    "2) Data Preprocessing",
    "3) Model Training",
    "4) Model Evaluation",
    "5) Prediction",
])

st.sidebar.markdown("---")
st.sidebar.caption("ELMo → Logistic Regression / Random Forest • Precision / Recall / F1 / ROC-AUC")

# ==============================
# Page 1 — Data Upload
# ==============================
def page_upload():
    """Read a dataset file (CSV / JSON / JSONL) and let the user map text/label columns."""
    st.title("1) Data Upload")
    st.markdown("Upload the Kaggle **Sarcasm** dataset (CSV/JSON/JSONL).")

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
                # Try JSON array; fallback to JSON lines
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

        # Column mapping UI
        cols = list(df.columns)
        st.subheader("Select Columns")
        default_text = "headline" if "headline" in cols else cols[0]
        default_label = "is_sarcastic" if "is_sarcastic" in cols else cols[-1]
        st.session_state.text_col = st.selectbox("Text column", cols, index=cols.index(default_text) if default_text in cols else 0)
        st.session_state.label_col = st.selectbox("Label column (0/1)", cols, index=cols.index(default_label) if default_label in cols else len(cols)-1)

        st.info("Tip: Common columns are **headline** (text) and **is_sarcastic** (label).")

# ==============================
# Page 2 — Data Preprocessing
# ==============================
def page_preprocess():
    """
    Clean text, split data, create ELMo embeddings, standardize inputs, and
    perform **downsampling** on the training set according to a chosen ratio.
    """
    st.title("2) Data Preprocessing — Tunable Downsampling")

    if st.session_state.df is None:
        st.warning("Please upload a dataset in **1) Data Upload**.")
        return

    df = st.session_state.df.copy()
    text_col = st.session_state.text_col
    label_col = st.session_state.label_col
    if text_col is None or label_col is None:
        st.warning("Select text and label columns in **1) Data Upload**.")
        return

    # -- Cleaning controls
    st.subheader("Text Cleaning")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state.clean_lower = st.checkbox("lowercase", value=st.session_state.clean_lower)
    with c2:
        st.session_state.clean_punct = st.checkbox("remove punctuation", value=st.session_state.clean_punct)
    with c3:
        st.session_state.dedupe = st.checkbox("drop duplicate texts", value=st.session_state.dedupe)

    # -- Apply basic cleaning
    df["__text__"] = df[text_col].astype(str).apply(
        lambda t: basic_clean(t, st.session_state.clean_lower, st.session_state.clean_punct)
    )

    # -- Robust label parsing: accept bool/strings or numeric, fallback to 0
    raw_lbl = df[label_col]
    if raw_lbl.dtype == bool:
        df["__label__"] = raw_lbl.astype(int)
    else:
        mapping = raw_lbl.astype(str).str.strip().str.lower().map({
            "1": 1, "true": 1, "yes": 1, "sarcastic": 1,
            "0": 0, "false": 0, "no": 0, "not sarcastic": 0
        })
        df["__label__"] = pd.to_numeric(raw_lbl, errors="coerce")
        df.loc[df["__label__"].isna(), "__label__"] = mapping
        df["__label__"] = df["__label__"].fillna(0).astype(int)

    # -- Optional deduplication (helps remove exact duplicate headlines)
    if st.session_state.dedupe:
        df = df.drop_duplicates(subset="__text__")

    # -- Quick class balance report
    st.markdown(f'<div class="pill">Rows after cleaning: {len(df):,}</div>', unsafe_allow_html=True)
    with st.expander("Class balance", expanded=True):
        vc = df["__label__"].value_counts().sort_index()
        n0 = int(vc.get(0, 0)); n1 = int(vc.get(1, 0)); N = max(1, n0 + n1)
        st.write(pd.DataFrame({
            "class": ["Not Sarcastic (0)", "Sarcastic (1)"],
            "count": [n0, n1],
            "percent": [round(100*n0/N, 2), round(100*n1/N, 2)]
        }))

    # -- Train/Test split controls
    st.subheader("Train/Test Split")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.test_size = st.slider("Test size fraction", 0.1, 0.4, float(st.session_state.test_size), 0.05)
    with c2:
        st.session_state.random_state = st.number_input("Random state", 0, 10_000, int(st.session_state.random_state), step=1)

    # -- Use stratify only if each class has >= 2 examples to avoid sklearn error
    counts = df["__label__"].value_counts()
    min_count = int(counts.min()) if len(counts) > 0 else 0
    stratify_arg = df["__label__"].values if min_count >= 2 else None
    if stratify_arg is None:
        st.warning("Stratified split disabled because at least one class has < 2 samples.")

    # -- Split data
    X = df["__text__"].values
    y = df["__label__"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=st.session_state.test_size,
        random_state=st.session_state.random_state,
        stratify=stratify_arg
    )

    # -- Downsampling ratio control (majority:minority ≥ 1.0)
    st.subheader("Imbalance Handling — Downsampling Ratio")
    st.caption("Reduce the majority class **in the training set** to reach a majority:minority ratio ≥ 1.0. "
               "Example: 1.0 → 50/50; 1.5 → majority ≈ 1.5× minority. Only the majority class is reduced.")
    st.session_state.down_maj_mult = st.slider("Target majority:minority ratio", 1.0, 3.0, float(st.session_state.down_maj_mult), 0.1)

    # -- Load ELMo (first time only)
    st.subheader("ELMo Embeddings")
    if st.session_state.elmo is None:
        if not TF_OK:
            st.error("""TensorFlow / tensorflow-hub not available or incompatible.
Install exact versions:

pip install tensorflow==2.15.0 tensorflow-hub==0.12.0

(ELMo v3 requires TF1 hub.Module.)""")
            return
        with st.spinner("Loading ELMo module from TF Hub… (first run may take a while)"):
            try:
                st.session_state.elmo = ELMoEmbedder(ELMO_URL)
            except Exception as e:
                st.error(f"Failed to load ELMo: {e}")
                return
        st.success("ELMo loaded.")

    # -- Embed train/test with ELMo (batched for efficiency)
    bsz = 32
    with st.spinner("Embedding training texts with ELMo…"):
        X_train_emb = st.session_state.elmo.embed(X_train, batch_size=bsz)
    with st.spinner("Embedding test texts with ELMo…"):
        X_test_emb = st.session_state.elmo.embed(X_test, batch_size=bsz)

    # -- Standardize embeddings for Logistic Regression (helps optimization)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train_emb)
    X_test_std  = scaler.transform(X_test_emb)

    # -- Downsampling on the training set (both LR & RF views) using chosen ratio
    maj_mult = float(st.session_state.down_maj_mult)
    
    # -- Visualize distribution before vs after downsampling
    st.subheader("Downsampling distribution charts")
    st.caption("Before vs after downsampling (training set).")
    st_plot_dist(y_before=y_train, y_after=y_lr_train, title="Class distribution (LR view)")
    st_plot_dist(y_before=y_train, y_after=y_rf_train, title="Class distribution (RF view)")

    # -- Report pre/post downsampling counts to the user
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

    # -- Persist artifacts to session state for the next pages
    st.session_state.X_train_emb = X_train_emb
    st.session_state.X_test_emb  = X_test_emb
    st.session_state.y_train     = y_train
    st.session_state.y_test      = y_test
    st.session_state.scaler      = scaler
st.session_state.prep_cache  = {
        "X_lr_train": X_lr_train, "y_lr_train": y_lr_train,
        "X_rf_train": X_rf_train, "y_rf_train": y_rf_train,
        "X_test_std": X_test_std
    }

st.success("Preprocessing complete. Proceed to **3) Model Training**.")

# ==============================
# Page 3 — Model Training
# ==============================
def page_train():
    """Train Logistic Regression (on standardized embeddings) and Random Forest (raw embeddings)."""
    st.title("3) Model Training")
    required = ["X_train_emb", "X_test_emb", "y_train", "y_test", "scaler", "prep_cache"]
    if not all(k in st.session_state and st.session_state[k] is not None for k in required):
        st.warning("Please finish **2) Data Preprocessing** first.")
        return

    cache = st.session_state.prep_cache
    X_lr_train = cache["X_lr_train"]; y_lr_train = cache["y_lr_train"]
    X_rf_train = cache["X_rf_train"]; y_rf_train = cache["y_rf_train"]

    # -- Simple hyperparameter controls
    st.subheader("Hyperparameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        C = st.number_input("Logistic Regression C (inverse regularization)", 0.01, 100.0, 1.0, step=0.05)
    with c2:
        n_estimators = st.number_input("RandomForest n_estimators", 50, 1000, 300, step=50)
    with c3:
        max_depth = st.number_input("RandomForest max_depth (0 = None)", 0, 100, 0, step=1)
        max_depth = None if max_depth == 0 else int(max_depth)

    # -- Train LR
    colA, colB = st.columns(2)
    with colA:
        with st.spinner("Training Logistic Regression…"):
            lr = LogisticRegression(C=C, solver="liblinear", random_state=st.session_state.random_state)
            lr.fit(X_lr_train, y_lr_train)

    # -- Train RF
    with colB:
        with st.spinner("Training Random Forest…"):
            rf = RandomForestClassifier(
                n_estimators=int(n_estimators),
                max_depth=max_depth,
                random_state=st.session_state.random_state,
                n_jobs=-1
            )
            rf.fit(X_rf_train, y_rf_train)

    st.session_state.models = {"lr": lr, "rf": rf}
    st.success("Training complete. Proceed to **4) Model Evaluation**.")

# ==============================
# Evaluation Utilities
# ==============================
def metric_table(metrics_dict):
    """Convert a nested metrics dictionary into a displayable DataFrame."""
    rows = []
    for model_name, m in metrics_dict.items():
        rows.append([model_name, m["Precision"], m["Recall"], m["F1"], m["ROC-AUC"]])
    return pd.DataFrame(rows, columns=["Model", "Precision", "Recall", "F1", "ROC-AUC"])

def _safe_roc_auc(y_true, scores):
    """Compute ROC-AUC, returning NaN when y_true is single-class (sklearn limitation)."""
    try:
        return roc_auc_score(y_true, scores)
    except Exception:
        return float("nan")

# ==============================
# Page 4 — Model Evaluation
# ==============================
def page_evaluation():
    """Compare LR vs RF using Precision, Recall, F1, ROC-AUC; show confusion matrices and ROC curves."""
    st.title("4) Model Evaluation")
    req = ["models", "X_test_emb", "y_test", "scaler", "prep_cache"]
    if not all(k in st.session_state and st.session_state[k] is not None for k in req):
        st.warning("Train models in **3) Model Training** first.")
        return

    models     = st.session_state.models
    scaler     = st.session_state.scaler
    X_test_emb = st.session_state.X_test_emb
    y_test     = st.session_state.y_test
    X_test_std = st.session_state.prep_cache["X_test_std"]

    lr = models["lr"]
    rf = models["rf"]

    # -- Probabilities for ROC-AUC; predictions will use a chosen threshold
    lr_proba = lr.predict_proba(X_test_std)[:, 1]
    rf_proba = rf.predict_proba(X_test_emb)[:, 1]

    # -- Decision threshold slider (affects P/R/F1; not ROC-AUC)
    st.session_state.threshold = st.slider(
        "Decision threshold (affects Precision/Recall/F1)", 0.1, 0.9, float(st.session_state.threshold), 0.05
    )
    thresh = st.session_state.threshold
    lr_pred = (lr_proba >= thresh).astype(int)
    rf_pred = (rf_proba >= thresh).astype(int)

    # -- Compute metrics safely (ROC-AUC may be NaN if single-class y_test)
    metrics = {
        "Logistic Regression": {
            "Precision": precision_score(y_test, lr_pred, zero_division=0),
            "Recall":    recall_score(y_test, lr_pred, zero_division=0),
            "F1":        f1_score(y_test, lr_pred, zero_division=0),
            "ROC-AUC":   _safe_roc_auc(y_test, lr_proba),
        },
        "Random Forest": {
            "Precision": precision_score(y_test, rf_pred, zero_division=0),
            "Recall":    recall_score(y_test, rf_pred, zero_division=0),
            "F1":        f1_score(y_test, rf_pred, zero_division=0),
            "ROC-AUC":   _safe_roc_auc(y_test, rf_proba),
        }
    }

    # -- Tabs: metrics table, confusion matrices, and ROC curves
    tab_perf, tab_cm, tab_roc = st.tabs(["Performance", "Confusion Matrices", "ROC Curves"])

    with tab_perf:
        st.subheader("Performance Comparison")
        dfm = metric_table(metrics).round(4)
        st.dataframe(dfm, use_container_width=True)
        # Simple "which is better by F1" indicator
        try:
            better = "Logistic Regression" if dfm.set_index("Model").loc["Logistic Regression", "F1"] >= dfm.set_index("Model").loc["Random Forest", "F1"] else "Random Forest"
            st.markdown(f"**Better F1 (at threshold={thresh:.2f}):** `{better}`")
        except Exception:
            pass

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
        if len(np.unique(y_test)) < 2:
            st.warning("ROC curves require both classes in y_test. Your test split has a single class. "
                       "Try reducing test size or disabling deduplication.")
        else:
            fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
            fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
            fig = plt.figure(figsize=(6, 5))
            plt.plot(fpr_lr, tpr_lr, label=f"LogReg (AUC={metrics['Logistic Regression']['ROC-AUC']:.3f})")
            plt.plot(fpr_rf, tpr_rf, label=f"RandForest (AUC={metrics['Random Forest']['ROC-AUC']:.3f})")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
            plt.title("ROC Curves"); plt.legend(loc="lower right")
            st.pyplot(fig)

# ==============================
# Page 5 — Prediction
# ==============================
def page_prediction():
    """
    Two modes:
    - Single text: quick input box, returns LR/RF probabilities and classes.
    - Batch CSV: upload a file, choose the text column, and download predictions.
    """
    st.title("5) Prediction")

    req = ["models", "scaler", "elmo"]
    if not all(k in st.session_state and st.session_state[k] is not None for k in req):
        st.warning("Please complete **Training** before predicting.")
        return

    models    = st.session_state.models
    scaler    = st.session_state.scaler
    elmo      = st.session_state.elmo
    threshold = st.session_state.get("threshold", 0.5)

    tab_single, tab_batch = st.tabs(["Single Text", "Batch Upload"])

    # -- Single text prediction
    with tab_single:
        text = st.text_area("Enter headline / text", height=120,
                            placeholder="e.g., 'Local man wins lottery, quits job to pursue full-time napping career'")
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

    # -- Batch CSV prediction
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
                texts   = bdf[text_col_name].astype(str).tolist()
                emb     = elmo.embed(texts, batch_size=32)
                x_std   = scaler.transform(emb)
                lr_prob = models["lr"].predict_proba(x_std)[:, 1]
                rf_prob = models["rf"].predict_proba(emb)[:, 1]
                lr_pred = (lr_prob >= threshold).astype(int)
                rf_pred = (rf_prob >= threshold).astype(int)

                out = bdf.copy()
                out["proba_lr"] = lr_prob
                out["pred_lr"]  = lr_pred
                out["proba_rf"] = rf_prob
                out["pred_rf"]  = rf_pred

            # -- Offer a download of results
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = f"sarcasm_predictions_{ts}.csv"
            out.to_csv(out_path, index=False)
            st.success(f"Done. Saved to {out_path}")
            st.download_button("Download predictions CSV",
                               data=out.to_csv(index=False).encode(),
                               file_name=out_path, mime="text/csv")

# ==============================
# Router: Display the selected page
# ==============================
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
