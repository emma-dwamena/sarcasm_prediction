#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sarcasm Detection App (Kaggle) — ELMo + Logistic Regression & Random Forest

Pages:
1) Data Loading
2) Data Preprocessing
3) Model Training
4) Model Evaluation
5) Prediction

How to run:
    pip install -U streamlit pandas numpy scikit-learn matplotlib tensorflow tensorflow_hub
    streamlit run sarcasm_streamlit_app.py

Notes:
- ELMo (from TF-Hub) will download the model the first time you run the app (needs internet).
- Kaggle dataset suggested: "News Headlines Dataset For Sarcasm Detection"
  Typical file: Sarcasm_Headlines_Dataset.json (JSON Lines) with keys: headline, is_sarcastic, article_link
"""

import io
import os
import re
import json
import time
import math
import typing as T

import numpy as np
import pandas as pd
import streamlit as st

# ML + Metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix, classification_report
)

# Plotting
import matplotlib.pyplot as plt

# TensorFlow / Hub (ELMo)
try:
    import tensorflow as tf
    import tensorflow_hub as hub
except Exception as e:
    tf = None
    hub = None


# ----------------------------
# App Config
# ----------------------------
st.set_page_config(
    page_title="Sarcasm Detection (ELMo + LR/RF)",
    layout="wide",
    page_icon="📰"
)

PAGES = [
    "1) Data Loading",
    "2) Data Preprocessing",
    "3) Model Training",
    "4) Model Evaluation",
    "5) Prediction",
]

ELMO_URL_DEFAULT = "https://tfhub.dev/google/elmo/3"  # Sentence-level embedding (1024-d)

# ----------------------------
# Helpers
# ----------------------------

def ensure_tfhub_available():
    if tf is None or hub is None:
        st.error("TensorFlow and/or TensorFlow Hub are not available. "
                 "Please install them:\n\n"
                 "`pip install tensorflow tensorflow_hub`")
        st.stop()


@st.cache_resource(show_spinner=False)
def load_elmo_layer(elmo_url: str):
    """Load ELMo KerasLayer from TF-Hub (cached)."""
    ensure_tfhub_available()
    with st.spinner("Loading ELMo from TensorFlow Hub... (first time may take a minute)"):
        # KerasLayer returns shape [batch, 1024] for string inputs
        layer = hub.KerasLayer(elmo_url, input_shape=[], dtype=tf.string, trainable=False)
    return layer


def clean_text(text: str) -> str:
    """Basic text cleanup; ELMo doesn't require heavy preprocessing, keep it light."""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    # normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # optional mild normalization (URLs -> token, lowercasing)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " <url> ", text)
    return text


def batch_iter(lst: T.List[str], batch_size: int) -> T.Iterator[T.List[str]]:
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def elmo_embed_texts(texts: T.List[str], elmo_layer, batch_size: int = 64) -> np.ndarray:
    """Compute sentence-level ELMo embeddings in batches. Returns [N, 1024]."""
    ensure_tfhub_available()
    all_vecs: T.List[np.ndarray] = []
    total = len(texts)
    prog = st.progress(0.0, text="Embedding with ELMo...")
    done = 0
    for chunk in batch_iter(texts, batch_size):
        # KerasLayer accepts tf.string tensors
        emb = elmo_layer(tf.constant(chunk))
        # emb shape: [batch, 1024]
        vecs = emb.numpy()
        all_vecs.append(vecs)
        done += len(chunk)
        prog.progress(min(1.0, done / total), text=f"Embedding with ELMo... ({done}/{total})")
    prog.empty()
    return np.vstack(all_vecs) if all_vecs else np.zeros((0, 1024), dtype=np.float32)


def safe_get(df: pd.DataFrame, keys: T.List[str]) -> str:
    for k in keys:
        if k in df.columns:
            return k
    return None


def set_state(**kwargs):
    for k, v in kwargs.items():
        st.session_state[k] = v


def get_state(k, default=None):
    return st.session_state.get(k, default)


def state_has(keys: T.List[str]) -> bool:
    return all(k in st.session_state for k in keys)


def metric_table(metrics_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame(metrics_dict).T
    df = df[["Precision", "Recall", "F1", "ROC-AUC"]]
    return df


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.title("📰 Sarcasm Detection")
    st.caption("ELMo embeddings + Logistic Regression & Random Forest")

    page = st.radio("Navigate", PAGES, index=0)

    st.divider()
    st.subheader("Run Settings")
    elmo_url = st.text_input("TensorFlow Hub ELMo URL", value=get_state("elmo_url", ELMO_URL_DEFAULT))
    set_state(elmo_url=elmo_url)

    if "models" in st.session_state:
        st.success("Models trained ✅")
    else:
        st.info("Models not trained yet")

    if "data" in st.session_state:
        st.success("Data loaded ✅")
    else:
        st.info("Data not loaded")


# ----------------------------
# Page 1 — Data Loading
# ----------------------------
def page_data_loading():
    st.header("1) Data Loading")
    st.write("Upload the Kaggle sarcasm dataset file (JSON Lines or CSV). "
             "For the Kaggle **News Headlines Dataset For Sarcasm Detection**, the JSON lines file typically "
             "contains fields: `headline`, `is_sarcastic`, optionally `article_link`.")

    uploaded = st.file_uploader("Upload JSONL (.json) or CSV file", type=["json", "csv"], accept_multiple_files=False)

    col1, col2 = st.columns([2, 1])
    with col1:
        sample_n = st.number_input("Optional: randomly sample N rows (0 = keep all)", min_value=0, value=0, step=100)

    with col2:
        seed = st.number_input("Random seed", min_value=0, value=42, step=1)

    if uploaded is not None:
        ext = os.path.splitext(uploaded.name)[-1].lower()
        try:
            if ext == ".json":
                df = pd.read_json(uploaded, lines=True)
            elif ext == ".csv":
                df = pd.read_csv(uploaded)
            else:
                st.error("Unsupported file type. Please upload .json or .csv")
                return
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            return

        # Identify columns
        text_col = safe_get(df, ["headline", "text", "content", "review", "comment"])
        label_col = safe_get(df, ["is_sarcastic", "label", "target", "sarcastic"])

        if text_col is None or label_col is None:
            st.warning("Could not auto-detect text/label columns. Please select them below.")
            cols = st.columns(2)
            with cols[0]:
                text_col = st.selectbox("Text column", df.columns.tolist())
            with cols[1]:
                label_col = st.selectbox("Label column (0/1)", df.columns.tolist())
        else:
            st.success(f"Detected text column: **{text_col}**, label column: **{label_col}**")

        # Optionally sample
        if sample_n and sample_n > 0 and sample_n < len(df):
            df = df.sample(n=sample_n, random_state=int(seed)).reset_index(drop=True)

        # Ensure binary labels
        try:
            labels = df[label_col].astype(int)
        except Exception:
            # Try mapping truthy strings to 1/0
            labels = df[label_col].astype(str).str.lower().map(
                {"1": 1, "true": 1, "yes": 1, "y": 1, "t": 1,
                 "0": 0, "false": 0, "no": 0, "n": 0, "f": 0}
            ).fillna(0).astype(int)

        df_clean = pd.DataFrame({
            "text": df[text_col].astype(str),
            "label": labels
        })

        st.write("Preview:")
        st.dataframe(df_clean.head(20))

        st.info(f"Rows: {len(df_clean)}, Positives (1): {int((df_clean['label']==1).sum())}, "
                f"Negatives (0): {int((df_clean['label']==0).sum())}")

        if st.button("Save dataset to session", type="primary"):
            set_state(data=df_clean)
            st.success("Dataset saved. Proceed to **2) Data Preprocessing** from the sidebar.")
    else:
        st.info("Awaiting file upload...")


# ----------------------------
# Page 2 — Data Preprocessing
# ----------------------------
def page_preprocessing():
    st.header("2) Data Preprocessing")
    if "data" not in st.session_state:
        st.warning("Please load data first in **1) Data Loading**.")
        return

    df: pd.DataFrame = st.session_state["data"].copy()

    st.subheader("Cleaning")
    st.write("Minimal cleaning is applied (lowercasing, URL masking). ELMo is robust to raw text, "
             "so heavy tokenization is not required.")
    apply_clean = st.checkbox("Apply basic cleaning", value=True)

    if apply_clean:
        df["text"] = df["text"].astype(str).map(clean_text)

    st.subheader("Train / Test Split")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    with col2:
        stratify = st.checkbox("Stratify by label", value=True)
    with col3:
        random_state = st.number_input("Random seed", min_value=0, value=42, step=1)

    X = df["text"].tolist()
    y = df["label"].astype(int).to_numpy()

    if st.button("Create Split", type="primary"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state),
            stratify=y if stratify else None
        )
        set_state(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        st.success(f"Split created. Train: {len(X_train)}, Test: {len(X_test)}. "
                   f"Proceed to **3) Model Training**.")


    # Show distribution
    if "X_train" in st.session_state:
        c1, c2 = st.columns(2)
        with c1:
            st.write("Train label distribution")
            train_labels = pd.Series(st.session_state["y_train"]).value_counts().rename({0:"Not Sarcastic",1:"Sarcastic"})
            st.bar_chart(train_labels)
        with c2:
            st.write("Test label distribution")
            test_labels = pd.Series(st.session_state["y_test"]).value_counts().rename({0:"Not Sarcastic",1:"Sarcastic"})
            st.bar_chart(test_labels)


# ----------------------------
# Page 3 — Model Training
# ----------------------------
def page_training():
    st.header("3) Model Training")
    required_keys = ["X_train", "X_test", "y_train", "y_test"]
    if not state_has(required_keys):
        st.warning("Please finish **2) Data Preprocessing** first.")
        return

    X_train: T.List[str] = get_state("X_train")
    X_test: T.List[str] = get_state("X_test")
    y_train: np.ndarray = get_state("y_train")
    y_test: np.ndarray = get_state("y_test")

    st.subheader("ELMo Embedding")
    col1, col2 = st.columns(2)
    with col1:
        batch_size = st.number_input("Embedding batch size", min_value=8, max_value=512, value=64, step=8)
    with col2:
        use_cache_embeddings = st.checkbox("Cache embeddings in session", value=True)

    # Load ELMo Layer
    try:
        elmo_layer = load_elmo_layer(get_state("elmo_url", ELMO_URL_DEFAULT))
    except Exception as e:
        st.error(f"Failed to load ELMo from TF-Hub: {e}")
        return

    # Compute embeddings
    if use_cache_embeddings and "X_train_emb" in st.session_state and "X_test_emb" in st.session_state:
        X_train_emb = st.session_state["X_train_emb"]
        X_test_emb = st.session_state["X_test_emb"]
        st.info("Using cached embeddings.")
    else:
        X_train_emb = elmo_embed_texts(X_train, elmo_layer, batch_size=batch_size)
        X_test_emb = elmo_embed_texts(X_test, elmo_layer, batch_size=batch_size)
        if use_cache_embeddings:
            set_state(X_train_emb=X_train_emb, X_test_emb=X_test_emb)

    st.write("Train embeddings shape:", X_train_emb.shape, "Test embeddings shape:", X_test_emb.shape)

    st.subheader("Model Hyperparameters")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Logistic Regression**")
        lr_c = st.number_input("C (inverse regularization strength)", min_value=0.001, max_value=100.0, value=1.0, step=0.1)
        lr_max_iter = st.number_input("max_iter", min_value=100, max_value=5000, value=1000, step=100)
        lr_penalty = st.selectbox("penalty", ["l2"], index=0)
        lr_solver = st.selectbox("solver", ["lbfgs", "liblinear", "saga"], index=0)
    with c2:
        st.markdown("**Random Forest**")
        rf_n_estimators = st.number_input("n_estimators", min_value=50, max_value=1000, value=300, step=50)
        rf_max_depth = st.number_input("max_depth (0 means None)", min_value=0, max_value=200, value=0, step=1)
        rf_min_samples_leaf = st.number_input("min_samples_leaf", min_value=1, max_value=20, value=1, step=1)
        rf_random_state = st.number_input("random_state", min_value=0, value=42, step=1)

    if st.button("Train Both Models", type="primary"):
        # Standardize for LR (helps optimization). Not necessary for RF, but harmless.
        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train_emb)
        X_test_std = scaler.transform(X_test_emb)

        lr = LogisticRegression(
            C=float(lr_c), penalty=lr_penalty, solver=lr_solver,
            max_iter=int(lr_max_iter), n_jobs=None
        )
        lr.fit(X_train_std, y_train)

        rf = RandomForestClassifier(
            n_estimators=int(rf_n_estimators),
            max_depth=None if rf_max_depth == 0 else int(rf_max_depth),
            min_samples_leaf=int(rf_min_samples_leaf),
            random_state=int(rf_random_state),
            n_jobs=-1
        )
        rf.fit(X_train_emb, y_train)  # RF on unscaled features

        artifacts = {
            "scaler": scaler,
            "lr": lr,
            "rf": rf,
        }
        set_state(models=artifacts)
        st.success("Training complete. Go to **4) Model Evaluation**.")


# ----------------------------
# Page 4 — Model Evaluation
# ----------------------------
def page_evaluation():
    st.header("4) Model Evaluation")
    required = ["models", "X_test_emb", "y_test", "X_test"]
    if not state_has(required):
        st.warning("Please train models in **3) Model Training** first.")
        return

    models = get_state("models")
    scaler: StandardScaler = models["scaler"]
    lr: LogisticRegression = models["lr"]
    rf: RandomForestClassifier = models["rf"]

    X_test_emb: np.ndarray = get_state("X_test_emb")
    y_test: np.ndarray = get_state("y_test")

    # Predict probabilities
    X_test_std = scaler.transform(X_test_emb)
    lr_proba = lr.predict_proba(X_test_std)[:, 1]
    rf_proba = rf.predict_proba(X_test_emb)[:, 1]

    # Class labels
    thresh = st.slider("Decision threshold (for Precision/Recall/F1)", 0.1, 0.9, 0.5, 0.05)
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

    st.subheader("Performance Comparison")
    dfm = metric_table(metrics).round(4)
    st.dataframe(dfm, use_container_width=True)

    # Confusion matrices
    c1, c2 = st.columns(2)
    with c1:
        st.write("Confusion Matrix — Logistic Regression")
        cm_lr = confusion_matrix(y_test, lr_pred)
        st.write(pd.DataFrame(cm_lr, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))
    with c2:
        st.write("Confusion Matrix — Random Forest")
        cm_rf = confusion_matrix(y_test, rf_pred)
        st.write(pd.DataFrame(cm_rf, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

    # ROC Curves
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

    st.caption("Metrics computed on the held-out test split. "
               "Precision/Recall/F1 depend on the selected decision threshold; ROC-AUC uses score distributions.")


# ----------------------------
# Page 5 — Prediction
# ----------------------------
def page_prediction():
    st.header("5) Prediction")
    required = ["models", "X_train_emb"]
    if not state_has(required):
        st.warning("Please train models in **3) Model Training** first.")
        return

    models = get_state("models")
    scaler: StandardScaler = models["scaler"]
    lr: LogisticRegression = models["lr"]
    rf: RandomForestClassifier = models["rf"]

    # Load ELMo
    try:
        elmo_layer = load_elmo_layer(get_state("elmo_url", ELMO_URL_DEFAULT))
    except Exception as e:
        st.error(f"Failed to load ELMo: {e}")
        return

    st.write("Enter a headline (or multiple, one per line):")
    user_text = st.text_area("Input", height=150, placeholder="example: the weather today is great...")

    col1, col2 = st.columns(2)
    with col1:
        threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05)
    with col2:
        apply_clean = st.checkbox("Apply same basic cleaning", value=True)

    if st.button("Predict", type="primary"):
        sentences = [s.strip() for s in user_text.split("\n") if s.strip()]
        if not sentences:
            st.warning("Please enter at least one line of text.")
            return

        if apply_clean:
            sentences = [clean_text(s) for s in sentences]

        emb = elmo_embed_texts(sentences, elmo_layer, batch_size=32)
        emb_std = scaler.transform(emb)

        lr_proba = lr.predict_proba(emb_std)[:, 1]
        rf_proba = rf.predict_proba(emb)[:, 1]

        results = []
        for s, p1, p2 in zip(sentences, lr_proba, rf_proba):
            results.append({
                "text": s,
                "LR_prob_sarcastic": float(p1),
                "LR_label": int(p1 >= threshold),
                "RF_prob_sarcastic": float(p2),
                "RF_label": int(p2 >= threshold),
            })
        df_res = pd.DataFrame(results)
        st.dataframe(df_res, use_container_width=True)

        st.success("Done!")


# ----------------------------
# Router
# ----------------------------
if page == "1) Data Loading":
    page_data_loading()
elif page == "2) Data Preprocessing":
    page_preprocessing()
elif page == "3) Model Training":
    page_training()
elif page == "4) Model Evaluation":
    page_evaluation()
elif page == "5) Prediction":
    page_prediction()
else:
    st.write("Use the sidebar to navigate.")
