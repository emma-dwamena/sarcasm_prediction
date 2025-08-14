import os, io, re, json
from datetime import datetime

import numpy as np
import hashlib
import pandas as pd
import streamlit as st

# --- UI helper (session only): stable hash of a list of texts ---
def _hash_texts(texts):
    import hashlib as _hl
    m = _hl.md5()
    for t in texts:
        if not isinstance(t, str):
            t = str(t)
        m.update(t.encode("utf-8"))
        m.update(b"\n")
    return m.hexdigest()
# --- end helper ---

# === UI helper: ELMo embedding with visible progress (no ML changes) ===
def _embed_with_progress(texts, elmo_module, batch_size=32, label="texts"):
    total = len(texts)
    if total == 0:
        return np.zeros((0, 1024), dtype=np.float32)
    prog = st.progress(0.0, text=f"Embedding {label} with ELMo… 0/{total}")
    out = []
    done = 0
    for i in range(0, total, batch_size):
        chunk = texts[i:i+batch_size]
        # Use the existing elmo .embed call to preserve behavior
        emb = elmo_module.embed(chunk, batch_size=len(chunk))
        out.append(emb)
        done = min(i + batch_size, total)
        prog.progress(done/total, text=f"Embedding {label} with ELMo… {done}/{total}")
    prog.empty()
    try:
        return np.vstack(out)
    except Exception:
        # If the embed already returns a single array (not per-batch), just return the last result
        return out[-1] if out else np.zeros((0, 1024), dtype=np.float32)
# === End helper ===

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

# ------------------------------
# ELMo via TensorFlow Hub (TF1)
# ------------------------------
ELMO_URL = "https://tfhub.dev/google/elmo/3"
TF_OK = True
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    tf.get_logger().setLevel('ERROR')
    tf.compat.v1.disable_eager_execution()
    _HAS_MODULE = hasattr(hub, "Module")
    if not _HAS_MODULE:
        TF_OK = False
except Exception:
    TF_OK = False

# ==============================
# Streamlit: Page Config & Theme
# ==============================
st.set_page_config(page_title="Sarcasm Detection", page_icon="📰", layout="wide")




# ==============================
# Session-State Initialization
# ==============================
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
    ss.setdefault("down_maj_mult", 1.0)
    ss.setdefault("elmo", None)
    ss.setdefault("X_train_emb", None)
    ss.setdefault("X_test_emb", None)
    ss.setdefault("X_train_key", None)
    ss.setdefault("X_test_key", None)
    ss.setdefault("y_train", None)
    ss.setdefault("y_test", None)
    ss.setdefault("scaler", None)
    ss.setdefault("models", {})
    ss.setdefault("threshold", 0.5)
    ss.setdefault("prep_cache", None)

_init_state()

# ==============================
# Basic Cleaning
# ==============================
_punct_pattern = re.compile(r"[^\w\s]")
def basic_clean(text, lower=True, remove_punct=True):
    if not isinstance(text, str): return ""
    t = text.strip()
    if lower: t = t.lower()
    if remove_punct: t = _punct_pattern.sub(" ", t)
    t = re.sub(r"\s+", " ", t)
    return t

# ==============================
# ELMo Embedder
# ==============================
class ELMoEmbedder:
    def __init__(self, url: str = ELMO_URL):
        if not TF_OK:
            st.error("""TensorFlow / tensorflow-hub not available or incompatible.
Install exact versions:

pip install tensorflow==2.15.0 tensorflow-hub==0.12.0

(ELMo v3 requires TF1 hub.Module.)""")
            raise RuntimeError("TF/Hub unavailable")
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.text_input = tf.compat.v1.placeholder(tf.string, shape=[None], name="text_input")
            self.module = hub.Module(url, trainable=False, name="elmo_module")
            elmo_out = self.module(self.text_input, signature="default", as_dict=True)["elmo"]
            self.sentence_emb = tf.reduce_mean(elmo_out, axis=1, name="sentence_embedding")
            self.init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer())
        self.sess = tf.compat.v1.Session(graph=self.graph)
        self.sess.run(self.init_op)

    def embed(self, texts, batch_size=32):
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
# Handling Imbalance - Downsampling
# ==============================
def downsample_ratio(X, y, maj_mult=1.0, random_state=42):
    """
    Downsample the majority class to achieve majority:minority ≈ maj_mult (>=1.0).
    Example: maj_mult=1.0 → 1:1; maj_mult=1.5 → majority ≈ 1.5× minority.
    """
    rng = np.random.RandomState(random_state)
    y = np.asarray(y).astype(int)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    n0, n1 = len(idx0), len(idx1)
    if n0 == 0 or n1 == 0:
        return X, y
    if n0 >= n1:
        maj_idx, min_idx = idx0, idx1
        nmaj, nmin = n0, n1
    else:
        maj_idx, min_idx = idx1, idx0
        nmaj, nmin = n1, n0
    target_maj = int(max(nmin, np.floor(nmin * float(maj_mult))))
    target_maj = min(target_maj, nmaj)
    if nmaj > target_maj:
        keep_maj = rng.choice(maj_idx, size=target_maj, replace=False)
    else:
        keep_maj = maj_idx
    keep_idx = np.concatenate([min_idx, keep_maj])
    rng.shuffle(keep_idx)
    return X[keep_idx], y[keep_idx]

# ==============================
# Downsampling Distribution Plot
# ==============================
def st_plot_dist(y_before, y_after, title):
    import matplotlib.pyplot as plt
    import numpy as np
    y_before = np.asarray(y_before).astype(int)
    y_after  = np.asarray(y_after).astype(int)
    def _counts(y):
        c = np.bincount(y, minlength=2)[:2]
        return int(c[0]), int(c[1])
    c0b, c1b = _counts(y_before)
    c0a, c1a = _counts(y_after)
    labels = ["Class 0 (Not Sarcastic)", "Class 1 (Sarcastic)"]
    x = np.arange(len(labels)); width = 0.35
    fig = plt.figure(figsize=(7, 5))
    plt.bar(x - width/2, [c0b, c1b], width, label="Before")
    plt.bar(x + width/2, [c0a, c1a], width, label="After")
    for i, v in enumerate([c0b, c1b]): plt.text(x[i] - width/2, v, str(v), ha='center', va='bottom')
    for i, v in enumerate([c0a, c1a]): plt.text(x[i] + width/2, v, str(v), ha='center', va='bottom')
    plt.xticks(x, labels); plt.ylabel("Count"); plt.title(title); plt.legend(loc="best"); plt.tight_layout()
    st.pyplot(fig)
def st_plot_cm(cm, title="Confusion Matrix", labels=("Actual 0","Actual 1"), preds=("Pred 0","Pred 1")):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(4.8, 4.2))
    ax = plt.gca()
    im = ax.imshow(cm, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1]); ax.set_xticklabels(list(preds))
    ax.set_yticks([0, 1]); ax.set_yticklabels(list(labels))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    st.pyplot(fig)

# ==============================
# Sidebar Navigation
# ==============================
st.sidebar.title("📰 Sarcasm Detector")
st.sidebar.markdown("---")
st.sidebar.caption("Upload → Preprocess → Train → Evaluate → Predict")
st.sidebar.markdown("---")

# ==============================
# Page 1 — Data Upload
# ==============================
def page_upload():
    st.title("Data Upload")
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
                st.error("Unsupported file type."); return
        except Exception as e:
            st.error(f"Failed to read file: {e}"); return
        st.session_state.df = df.copy()
        st.success(f"Loaded shape: {df.shape[0]} rows × {df.shape[1]} columns")
        with st.expander("Preview (first 10 rows)", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)
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
    st.title("Data Preprocessing")
    if st.session_state.df is None:
        st.warning("Please upload a dataset in **Data Upload**."); return
    df = st.session_state.df.copy()
    text_col = st.session_state.text_col; label_col = st.session_state.label_col
    if text_col is None or label_col is None:
        st.warning("Select text and label columns in **Data Upload**."); return

    st.subheader("Text Cleaning")
    c1, c2, c3 = st.columns(3)
    with c1: st.session_state.clean_lower = st.checkbox("lowercase", value=st.session_state.clean_lower)
    with c2: st.session_state.clean_punct = st.checkbox("remove punctuation", value=st.session_state.clean_punct)
    with c3: st.session_state.dedupe = st.checkbox("drop duplicate texts", value=st.session_state.dedupe)

    df["__text__"] = df[text_col].astype(str).apply(lambda t: basic_clean(t, st.session_state.clean_lower, st.session_state.clean_punct))

    raw_lbl = df[label_col]
    if raw_lbl.dtype == bool:
        df["__label__"] = raw_lbl.astype(int)
    else:
        mapping = raw_lbl.astype(str).str.strip().str.lower().map({"1":1,"true":1,"yes":1,"sarcastic":1,"0":0,"false":0,"no":0,"not sarcastic":0})
        df["__label__"] = pd.to_numeric(raw_lbl, errors="coerce"); df.loc[df["__label__"].isna(),"__label__"] = mapping
        df["__label__"] = df["__label__"].fillna(0).astype(int)

    if st.session_state.dedupe:
        df = df.drop_duplicates(subset="__text__")

    st.markdown(f'<div>Rows after cleaning: <b>{len(df):,}</b></div>', unsafe_allow_html=True)
    with st.expander("Class balance", expanded=True):
        vc = df["__label__"].value_counts().sort_index()
        n0 = int(vc.get(0,0)); n1 = int(vc.get(1,0)); N = max(1, n0+n1)
        st.write(pd.DataFrame({"class":["Not Sarcastic (0)","Sarcastic (1)"], "count":[n0,n1], "percent":[round(100*n0/N,2), round(100*n1/N,2)]}))

    st.subheader("Train/Test Split")
    c1, c2 = st.columns(2)
    with c1: st.session_state.test_size = st.slider("Test size", 0.1, 0.4, float(st.session_state.test_size), 0.05)
    with c2: st.session_state.random_state = st.number_input("Random state", 0, 10000, int(st.session_state.random_state), step=1)

    counts = df["__label__"].value_counts(); min_count = int(counts.min()) if len(counts)>0 else 0
    stratify_arg = df["__label__"].values if min_count >= 2 else None
    if stratify_arg is None:
        st.warning("Stratified split disabled because at least one class has < 2 samples.")

    X = df["__text__"].values; y = df["__label__"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=st.session_state.test_size,
                                                        random_state=st.session_state.random_state, stratify=stratify_arg)

    st.subheader("Handling Imbalance — Downsampling")
    st.caption("Reduce the majority class **in the training set** to reach a majority:minority ratio ≥ 1.0 (e.g., 1.0→50/50, 1.5→1.5×).")
    st.session_state.down_maj_mult = st.slider("Target majority:minority ratio", 1.0, 3.0, float(st.session_state.down_maj_mult), 0.1)

    st.subheader("ELMo Embeddings")
    if st.session_state.elmo is None:
        if not TF_OK:
            st.error("""TensorFlow / tensorflow-hub not available or incompatible.
Install exact versions:

pip install tensorflow==2.15.0 tensorflow-hub==0.12.0

(ELMo v3 requires TF1 hub.Module.)"""); return
        with st.spinner("Loading ELMo module from TF Hub…"):
            try:
                st.session_state.elmo = ELMoEmbedder(ELMO_URL)
            except Exception as e:
                st.error(f"Failed to load ELMo: {e}"); return
        st.success("ELMo loaded.")
    bsz = 32
    # --- Session-cached ELMo embeddings (UI/session only) ---
    st.session_state.setdefault("X_train_key", None)
    st.session_state.setdefault("X_test_key", None)
    key_train = _hash_texts(X_train)
    key_test  = _hash_texts(X_test)
    reuse_train = (st.session_state.X_train_emb is not None and st.session_state.get("X_train_key") == key_train)
    reuse_test  = (st.session_state.X_test_emb is not None and st.session_state.get("X_test_key") == key_test)
    if reuse_train:
        X_train_emb = st.session_state.X_train_emb
    else:
        X_train_emb = _embed_with_progress(X_train, st.session_state.elmo, batch_size=bsz, label="training texts")
        st.session_state.X_train_emb = X_train_emb
        st.session_state.X_train_key = key_train
    if reuse_test:
        X_test_emb = st.session_state.X_test_emb
    else:
        X_test_emb = _embed_with_progress(X_test, st.session_state.elmo, batch_size=bsz, label="test texts")
        st.session_state.X_test_emb = X_test_emb
        st.session_state.X_test_key = key_test
    # --- End session-cached embeddings ---
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train_emb)
    X_test_std  = scaler.transform(X_test_emb)

    maj_mult = float(st.session_state.down_maj_mult)
    X_lr_train, y_lr_train = downsample_ratio(X_train_std, y_train, maj_mult=maj_mult, random_state=st.session_state.random_state)
    X_rf_train, y_rf_train = downsample_ratio(X_train_emb, y_train, maj_mult=maj_mult, random_state=st.session_state.random_state)

    # Charts: before vs after
    st.subheader("Downsampling distribution charts")
    st.caption("Before vs after downsampling (training set).")
    st_plot_dist(y_before=y_train, y_after=y_lr_train, title="Class distribution (LR view)")
    st_plot_dist(y_before=y_train, y_after=y_rf_train, title="Class distribution (RF view)")

    # Save artifacts to session
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
    st.success("Preprocessing complete. Proceed to **Model Training**.")

# ==============================
# Page 3 — Model Training
# ==============================
def page_train():
    st.title("Model Training")
    required = ["X_train_emb", "X_test_emb", "y_train", "y_test", "scaler", "prep_cache"]
    if not all(k in st.session_state and st.session_state[k] is not None for k in required):
        st.warning("Please finish **Data Preprocessing** first."); return
    cache = st.session_state.prep_cache
    X_lr_train = cache["X_lr_train"]; y_lr_train = cache["y_lr_train"]
    X_rf_train = cache["X_rf_train"]; y_rf_train = cache["y_rf_train"]

    st.subheader("Hyperparameters")
    c1, c2, c3 = st.columns(3)
    with c1: C = st.number_input("Logistic Regression C", 0.01, 100.0, 1.0, step=0.05)
    with c2: n_estimators = st.number_input("RandomForest n_estimators", 50, 1000, 300, step=50)
    with c3:
        max_depth = st.number_input("RandomForest max_depth (0=None)", 0, 100, 0, step=1)
        max_depth = None if max_depth == 0 else int(max_depth)

    colA, colB = st.columns(2)
    with colA:
        with st.spinner("Training Logistic Regression…"):
            lr = LogisticRegression(C=C, solver="liblinear", random_state=st.session_state.random_state)
            lr.fit(X_lr_train, y_lr_train)
    with colB:
        with st.spinner("Training Random Forest…"):
            rf = RandomForestClassifier(n_estimators=int(n_estimators), max_depth=max_depth,
                                        random_state=st.session_state.random_state, n_jobs=-1)
            rf.fit(X_rf_train, y_rf_train)

    st.session_state.models = {"lr": lr, "rf": rf}
    st.success("Training complete. Proceed to **Model Evaluation**.")

# ==============================
# Page 4 — Model Evaluation
# ==============================
def _safe_auc(y_true, scores):
    try: return roc_auc_score(y_true, scores)
    except Exception: return float("nan")

def page_evaluation():
    st.title("Model Evaluation")
    req = ["models", "X_test_emb", "y_test", "scaler", "prep_cache"]
    if not all(k in st.session_state and st.session_state[k] is not None for k in req):
        st.warning("Train models in **Model Training** first."); return

    models = st.session_state.models
    scaler = st.session_state.scaler
    X_test_emb = st.session_state.X_test_emb
    y_test = st.session_state.y_test
    X_test_std = st.session_state.prep_cache["X_test_std"]

    lr = models["lr"]; rf = models["rf"]
    lr_proba = lr.predict_proba(X_test_std)[:, 1]
    rf_proba = rf.predict_proba(X_test_emb)[:, 1]

    st.session_state.threshold = st.slider("Decision threshold", 0.1, 0.9, float(st.session_state.threshold), 0.05)
    t = st.session_state.threshold
    lr_pred = (lr_proba >= t).astype(int); rf_pred = (rf_proba >= t).astype(int)

    metrics = {
        "Logistic Regression": {
            "Precision": precision_score(y_test, lr_pred, zero_division=0),
            "Recall":    recall_score(y_test, lr_pred, zero_division=0),
            "F1":        f1_score(y_test, lr_pred, zero_division=0),
            "ROC-AUC":   _safe_auc(y_test, lr_proba),
        },
        "Random Forest": {
            "Precision": precision_score(y_test, rf_pred, zero_division=0),
            "Recall":    recall_score(y_test, rf_pred, zero_division=0),
            "F1":        f1_score(y_test, rf_pred, zero_division=0),
            "ROC-AUC":   _safe_auc(y_test, rf_proba),
        }
    }
    tab_perf, tab_cm, tab_roc = st.tabs(["Performance", "Confusion Matrices", "ROC Curves"])
    with tab_perf:
        dfm = pd.DataFrame([[k, v["Precision"], v["Recall"], v["F1"], v["ROC-AUC"]] for k,v in metrics.items()],
                           columns=["Model","Precision","Recall","F1","ROC-AUC"]).round(4)
        st.dataframe(dfm, use_container_width=True)
    with tab_cm:
        c1, c2 = st.columns(2)
        with c1:
            st.write("Confusion Matrix — Logistic Regression")
            cm_lr = confusion_matrix(y_test, lr_pred)
            st_plot_cm(cm_lr, title="LogReg Confusion Matrix", labels=("Actual 0","Actual 1"), preds=("Pred 0","Pred 1"))
        with c2:
            st.write("Confusion Matrix — Random Forest")
            cm_rf = confusion_matrix(y_test, rf_pred)
            st_plot_cm(cm_rf, title="RandForest Confusion Matrix", labels=("Actual 0","Actual 1"), preds=("Pred 0","Pred 1"))
    with tab_roc:
        import matplotlib.pyplot as plt
        if len(np.unique(y_test)) < 2:
            st.warning("ROC curves require both classes in y_test.")
        else:
            fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
            fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
            fig = plt.figure(figsize=(6,5))
            plt.plot(fpr_lr, tpr_lr, label=f"LogReg (AUC={metrics['Logistic Regression']['ROC-AUC']:.3f})")
            plt.plot(fpr_rf, tpr_rf, label=f"RandForest (AUC={metrics['Random Forest']['ROC-AUC']:.3f})")
            plt.plot([0,1],[0,1], linestyle="--"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curves"); plt.legend(loc="lower right")
            st.pyplot(fig)

# ==============================
# Page 5 — Prediction
# ==============================
def page_prediction():
    st.title("Prediction")
    req = ["models", "scaler", "elmo"]
    if not all(k in st.session_state and st.session_state[k] is not None for k in req):
        st.warning("Please complete **Training** before predicting."); return
    models = st.session_state.models; scaler = st.session_state.scaler; elmo = st.session_state.elmo
    threshold = st.session_state.get("threshold", 0.5)

    tab_single, tab_batch = st.tabs(["Single Text", "Batch Upload"])
    with tab_single:
        text = st.text_area("Enter headline / text", height=120, placeholder="e.g., 'Scientists discover water on the Sun (sure).'" )
        if st.button("Predict"):
            if not text.strip(): st.warning("Enter some text.")
            else:
                emb = elmo.embed([text]); x_std = scaler.transform(emb)
                lr_p = models["lr"].predict_proba(x_std)[:,1][0]; rf_p = models["rf"].predict_proba(emb)[:,1][0]
                lr_pred = int(lr_p >= threshold); rf_pred = int(rf_p >= threshold)
                c1, c2 = st.columns(2)
                with c1: st.metric("Logistic Regression", f"{'Sarcastic' if lr_pred else 'Not Sarcastic'}", delta=f"P={lr_p:.3f}")
                with c2: st.metric("Random Forest", f"{'Sarcastic' if rf_pred else 'Not Sarcastic'}", delta=f"P={rf_p:.3f}")

    with tab_batch:
        st.write("Upload a CSV for batch predictions.")
        bf = st.file_uploader("Upload CSV", type=["csv","txt"], key="batch_csv")
        text_col_name = st.text_input("Text column name in CSV", value=st.session_state.text_col or "headline")
        if bf is not None:
            try: bdf = pd.read_csv(bf)
            except Exception as e: st.error(f"Could not read CSV: {e}"); return
            if text_col_name not in bdf.columns:
                st.error(f"Column '{text_col_name}' not in CSV."); return
            with st.spinner("Embedding and predicting…"):
                texts = bdf[text_col_name].astype(str).tolist()
                emb = elmo.embed(texts, batch_size=32); x_std = scaler.transform(emb)
                lr_prob = models["lr"].predict_proba(x_std)[:,1]; rf_prob = models["rf"].predict_proba(emb)[:,1]
                lr_pred = (lr_prob >= threshold).astype(int); rf_pred = (rf_prob >= threshold).astype(int)
                out = bdf.copy()
                out["proba_lr"] = lr_prob; out["pred_lr"] = lr_pred
                out["proba_rf"] = rf_prob; out["pred_rf"] = rf_pred
            ts = datetime.now().strftime("%Y%m%d_%H%M%S"); out_path = f"sarcasm_predictions_{ts}.csv"
            out.to_csv(out_path, index=False)
            st.success(f"Done. Saved to {out_path}")
            st.download_button("Download predictions CSV", data=out.to_csv(index=False).encode(), file_name=out_path, mime="text/csv")

# ==============================
# Router
# ==============================
st.sidebar.markdown("---")

# --- Fixed top navigation tabs (UI-only; ML code untouched) ---
st.markdown(
    """
    <style>
    /* Pin the FIRST st.tabs container to the top */
    div[data-testid='stTabs']:first-of-type {
        position: fixed; top: 0; left: 0; right: 0; width: 100%;
        z-index: 10000;
        background: var(--bg, #f6f8fb);
        margin: 0;
    }
    /* Style the tablist and add a divider + shadow */
    div[data-testid='stTabs']:first-of-type > div[role='tablist'] {
        position: relative;
        border-bottom: 1px solid #e5e7eb;
        box-shadow: 0 2px 6px rgba(0,0,0,.06);
        padding-top: .35rem; padding-bottom: .35rem;
        background: transparent;
    }
    /* Make space so content isn't hidden under the fixed tabs */
    .block-container { padding-top: 4.8rem; }
    /* Keep Streamlit header/menu above tabs */
    header[data-testid='stHeader'] { z-index: 10100; }
    </style>
    """,
    unsafe_allow_html=True,
)

_tab_labels = [
    "About",
    "Home & Data Overview",
    "Data Preprocessing",
    "Model Training",
    "Model Evaluation",
    "Prediction Interface",
]
_tabs = st.tabs(_tab_labels)

with _tabs[0]:
    st.markdown("""
    <div style="background-color: #f2f7f7; padding: 2rem; border-radius: 1rem; margin-bottom: 2rem;">
        <h2 style="color: #030a0a; text-align: center;">📌 About This App</h2>
        <p style="text-align:center;max-width:900px;margin:0 auto;">
            This dashboard predicts Sarcasm in Comments Using ELMo
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color: #f2f7f7; padding: 2rem; border-radius: 1rem; margin-bottom: 2rem;">
        <h2 style="color: #030a0a; text-align: center;">👥 Group 7 Team Members</h2>
        <div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
            <div>• Erwin K. Opare-Essel - 22254064</div>
            <div>• Emmanuel Oduro Dwamena - 11410636</div>
            <div>• Elizabeth Afranewaa Abayateye - 22252474</div>
            <div>• Elien Samira Osumanu - 11410414</div>
            <div>• Innocent Arkaah- 11410788</div>
            <div>• Sheena Pognaa Dasoberi - 22252392</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with _tabs[1]:
    # Your original 'home/data' page
    page_upload()

with _tabs[2]:
    st.subheader("Data Preprocessing")
    if st.button("▶ Start Preprocessing", key="btn_preprocess_start"):
        page_preprocess()
    else:
        st.info("Click **Start Preprocessing** to begin.")

with _tabs[3]:
    st.subheader("Model Training")
    if st.button("▶ Start Training", key="btn_train_start"):
        page_train()
    else:
        st.info("Click **Start Training** to train models with the current preprocessing & hyperparameters.")

with _tabs[4]:
    page_evaluation()

with _tabs[5]:
    # Single & batch prediction UI (unique keys are already defined inside this function)
    page_prediction()


