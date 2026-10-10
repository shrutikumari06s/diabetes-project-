"""
app.py
======
Streamlit dashboard for the AI-Based Diabetes Risk Prediction project.

Start the dashboard with:
    python -m streamlit run app.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
DATASET_PATH  = os.path.join("data", "diabetes_prediction_dataset.csv")
PIPELINE_PATH = os.path.join("models", "best_pipeline.joblib")
METRICS_PATH  = os.path.join("models", "model_metrics.json")

# ─────────────────────────────────────────────
# AUTO-TRAIN: if model files are missing, train now
# (required for Streamlit Cloud where .joblib is not in git)
# ─────────────────────────────────────────────
def auto_train():
    """Run the training pipeline silently if model files do not exist."""
    if os.path.exists(PIPELINE_PATH) and os.path.exists(METRICS_PATH):
        return  # already trained

    if not os.path.exists(DATASET_PATH):
        return  # dataset missing — can't train

    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix,
    )

    df = pd.read_csv(DATASET_PATH).drop_duplicates()
    X = df.drop(columns=["diabetes"])
    y = df["diabetes"]

    numerical_features   = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
    binary_features      = ["hypertension", "heart_disease"]
    categorical_features = ["gender", "smoking_history"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(),                                        numerical_features),
        ("bin", "passthrough",                                           binary_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ], remainder="drop")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1),
    }

    results = {}
    for name, model in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        results[name] = {
            "accuracy":         round(float(accuracy_score(y_test, y_pred)),  4),
            "precision":        round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall":           round(float(recall_score(y_test, y_pred, zero_division=0)),    4),
            "f1_score":         round(float(f1_score(y_test, y_pred, zero_division=0)),        4),
            "roc_auc":          round(float(roc_auc_score(y_test, y_prob)),    4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "pipeline":         pipe,
        }

    best_name = max(results, key=lambda n: results[n]["f1_score"])
    os.makedirs("models", exist_ok=True)
    joblib.dump(results[best_name]["pipeline"], PIPELINE_PATH)

    metrics_out = {
        "best_model": best_name,
        "selection_criterion": "F1-score (positive class)",
        "models": {n: {k: v for k, v in r.items() if k != "pipeline"} for n, r in results.items()},
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_out, f, indent=2)

# Run auto-train before anything else loads
with st.spinner("Setting up models for first run — please wait (1-2 minutes)..."):
    auto_train()

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Prediction Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.0rem;
        font-weight: 700;
        color: #1a3a5c;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4a6080;
        margin-bottom: 1rem;
    }
    .disclaimer-box {
        background-color: #fff8e1;
        border-left: 5px solid #f39c12;
        padding: 0.8rem 1.2rem;
        border-radius: 4px;
        font-size: 0.92rem;
        color: #7d5a00;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #f0f4f8;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1a3a5c;
        border-bottom: 2px solid #3b82d4;
        padding-bottom: 0.3rem;
        margin-bottom: 1rem;
    }
    .prediction-positive {
        background-color: #fde8e8;
        border: 2px solid #e53e3e;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        font-size: 1.1rem;
    }
    .prediction-negative {
        background-color: #e8f5e9;
        border: 2px solid #38a169;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        font-size: 1.1rem;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPER: LOAD RESOURCES (CACHED)
# ─────────────────────────────────────────────
@st.cache_data
def load_dataset():
    if not os.path.exists(DATASET_PATH):
        return None
    return pd.read_csv(DATASET_PATH)

@st.cache_resource
def load_pipeline():
    if not os.path.exists(PIPELINE_PATH):
        return None
    return joblib.load(PIPELINE_PATH)

@st.cache_data
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


df      = load_dataset()
pipeline = load_pipeline()
metrics  = load_metrics()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg",
             width=80, caption="IBM SkillsBuild AI/ML Internship")
    st.markdown("---")
    st.markdown("### 📋 Navigation")
    section = st.radio(
        "Go to section:",
        ["🏠 Project Introduction",
         "📊 Dataset Overview",
         "🤖 Model Performance",
         "🔬 Predict Risk"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Dataset**")
    if df is not None:
        st.markdown(f"- Rows: `{len(df):,}`")
        st.markdown(f"- Features: `{df.shape[1] - 1}`")
    else:
        st.warning("Dataset not found.")

    st.markdown("**Model**")
    if pipeline is not None and metrics is not None:
        st.success(f"✔ Loaded: {metrics.get('best_model','?')}")
    else:
        st.warning("Run `python train_model.py` first.")

    st.markdown("---")
    st.caption("This project is for educational purposes only. Not a medical diagnostic tool.")


# ─────────────────────────────────────────────
# SECTION 1 — PROJECT INTRODUCTION
# ─────────────────────────────────────────────
if "🏠 Project Introduction" in section:
    st.markdown('<p class="main-header">🩺 AI-Based Diabetes Risk Prediction Dashboard</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">IBM SkillsBuild AI/ML Internship — Educational Machine Learning Project</p>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
    ⚠️ <strong>Educational Disclaimer:</strong> This dashboard was built as part of an academic 
    internship project. The predictions made by this tool are <strong>model estimates based on 
    patterns in training data</strong>. They are <strong>not</strong> a medical diagnosis, clinical 
    assessment, or substitute for professional healthcare advice. Do not use this tool to make 
    any health decisions.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<p class="section-title">Project Objective</p>', unsafe_allow_html=True)
        st.write("""
        This project demonstrates how **machine learning** can analyze health-related records 
        and classify them into two categories:

        - **Class 0** — Negative (non-diabetic pattern)  
        - **Class 1** — Positive (diabetic pattern)

        The goal is to compare two supervised learning algorithms and evaluate their ability to 
        identify the positive (minority) class correctly.
        """)

        st.markdown('<p class="section-title">What is Machine Learning?</p>', unsafe_allow_html=True)
        st.write("""
        Machine learning is a branch of artificial intelligence where a computer program 
        **learns patterns from data** without being explicitly told the rules. In this project:
        
        1. A dataset of 100,000 health records is used for training.
        2. The model learns which combination of features is associated with the positive class.
        3. When a new record is provided, the model applies the learned patterns to predict the class.
        """)

    with col2:
        st.markdown('<p class="section-title">Key Features Used</p>', unsafe_allow_html=True)
        features_info = {
            "gender": "Biological sex",
            "age": "Age in years",
            "hypertension": "High blood pressure (0/1)",
            "heart_disease": "Heart disease history (0/1)",
            "smoking_history": "Smoking status",
            "bmi": "Body Mass Index",
            "HbA1c_level": "Average blood sugar (3 months)",
            "blood_glucose_level": "Current blood glucose (mg/dL)",
        }
        for feat, desc in features_info.items():
            st.markdown(f"**`{feat}`** — {desc}")

        st.markdown("---")
        st.markdown("**Target column:** `diabetes`")
        st.markdown("- `0` = Negative class")
        st.markdown("- `1` = Positive class")


# ─────────────────────────────────────────────
# SECTION 2 — DATASET OVERVIEW
# ─────────────────────────────────────────────
elif "📊 Dataset Overview" in section:
    st.markdown('<p class="main-header">📊 Dataset Overview</p>', unsafe_allow_html=True)

    if df is None:
        st.error(f"Dataset not found at `{DATASET_PATH}`. Please place the CSV in the `data/` folder.")
        st.stop()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Features", df.shape[1] - 1)
    col3.metric("Positive Class (1)", f"{(df['diabetes']==1).sum():,}")
    col4.metric("Negative Class (0)", f"{(df['diabetes']==0).sum():,}")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    # ── Class distribution pie chart ──
    with col_left:
        st.markdown('<p class="section-title">Target Class Distribution</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = df["diabetes"].value_counts().sort_index()
        labels = ["Negative (0)\nNon-Diabetic", "Positive (1)\nDiabetic"]
        colors = ["#4a90d9", "#e05c5c"]
        wedges, texts, autotexts = ax.pie(
            counts.values, labels=labels, autopct="%1.1f%%",
            colors=colors, startangle=90,
            textprops={"fontsize": 10},
        )
        for at in autotexts:
            at.set_fontsize(11)
            at.set_fontweight("bold")
        ax.set_title("Class Distribution (100,000 records)", fontsize=11, pad=12)
        st.pyplot(fig)
        plt.close(fig)

    # ── Age distribution by class ──
    with col_right:
        st.markdown('<p class="section-title">Age Distribution by Class</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        df_neg = df[df["diabetes"] == 0]["age"]
        df_pos = df[df["diabetes"] == 1]["age"]
        ax.hist(df_neg, bins=30, alpha=0.6, color="#4a90d9", label="Negative (0)", density=True)
        ax.hist(df_pos, bins=30, alpha=0.7, color="#e05c5c", label="Positive (1)", density=True)
        ax.set_xlabel("Age (years)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title("Age Distribution by Diabetes Class", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    # ── BMI & HbA1c box plots ──
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.markdown('<p class="section-title">BMI by Class</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        data_to_plot = [
            df[df["diabetes"] == 0]["bmi"].dropna().values,
            df[df["diabetes"] == 1]["bmi"].dropna().values,
        ]
        bp = ax.boxplot(data_to_plot, patch_artist=True,
                        tick_labels=["Negative (0)", "Positive (1)"])
        bp["boxes"][0].set_facecolor("#4a90d9")
        bp["boxes"][1].set_facecolor("#e05c5c")
        ax.set_ylabel("BMI", fontsize=10)
        ax.set_title("BMI Distribution by Class", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    with col_r2:
        st.markdown('<p class="section-title">HbA1c Level by Class</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        data_to_plot = [
            df[df["diabetes"] == 0]["HbA1c_level"].dropna().values,
            df[df["diabetes"] == 1]["HbA1c_level"].dropna().values,
        ]
        bp = ax.boxplot(data_to_plot, patch_artist=True,
                        tick_labels=["Negative (0)", "Positive (1)"])
        bp["boxes"][0].set_facecolor("#4a90d9")
        bp["boxes"][1].set_facecolor("#e05c5c")
        ax.set_ylabel("HbA1c Level (%)", fontsize=10)
        ax.set_title("HbA1c Level Distribution by Class", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    # ── Missing values & smoking history ──
    st.markdown("---")
    col_l3, col_r3 = st.columns(2)

    with col_l3:
        st.markdown('<p class="section-title">Missing Values Summary</p>', unsafe_allow_html=True)
        missing = df.isnull().sum().reset_index()
        missing.columns = ["Column", "Missing Count"]
        missing["Missing %"] = (missing["Missing Count"] / len(df) * 100).round(2)
        st.dataframe(missing, use_container_width=True, hide_index=True)
        if missing["Missing Count"].sum() == 0:
            st.success("✔ No missing values found in the dataset.")

    with col_r3:
        st.markdown('<p class="section-title">Smoking History Distribution</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        smoke_counts = df["smoking_history"].value_counts()
        colors_bar = ["#4a90d9", "#e05c5c", "#f39c12", "#27ae60", "#8e44ad", "#16a085"]
        bars = ax.bar(smoke_counts.index, smoke_counts.values,
                      color=colors_bar[:len(smoke_counts)], edgecolor="white")
        ax.set_xlabel("Smoking History", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title("Smoking History Categories", fontsize=11)
        plt.xticks(rotation=30, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 200,
                    f"{int(bar.get_height()):,}",
                    ha="center", va="bottom", fontsize=8)
        st.pyplot(fig)
        plt.close(fig)


# ─────────────────────────────────────────────
# SECTION 3 — MODEL PERFORMANCE
# ─────────────────────────────────────────────
elif "🤖 Model Performance" in section:
    st.markdown('<p class="main-header">🤖 Model Performance</p>', unsafe_allow_html=True)

    if metrics is None or pipeline is None:
        st.error("Model files not found.")
        st.info(
            "**To generate the model files, run the training script:**\n\n"
            "```\npython train_model.py\n```\n\n"
            "This will create `models/best_pipeline.joblib` and `models/model_metrics.json`."
        )
        st.stop()

    best_name = metrics.get("best_model", "Unknown")
    criterion = metrics.get("selection_criterion", "")
    models_data = metrics.get("models", {})

    st.info(f"✔ **Best model selected:** {best_name}  |  **Selection criterion:** {criterion}")

    # ── Summary comparison table ──
    st.markdown('<p class="section-title">Model Comparison (Test Set Results)</p>',
                unsafe_allow_html=True)
    rows = []
    for mname, mdata in models_data.items():
        row = {
            "Model": mname,
            "Accuracy": f"{mdata['accuracy']:.4f}",
            "Precision": f"{mdata['precision']:.4f}",
            "Recall": f"{mdata['recall']:.4f}",
            "F1-Score": f"{mdata['f1_score']:.4f}",
            "ROC-AUC": f"{mdata['roc_auc']:.4f}" if mdata.get("roc_auc") else "N/A",
        }
        rows.append(row)

    comparison_df = pd.DataFrame(rows)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.markdown("**Metric Explanations:**")
    st.markdown("""
    | Metric | What it measures |
    |--------|-----------------|
    | **Accuracy** | % of all records classified correctly |
    | **Precision** | Of all predicted positives, how many are truly positive |
    | **Recall** | Of all actual positives, how many did the model catch |
    | **F1-Score** | Harmonic mean of Precision and Recall (balanced view) |
    | **ROC-AUC** | Model's ability to distinguish between classes (1.0 = perfect) |
    """)

    st.markdown("---")

    # ── Per-model confusion matrices ──
    st.markdown('<p class="section-title">Confusion Matrices</p>', unsafe_allow_html=True)
    st.caption("A confusion matrix shows how many records were correctly and incorrectly classified.")

    n_models = len(models_data)
    cols = st.columns(n_models)

    for idx, (mname, mdata) in enumerate(models_data.items()):
        cm = np.array(mdata["confusion_matrix"])
        with cols[idx]:
            fig, ax = plt.subplots(figsize=(4.5, 3.8))
            im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
            plt.colorbar(im, ax=ax)
            class_labels = ["Negative (0)", "Positive (1)"]
            tick_marks = np.arange(2)
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(class_labels, fontsize=8, rotation=15)
            ax.set_yticklabels(class_labels, fontsize=8)
            thresh = cm.max() / 2.0
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{cm[i, j]:,}",
                            ha="center", va="center", fontsize=12,
                            color="white" if cm[i, j] > thresh else "black",
                            fontweight="bold")
            ax.set_xlabel("Predicted Label", fontsize=9)
            ax.set_ylabel("True Label", fontsize=9)
            star = " ★ SELECTED" if mname == best_name else ""
            ax.set_title(f"{mname}{star}", fontsize=10, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
            st.markdown(f"""
            - **TN** (True Negative): {tn:,}
            - **FP** (False Positive): {fp:,}
            - **FN** (False Negative): {fn:,}
            - **TP** (True Positive): {tp:,}
            """)

    # ── Bar chart comparison ──
    st.markdown("---")
    st.markdown('<p class="section-title">Metric Comparison Chart</p>', unsafe_allow_html=True)

    metric_keys   = ["accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    model_names   = list(models_data.keys())
    x = np.arange(len(metric_labels))
    width = 0.35
    bar_colors = ["#4a90d9", "#e05c5c"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (mname, mdata) in enumerate(models_data.items()):
        vals = [mdata[k] for k in metric_keys]
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width,
                      label=mname, color=bar_colors[i % len(bar_colors)],
                      edgecolor="white", alpha=0.9)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Model Performance Comparison on Test Set", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ─────────────────────────────────────────────
# SECTION 4 — PREDICTION INTERFACE
# ─────────────────────────────────────────────
elif "🔬 Predict Risk" in section:
    st.markdown('<p class="main-header">🔬 Predict Diabetes Risk Class</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
    ⚠️ <strong>Reminder:</strong> The result below is a <strong>model estimate</strong> based on 
    statistical patterns in training data. It is <strong>not</strong> a medical diagnosis. 
    Please consult a qualified healthcare professional for any health concerns.
    </div>
    """, unsafe_allow_html=True)

    if pipeline is None:
        st.error("Model pipeline not found.")
        st.info(
            "**Please run the training script first:**\n\n"
            "```\npython train_model.py\n```\n\n"
            "Once complete, refresh this page."
        )
        st.stop()

    if metrics is not None:
        best_name = metrics.get("best_model", "")
        st.caption(f"Using model: **{best_name}**")

    st.markdown('<p class="section-title">Enter Health Record Details</p>', unsafe_allow_html=True)

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox(
                "Gender",
                ["Female", "Male", "Other"],
                help="Biological sex as recorded in the dataset."
            )
            age = st.number_input(
                "Age (years)",
                min_value=0.0, max_value=120.0, value=45.0, step=1.0,
                help="Age between 0 and 120 years."
            )
            hypertension = st.selectbox(
                "Hypertension (High Blood Pressure)",
                [0, 1],
                format_func=lambda x: "Yes (1)" if x == 1 else "No (0)",
                help="Does the person have high blood pressure?"
            )
            heart_disease = st.selectbox(
                "Heart Disease",
                [0, 1],
                format_func=lambda x: "Yes (1)" if x == 1 else "No (0)",
                help="Does the person have a heart disease history?"
            )

        with col2:
            smoking_history = st.selectbox(
                "Smoking History",
                ["never", "No Info", "current", "former", "ever", "not current"],
                help="Select the person's smoking history."
            )
            bmi = st.number_input(
                "BMI (Body Mass Index)",
                min_value=10.0, max_value=60.0, value=27.3, step=0.1,
                help="Normal range is roughly 18.5–24.9."
            )
            hba1c = st.number_input(
                "HbA1c Level (%)",
                min_value=3.5, max_value=15.0, value=5.7, step=0.1,
                help="Average blood sugar over the past 2–3 months. Normal < 5.7%."
            )
            glucose = st.number_input(
                "Blood Glucose Level (mg/dL)",
                min_value=50, max_value=500, value=140, step=1,
                help="Fasting blood glucose. Normal < 100 mg/dL."
            )

        submitted = st.form_submit_button("🔍 Predict Class", use_container_width=True)

    if submitted:
        # ── Input validation ──
        errors = []
        if age < 0 or age > 120:
            errors.append("Age must be between 0 and 120.")
        if bmi < 10 or bmi > 60:
            errors.append("BMI must be between 10 and 60.")
        if hba1c < 3.5 or hba1c > 15:
            errors.append("HbA1c must be between 3.5 and 15.")
        if glucose < 50 or glucose > 500:
            errors.append("Blood glucose must be between 50 and 500.")

        if errors:
            for err in errors:
                st.error(f"⚠ {err}")
        else:
            # ── Build input DataFrame with exact feature names ──
            input_data = pd.DataFrame([{
                "gender":              gender,
                "age":                 float(age),
                "hypertension":        int(hypertension),
                "heart_disease":       int(heart_disease),
                "smoking_history":     smoking_history,
                "bmi":                 float(bmi),
                "HbA1c_level":         float(hba1c),
                "blood_glucose_level": int(glucose),
            }])

            try:
                prediction = pipeline.predict(input_data)[0]
                has_proba  = hasattr(pipeline.named_steps["classifier"], "predict_proba")

                if has_proba:
                    proba    = pipeline.predict_proba(input_data)[0]
                    prob_pos = proba[1]

                st.markdown("---")
                st.markdown("### Prediction Result")

                if prediction == 1:
                    st.markdown("""
                    <div class="prediction-positive">
                    🔴 <strong>Predicted Class: 1 — Positive (Diabetic Pattern)</strong><br>
                    The model identified patterns in this record that match the positive class in the training data.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="prediction-negative">
                    🟢 <strong>Predicted Class: 0 — Negative (Non-Diabetic Pattern)</strong><br>
                    The model identified patterns in this record that match the negative class in the training data.
                    </div>
                    """, unsafe_allow_html=True)

                if has_proba:
                    st.markdown(f"""
                    **Estimated probability for Positive class (diabetes = 1):** `{prob_pos:.4f}` 
                    ({prob_pos*100:.1f}%)
                    """)

                    # Probability gauge bar
                    fig, ax = plt.subplots(figsize=(6, 1.2))
                    ax.barh(0, prob_pos, color="#e05c5c" if prob_pos >= 0.5 else "#4a90d9",
                            height=0.5)
                    ax.barh(0, 1 - prob_pos, left=prob_pos,
                            color="#e0e0e0", height=0.5)
                    ax.set_xlim(0, 1)
                    ax.set_yticks([])
                    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9)
                    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=1)
                    ax.set_title("Estimated Probability for Positive Class", fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                st.markdown("""
                > **Note:** This is a machine learning estimate based on statistical patterns  
                > in the training dataset. It is **not** a confirmed medical diagnosis.  
                > Always consult a licensed healthcare professional for medical advice.
                """)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info("Please verify that the model pipeline is compatible. Re-run `python train_model.py` if needed.")
