"""
Road Accident Severity Prediction — Streamlit Application
==========================================================
An interactive web app that provides:
  1. HOME — Project overview
  2. PREDICT SEVERITY — User-friendly prediction form
  3. MODEL PERFORMANCE — Metrics, reports, confusion matrices
  4. DATASET INSIGHTS — EDA charts generated during training

Launch:
    streamlit run app.py
"""

import os
import json

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "road_accident_severity_model.joblib")
META_PATH = os.path.join(PROJECT_DIR, "models", "model_metadata.json")
REPORTS_DIR = os.path.join(PROJECT_DIR, "reports")
TRAIN_CSV = os.path.join(PROJECT_DIR, "train.csv")

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Road Accident Severity Prediction",
    page_icon="🚗",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    """Load the trained pipeline from disk."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    """Load model metadata from disk."""
    if not os.path.exists(META_PATH):
        return None
    with open(META_PATH, "r") as f:
        return json.load(f)


@st.cache_data
def load_train_data():
    """Load training CSV to extract unique values for selectboxes."""
    if not os.path.exists(TRAIN_CSV):
        return None
    df = pd.read_csv(TRAIN_CSV)
    # Clean string 'na' values
    df = df.replace(["na", "Na", "NA"], np.nan)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df


def get_unique_values(df, col):
    """Get sorted unique non-null values for a categorical column."""
    if df is None or col not in df.columns:
        return ["Unknown"]
    vals = df[col].dropna().unique().tolist()
    vals = [v for v in vals if isinstance(v, str) and v.strip()]
    return sorted(set(vals)) if vals else ["Unknown"]


# =========================================================================
# SIDEBAR NAVIGATION
# =========================================================================
st.sidebar.title("🚗 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Home", "🔮 Predict Severity", "📊 Model Performance", "📈 Dataset Insights"],
)

# =========================================================================
# PAGE 1: HOME
# =========================================================================
if page == "🏠 Home":
    st.title("🚗 Road Accident Severity Prediction")
    st.markdown("### Using Machine Learning")

    st.markdown("---")

    st.markdown("""
    #### 🎯 Project Objective

    This project uses **machine learning** to predict the **severity of road traffic accidents**
    based on various factors such as weather conditions, road surface, light conditions,
    vehicle type, driver characteristics, and more.

    Predicting accident severity can help:
    - **Traffic authorities** allocate emergency resources more effectively
    - **Urban planners** identify high-risk areas
    - **Insurance companies** assess risk factors
    - **Researchers** understand accident dynamics
    """)

    st.markdown("#### 🔄 ML Workflow")
    st.markdown("""
    ```
    Raw Data → Data Cleaning → Feature Engineering → Train/Validation Split
         → Model Training (LR, DT, RF) → Model Evaluation (Macro F1)
         → Best Model Selection → Save Pipeline → Streamlit Deployment
    ```
    """)

    st.markdown("#### 📊 Severity Classes")
    st.markdown("""
    The model predicts one of three accident severity levels:
    - 🟢 **Slight Injury** — Minor injuries
    - 🟡 **Serious Injury** — Significant injuries requiring medical attention
    - 🔴 **Fatal Injury** — Loss of life
    """)

    st.info(
        "⚠️ **Disclaimer**: This project is built for **educational purposes only**. "
        "It should not be used for real-world accident severity assessment or decision-making."
    )


# =========================================================================
# PAGE 2: PREDICT SEVERITY
# =========================================================================
elif page == "🔮 Predict Severity":
    st.title("🔮 Predict Accident Severity")
    st.markdown("Fill in the accident details below and click **Predict** to see the result.")
    st.markdown("---")

    # Load resources
    pipeline = load_pipeline()
    metadata = load_metadata()
    train_df = load_train_data()

    if pipeline is None or metadata is None:
        st.error(
            "❌ **Model not found!** Please run `python src/train_model.py` first "
            "to train and save the model."
        )
    else:
        cat_features = metadata.get("categorical_features", [])
        num_features = metadata.get("numerical_features", [])

        st.markdown("#### Categorical Features")
        input_data = {}

        # Create columns for a cleaner layout
        cols = st.columns(3)
        for i, feat in enumerate(cat_features):
            with cols[i % 3]:
                options = get_unique_values(train_df, feat)
                input_data[feat] = st.selectbox(
                    feat.replace("_", " ").title(),
                    options=options,
                    key=f"cat_{feat}",
                )

        if num_features:
            st.markdown("#### Numerical Features")
            num_cols = st.columns(3)
            for i, feat in enumerate(num_features):
                with num_cols[i % 3]:
                    if train_df is not None and feat in train_df.columns:
                        col_data = pd.to_numeric(train_df[feat], errors="coerce").dropna()
                        min_val = int(col_data.min())
                        max_val = int(col_data.max())
                        default_val = int(col_data.median())
                    else:
                        min_val, max_val, default_val = 0, 100, 1

                    input_data[feat] = st.number_input(
                        feat.replace("_", " ").title(),
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        key=f"num_{feat}",
                    )

        st.markdown("---")

        if st.button("🚀 Predict Accident Severity", type="primary", use_container_width=True):
            try:
                # Build a single-row DataFrame
                input_df = pd.DataFrame([input_data])

                # Predict
                prediction = pipeline.predict(input_df)[0]

                # Display result prominently
                severity_colors = {
                    "Slight Injury": "🟢",
                    "Serious Injury": "🟡",
                    "Fatal Injury": "🔴",
                }
                icon = severity_colors.get(prediction, "⚪")

                st.markdown("### Prediction Result")
                st.success(f"{icon} **Predicted Severity: {prediction}**")

                # Probabilities
                if hasattr(pipeline, "predict_proba"):
                    proba = pipeline.predict_proba(input_df)[0]
                    class_labels = metadata.get("class_labels", [])

                    st.markdown("#### Class Probabilities")
                    prob_df = pd.DataFrame({
                        "Severity Class": class_labels,
                        "Probability": proba,
                    })
                    prob_df["Probability (%)"] = (prob_df["Probability"] * 100).round(2)

                    # Display as a nice table
                    st.dataframe(
                        prob_df[["Severity Class", "Probability (%)"]],
                        use_container_width=True,
                        hide_index=True,
                    )

                    # Bar chart
                    fig, ax = plt.subplots(figsize=(8, 4))
                    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
                    bars = ax.barh(class_labels, proba, color=colors[:len(class_labels)])
                    ax.set_xlabel("Probability")
                    ax.set_title("Prediction Probability Distribution")
                    ax.set_xlim(0, 1)
                    for bar, p in zip(bars, proba):
                        ax.text(
                            bar.get_width() + 0.02,
                            bar.get_y() + bar.get_height() / 2,
                            f"{p:.2%}",
                            va="center",
                            fontweight="bold",
                        )
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")


# =========================================================================
# PAGE 3: MODEL PERFORMANCE
# =========================================================================
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")
    st.markdown("---")

    metadata = load_metadata()

    if metadata is None:
        st.error(
            "❌ **Model metadata not found!** Please run `python src/train_model.py` first."
        )
    else:
        # Best model info
        st.markdown(f"### 🏆 Selected Model: **{metadata['best_model']}**")
        st.markdown(f"**Target column:** `{metadata['target_column']}`")
        st.markdown(f"**Class labels:** {', '.join(metadata['class_labels'])}")

        st.markdown("---")

        # Validation metrics comparison table
        st.markdown("### Validation Metrics Comparison")

        val_metrics = metadata.get("validation_metrics", {})
        if val_metrics:
            metrics_df = pd.DataFrame(val_metrics).T
            metrics_df.index.name = "Model"
            st.dataframe(metrics_df.style.highlight_max(axis=0, color="#a8e6cf"), use_container_width=True)

        # Classification reports
        st.markdown("### Classification Reports")
        reports = metadata.get("classification_reports", {})
        for model_name, report in reports.items():
            with st.expander(f"📋 {model_name}"):
                st.code(report)

        # Confusion matrices
        st.markdown("### Confusion Matrices")
        cm_files = {
            "Logistic Regression": "cm_logistic_regression.png",
            "Decision Tree": "cm_decision_tree.png",
            "Random Forest": "cm_random_forest.png",
        }

        cm_cols = st.columns(3)
        for i, (name, filename) in enumerate(cm_files.items()):
            path = os.path.join(REPORTS_DIR, filename)
            if os.path.exists(path):
                with cm_cols[i]:
                    st.image(path, caption=name, use_container_width=True)

        # Test metrics (if available)
        test_metrics = metadata.get("test_metrics")
        if test_metrics:
            st.markdown("---")
            st.markdown("### 🧪 Final Test Set Evaluation")
            test_display = {k: v for k, v in test_metrics.items() if k != "classification_report"}
            st.dataframe(pd.DataFrame([test_display]), use_container_width=True)
            if "classification_report" in test_metrics:
                with st.expander("📋 Test Classification Report"):
                    st.code(test_metrics["classification_report"])
        else:
            st.markdown("---")
            st.info(
                "ℹ️ The Kaggle test set does **not contain target labels**. "
                "Test metrics are not available. Model was selected based on "
                "validation set macro F1 score."
            )


# =========================================================================
# PAGE 4: DATASET INSIGHTS
# =========================================================================
elif page == "📈 Dataset Insights":
    st.title("📈 Dataset Insights — EDA Charts")
    st.markdown("---")

    # Define chart files and titles in order
    charts = [
        ("01_target_distribution.png", "Accident Severity Distribution"),
        ("02_severity_vs_weather.png", "Severity vs Weather Conditions"),
        ("03_severity_vs_road_surface.png", "Severity vs Road Surface Conditions"),
        ("04_severity_vs_light.png", "Severity vs Light Conditions"),
        ("05_severity_vs_road_type.png", "Severity vs Road Surface Type"),
        ("06_severity_vs_collision_type.png", "Severity vs Type of Collision"),
        ("07_severity_vs_day_of_week.png", "Severity vs Day of Week"),
        ("08_severity_vs_cause.png", "Severity vs Cause of Accident"),
        ("09_numerical_distributions.png", "Numerical Feature Distributions"),
    ]

    found_any = False
    for filename, title in charts:
        path = os.path.join(REPORTS_DIR, filename)
        if os.path.exists(path):
            found_any = True
            st.markdown(f"#### {title}")
            st.image(path, use_container_width=True)
            st.markdown("---")

    if not found_any:
        st.warning(
            "⚠️ No EDA charts found. Please run `python src/train_model.py` first "
            "to generate the charts."
        )
