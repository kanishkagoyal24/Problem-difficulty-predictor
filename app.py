import streamlit as st
import numpy as np
import pandas as pd
import re
import joblib
from scipy.sparse import hstack

# ---------------------------------------------------------
# 1. LOAD ASSETS
# ---------------------------------------------------------
@st.cache_resource
def load_assets():
    clf_model = joblib.load("clf_model.pkl")
    reg_model = joblib.load("reg_model.pkl")
    scaler = joblib.load("scaler.pkl")
    le = joblib.load("label_encoder.pkl")
    tfidf = joblib.load("tfidf.pkl")
    return clf_model, reg_model, scaler, le, tfidf

clf_model, reg_model, scaler, le, tfidf = load_assets()

# ---------------------------------------------------------
# 2. PREPROCESSING (UNCHANGED)
# ---------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\t", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def max_numeric_value(text):
    nums = re.findall(r"\d+", text)
    return max(map(int, nums)) if nums else 0

def extract_features(full_text):
    clean = clean_text(full_text)

    X_tfidf = tfidf.transform([clean])

    text_length = len(clean)
    num_words = len(clean.split())
    max_constraint = max_numeric_value(full_text)
    log_max_constraint = np.log1p(max_constraint)
    constraint_density = log_max_constraint / (num_words + 1)
    num_count = len(re.findall(r"\d+", clean))

    X_extra = np.array([[text_length, num_words, constraint_density, num_count]])

    X = hstack([X_tfidf, X_extra])
    X = scaler.transform(X)

    return X

# ---------------------------------------------------------
# 3. PAGE CONFIG + HEADER
# ---------------------------------------------------------
st.set_page_config(
    page_title="ACM Difficulty Predictor",
    page_icon="🧩",
    layout="centered"
)

st.markdown(
    """
    <h1 style="text-align:center;">🧩 ACM Problem Difficulty Predictor</h1>
    <p style="text-align:center; color:gray;">
    Predict difficulty level and score for competitive programming problems
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------------------------------------------------------
# 4. INPUT SECTION (SEPARATED)
# ---------------------------------------------------------
st.subheader("📘 Problem Statement")

desc = st.text_area(
    "📝 Problem Description",
    height=200,
    placeholder="Describe the problem statement here..."
)

inp = st.text_area(
    "📥 Input Description",
    height=120,
    placeholder="Describe the input format..."
)

out = st.text_area(
    "📤 Output Description",
    height=120,
    placeholder="Describe the expected output..."
)

full_text = f"{desc}\n{inp}\n{out}"

st.divider()

# ---------------------------------------------------------
# 5. ANALYZE BUTTON
# ---------------------------------------------------------
analyze = st.button("🚀 Analyze Difficulty", use_container_width=True)

if analyze:
    if not desc.strip():
        st.warning("Please enter at least the problem description.")
    else:
        with st.spinner("Analyzing problem complexity..."):
            X_input = extract_features(full_text)

            pred_class_idx = clf_model.predict(X_input)[0]
            pred_class = le.inverse_transform([pred_class_idx])[0]

            pred_score = float(reg_model.predict(X_input)[0])
            pred_score = max(0.0, min(10.0, pred_score))

        st.divider()

        # -------------------------------------------------
        # 6. RESULTS
        # -------------------------------------------------
        st.subheader("📊 Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🎯 Difficulty Class")
            color = (
                "green" if pred_class.lower() == "easy"
                else "orange" if pred_class.lower() == "medium"
                else "red"
            )
            st.markdown(f"<h2 style='color:{color}'>{pred_class}</h2>", unsafe_allow_html=True)

        with col2:
            st.markdown("### 📈 Difficulty Score")
            st.metric("Score (0–10)", f"{pred_score:.2f}")
            st.progress(pred_score / 10)

        st.success("✅ Analysis complete!")
