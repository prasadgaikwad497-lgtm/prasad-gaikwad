import os
import pickle
import time
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_confetti import confetti

# ---------------------------------------------------------
# Page Configuration & Custom CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Grade Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS for attractive UI styling
st.markdown(
    """
    <style>
    /* Main title formatting */
    .main-title {
        font-size: 2.6rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    /* Input section cards */
    .css-1r6slb0, .stForm {
        background-color: #F9FAFB;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
        margin-top: 20px;
    }
    .result-card h2 {
        color: #FFFFFF !important;
        margin: 0;
        font-size: 2rem;
    }
    .result-card p {
        font-size: 1.1rem;
        margin-top: 5px;
        opacity: 0.9;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Model Loading
# ---------------------------------------------------------
MODEL_FILE = "model.pkl"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        return None
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    return model


model = load_model()

# ---------------------------------------------------------
# Header Section
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">🎓 Academic Performance Predictor</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">Enter student subject marks to calculate predictions using your trained KNN model.</div>',
    unsafe_allow_html=True,
)

if model is None:
    st.error(
        f"⚠️ Could not find `{MODEL_FILE}` in the current directory. Please make sure the file is present alongside `app.py`."
    )
    st.stop()

# ---------------------------------------------------------
# Feature Input Sidebar & Form
# ---------------------------------------------------------
st.sidebar.header("📊 Options & Info")
st.sidebar.info(
    "This app uses a **K-Neighbors Classifier** trained on student academic records."
)

st.subheader("📝 Enter Student Marks")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        unnamed_0 = st.number_input(
            "Student ID / Index (Unnamed: 0)",
            min_value=0,
            value=1,
            step=1,
            help="Index or ID entry from the dataset",
        )
        hindi = st.number_input(
            "Hindi", min_value=0, max_value=100, value=75, step=1
        )
        english = st.number_input(
            "English", min_value=0, max_value=100, value=80, step=1
        )
        science = st.number_input(
            "Science", min_value=0, max_value=100, value=85, step=1
        )

    with col2:
        maths = st.number_input(
            "Maths", min_value=0, max_value=100, value=90, step=1
        )
        history = st.number_input(
            "History", min_value=0, max_value=100, value=70, step=1
        )
        geography = st.number_input(
            "Geography", min_value=0, max_value=100, value=78, step=1
        )
        div = st.number_input(
            "Division Code (Div)",
            min_value=0,
            value=1,
            step=1,
            help="Division category code",
        )

    # Automatic calculation of total marks
    total_calculated = hindi + english + science + maths + history + geography
    st.markdown(f"**Calculated Total Marks:** `{total_calculated}`")

    # Form submission button
    submit_button = st.form_submit_button(
        "🚀 Predict Result", use_container_width=True
    )

# ---------------------------------------------------------
# Prediction & Visual Effects
# ---------------------------------------------------------
if submit_button:
    # Prepare features array in the exact order expected by the model:
    # ['Unnamed: 0', 'Hindi', 'English', 'Science', 'Maths', 'History', 'Geograpgy', 'Total', 'Div']
    input_features = np.array(
        [
            [
                unnamed_0,
                hindi,
                english,
                science,
                maths,
                history,
                geography,
                total_calculated,
                div,
            ]
        ]
    )

    # Add interactive loading effect
    with st.spinner("Analyzing marks and computing prediction..."):
        time.sleep(0.8)
        prediction = model.predict(input_features)[0]

    # Trigger celebration effect (Confetti + Balloons)
    confetti()
    st.balloons()

    # Display Result Card
    st.markdown(
        f"""
        <div class="result-card">
            <p>Prediction Outcome</p>
            <h2>Result / Grade Class: {prediction}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Feature breakdown expandable view
    with st.expander("🔍 View Input Summary"):
        summary_df = pd.DataFrame(
            input_features,
            columns=[
                "Unnamed: 0",
                "Hindi",
                "English",
                "Science",
                "Maths",
                "History",
                "Geography",
                "Total",
                "Div",
            ],
        )
        st.dataframe(summary_df, use_container_width=True)
