import os
import pickle
import time
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Grade Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom Styling (Vertical Layout, Modern Color Palette & Shadows)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global background accent */
    .stApp {
        background-color: #F8FAFC;
    }

    /* Container Box Styling */
    .block-container {
        max-width: 750px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Header styling */
    .app-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: #FFFFFF;
        padding: 28px 20px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
        margin-bottom: 25px;
    }
    .app-header h1 {
        font-size: 2.2rem !important;
        font-weight: 700;
        margin: 0;
        color: #F8FAFC !important;
    }
    .app-header p {
        font-size: 1rem;
        color: #94A3B8;
        margin-top: 8px;
        margin-bottom: 0;
    }

    /* Vertical Form Card Styling */
    div[data-testid="stForm"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 28px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05);
    }

    /* Input Field Labels */
    .stNumberInput label {
        font-weight: 600 !important;
        color: #334155 !important;
    }

    /* Calculated Total Box */
    .total-box {
        background-color: #F1F5F9;
        border-left: 4px solid #0EA5E9;
        padding: 14px 18px;
        border-radius: 8px;
        font-weight: 600;
        color: #0F172A;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    /* Result Banner */
    .result-card {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%);
        color: #FFFFFF;
        padding: 28px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 12px 24px -4px rgba(2, 132, 199, 0.35);
        margin-top: 25px;
        margin-bottom: 25px;
    }
    .result-card h2 {
        color: #FFFFFF !important;
        margin: 8px 0 0 0;
        font-size: 2.2rem;
        font-weight: 800;
    }
    .result-card p {
        font-size: 1.05rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #BAE6FD;
        margin: 0;
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
        return pickle.load(f)


model = load_model()

# ---------------------------------------------------------
# Header Section
# ---------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>🎓 Academic Performance Predictor</h1>
        <p>Enter individual subject scores below to evaluate academic output</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is None:
    st.error(
        f"⚠️ File `{MODEL_FILE}` was not found in the project root directory. Please verify that the trained model file is uploaded."
    )
    st.stop()

# ---------------------------------------------------------
# Vertical Form Layout
# ---------------------------------------------------------
with st.form("grade_prediction_form"):
    st.subheader("📋 Student Parameters & Subject Marks")

    unnamed_0 = st.number_input(
        "Student ID / Index (Unnamed: 0)",
        min_value=0,
        value=1,
        step=1,
        help="Identifier index corresponding to the dataset format",
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
        help="Division code identifier",
    )

    # Dynamic total calculation
    total_calculated = hindi + english + science + maths + history + geography

    st.markdown(
        f"""
        <div class="total-box">
            📊 Calculated Total Marks: <span style="color: #0EA5E9;">{total_calculated} / 600</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    submit_button = st.form_submit_button(
        "🚀 Compute Prediction", use_container_width=True
    )

# ---------------------------------------------------------
# Prediction Execution
# ---------------------------------------------------------
if submit_button:
    # Construct exact feature order:
    # ['Unnamed: 0', 'Hindi', 'English', 'Science', 'Maths', 'History', 'Geograpgy', 'Total', 'Div']
    input_data = np.array(
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

    with st.spinner("Processing features..."):
        time.sleep(0.5)
        prediction = model.predict(input_data)[0]

    # Native animations replacing confetti dependency
    st.balloons()

    # Result Display Card
    st.markdown(
        f"""
        <div class="result-card">
            <p>Predicted Performance Grade</p>
            <h2>{prediction}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Input Breakdown Expander
    with st.expander("📄 Review Submitted Input Summary"):
        summary_df = pd.DataFrame(
            input_data,
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
