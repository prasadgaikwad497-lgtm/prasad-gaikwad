import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Result Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------
# Custom Styling (Pure White Background & Simple Cards)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Force main app background to solid white */
    .stApp {
        background-color: #FFFFFF !important;
    }

    /* Limit container width for a clean vertical stack */
    .block-container {
        max-width: 650px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Minimalist Header */
    .main-header {
        text-align: center;
        margin-bottom: 25px;
    }
    .main-header h1 {
        font-size: 2.2rem;
        color: #1E293B;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .main-header p {
        color: #64748B;
        font-size: 1rem;
    }

    /* Clean Form Styling */
    div[data-testid="stForm"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }

    /* Total Marks Box */
    .total-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 12px;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
        color: #334155;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    /* Result Cards */
    .pass-card {
        background-color: #10B981;
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25);
    }
    .fail-card {
        background-color: #EF4444;
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.25);
    }
    .result-card h2 {
        color: white !important;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
    }
    .result-card p {
        margin: 0;
        font-size: 1rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Load Trained Model
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
# Application Header
# ---------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>🎓 Student Result Predictor</h1>
        <p>Enter individual subject scores to predict Pass or Fail</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is None:
    st.error(
        f"⚠️ `{MODEL_FILE}` was not found in the project root folder. Please ensure your model file is uploaded."
    )
    st.stop()

# ---------------------------------------------------------
# Vertical Form Fields
# ---------------------------------------------------------
with st.form("prediction_form"):
    unnamed_0 = st.number_input(
        "Student ID / Index (Unnamed: 0)",
        min_value=0,
        value=1,
        step=1,
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
    )

    # Automatic Total Score Calculation
    total_calculated = hindi + english + science + maths + history + geography

    st.markdown(
        f"""
        <div class="total-box">
            Calculated Total Score: <b>{total_calculated} / 600</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    submit_button = st.form_submit_button(
        "Predict Result", use_container_width=True
    )

# ---------------------------------------------------------
# Prediction & Pass/Fail Output Logic
# ---------------------------------------------------------
if submit_button:
    # Feature array in exact trained sequence:
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

    # Get model prediction (e.g., 1 or 0)
    raw_prediction = model.predict(input_data)[0]

    # Convert 1/0 or string equivalent into PASS / FAIL display
    if str(raw_prediction).strip() in ["1", "1.0", "Pass", "PASS"]:
        st.balloons()
        st.markdown(
            """
            <div class="pass-card result-card">
                <p>Final Outcome</p>
                <h2>PASS 🎉</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="fail-card result-card">
                <p>Final Outcome</p>
                <h2>FAIL ❌</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Summary View
    with st.expander("Show Submitted Details"):
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
