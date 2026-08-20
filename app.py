import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_artifacts():
    model = joblib.load("loan_apro.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")

    return model, scaler, columns


model, scaler, columns = load_artifacts()

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

.stApp {
    background: #f5f7fb;
    color: #000000;
}

.stApp,
.stApp p,
.stApp label,
.stApp span,
.stApp div,
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6 {
    color: #000000;
}

/* Main container */
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Hero */
.hero {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    padding: 35px;
    border-radius: 20px;
    color: #000000;
    margin-bottom: 30px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 8px;
}

.hero p {
    font-size: 17px;
    color: #000000;
}

/* Section */
.section-title {
    font-size: 24px;
    font-weight: 700;
    color: #111827;
    margin-top: 20px;
    margin-bottom: 15px;
}

/* Form labels */
div[data-testid="stWidgetLabel"] label,
div[data-testid="stWidgetLabel"] p {
    color: #000000;
}

/* Result card */
.approved {
    background: #dcfce7;
    border: 1px solid #86efac;
    padding: 25px;
    border-radius: 16px;
    margin-top: 25px;
}

.rejected {
    background: #fee2e2;
    border: 1px solid #fca5a5;
    padding: 25px;
    border-radius: 16px;
    margin-top: 25px;
}

.result-title {
    font-size: 28px;
    font-weight: 700;
}

.result-text {
    font-size: 16px;
    margin-top: 8px;
}

/* Button */
.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 50px;
    font-size: 17px;
    font-weight: 600;
}

/* Footer */
.footer {
    text-align: center;
    color: #000000;
    padding-top: 40px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------

st.markdown("""
<div class="hero">

<h1>🏦 Loan Approval Predictor</h1>

<p>
AI-powered loan eligibility prediction using Machine Learning.
Enter applicant information to estimate whether the loan
application is likely to be approved.
</p>

</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 🏦 Loan Predictor")

    st.markdown("""
    ### About

    This application uses a **Logistic Regression**
    machine learning model to predict loan approval.

    ### Model Pipeline

    **Input → Encoding → Scaling → Prediction**

    ### Model

    Logistic Regression

    ### Disclaimer

    This prediction is for educational purposes only
    and should not be considered an actual lending decision.
    """)

# --------------------------------------------------
# APPLICANT INFORMATION
# --------------------------------------------------

st.markdown(
    '<div class="section-title">👤 Applicant Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    married = st.selectbox(
        "Married",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["0", "1", "2", "3+"]
    )

with col2:

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["No", "Yes"]
    )

    credit_history = st.selectbox(
        "Credit History",
        ["Good", "No Credit History"]
    )

with col3:

    property_area = st.selectbox(
        "Property Area",
        ["Urban", "Semiurban", "Rural"]
    )

# --------------------------------------------------
# FINANCIAL INFORMATION
# --------------------------------------------------

st.markdown(
    '<div class="section-title">💰 Financial Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0.0,
        value=5000.0,
        step=500.0
    )

with col2:

    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0.0,
        value=0.0,
        step=500.0
    )

with col3:

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=1.0,
        value=150.0,
        step=10.0
    )

col1, col2 = st.columns(2)

with col1:

    loan_term = st.selectbox(
        "Loan Term (months)",
        [360, 180, 120, 84, 60, 48, 36, 24, 12]
    )

with col2:

    st.markdown("###")
    st.info(
        "💡 Enter your financial details as accurately as possible."
    )

# --------------------------------------------------
# ENCODING
# --------------------------------------------------

gender_map = {
    "Female": 0,
    "Male": 1
}

married_map = {
    "No": 0,
    "Yes": 1
}

dependents_map = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3+": 3
}

education_map = {
    "Graduate": 0,
    "Not Graduate": 1
}

self_employed_map = {
    "No": 0,
    "Yes": 1
}

credit_history_map = {
    "No Credit History": 0,
    "Good": 1
}

property_area_map = {
    "Rural": 0,
    "Semiurban": 1,
    "Urban": 2
}

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)

predict_button = st.button(
    "🔍 Predict Loan Approval"
)

if predict_button:

    # Create input dataframe
    input_data = pd.DataFrame([{
        "Gender": gender_map[gender],
        "Married": married_map[married],
        "Dependents": dependents_map[dependents],
        "Education": education_map[education],
        "Self_Employed": self_employed_map[self_employed],
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history_map[credit_history],
        "Property_Area": property_area_map[property_area]
    }])

    # Make sure feature order matches training
    input_data = input_data[columns]

    # Scale
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Probability
    probability = model.predict_proba(input_scaled)[0]

    approval_probability = probability[1] * 100

    # --------------------------------------------------
    # RESULT
    # --------------------------------------------------

    if prediction == 1:

        st.markdown(f"""
        <div class="approved">

        <div class="result-title">
        ✅ Loan Likely to be Approved
        </div>

        <div class="result-text">
        Based on the information provided, the model predicts
        that the loan application is likely to be approved.
        </div>

        <br>

        <b>Model Confidence: {approval_probability:.2f}%</b>

        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="rejected">

        <div class="result-title">
        ❌ Loan Likely Not to be Approved
        </div>

        <div class="result-text">
        Based on the information provided, the model predicts
        that the loan application is unlikely to be approved.
        </div>

        <br>

        <b>Model Confidence: {(1 - probability[1]) * 100:.2f}%</b>

        </div>
        """, unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">

Built with ❤️ using Python, Scikit-learn and Streamlit

<br>

Machine Learning • Data Science • AI

</div>
""", unsafe_allow_html=True)
