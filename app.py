import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# Set up page configurations for premium aesthetics
st.set_page_config(
    page_title="Depression Risk Classifier",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for premium dark-themed UI and glassmorphism styling
st.markdown("""
<style>
    .reportview-container {
        background: #0f172a;
    }
    .main {
        background-color: #0f172a;
        color: #f8fafc;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    h1, h2, h3 {
        color: #f1f5f9 !important;
        font-weight: 700;
    }
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 12px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4);
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.6);
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
    }
    .card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }
    .risk-high {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: #fca5a5;
    }
    .risk-low {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: #a7f3d0;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model and features list safely
@st.cache_resource
def load_resources():
    model = joblib.load("random_forest_model.pkl")
    with open("model_features.json", "r") as f:
        features = json.load(f)
    return model, features

try:
    model, model_features = load_resources()
except Exception as e:
    st.error(f"Error loading model files: {e}. Please ensure 'random_forest_model.pkl' and 'model_features.json' exist in the same directory.")
    st.stop()

# Header Section
st.title("🧠 Depression Risk Assessment Dashboard")
st.markdown("A clinical & psychological predictor driven by an ensemble Random Forest model.")

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📋 Demographic & Profiling Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=18, max_value=65, value=25, step=1)
    gender = st.selectbox("Gender", ["Female", "Male"])
    role = st.selectbox("Current Role", ["Student", "Working Professional"])

with col2:
    city = st.selectbox("City", [
        "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Pune", "Chennai", 
        "Kolkata", "Jaipur", "Lucknow", "Surat", "Thane", "Kalyan", 
        "Nagpur", "Patna", "Indore", "Bhopal", "Ludhiana", "Ghaziabad", 
        "Agra", "Nashik", "Meerut", "Rajkot", "Varanasi", "Srinagar", "Vadodara"
    ])
    degree = st.selectbox("Highest Degree achieved / pursuing", [
        "Class 12", "BA", "B.Com", "B.Sc", "B.Tech", "B.Arch", "BEd", "BBA", "BCA", 
        "MA", "M.Com", "M.Sc", "M.Tech", "MBA", "MCA", "ME", "MBBS", "MD", "LLB", "LLM", "PhD"
    ])
st.markdown('</div>', unsafe_allow_html=True)

# Lifestyle & Biological Regulators
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🥗 Lifestyle & Biological Regulators")

col3, col4 = st.columns(2)

with col3:
    sleep_duration = st.selectbox("Sleep Duration", [
        "Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"
    ])
    diet = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])

with col4:
    sleep_hours = st.number_value = st.slider("Work / Study Hours (daily)", min_value=1.0, max_value=16.0, value=7.0, step=0.5)
st.markdown('</div>', unsafe_allow_html=True)

# Stressors & Clinical Indicators
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("⚡ Stressors & Clinical History")

col5, col6 = st.columns(2)

with col5:
    family_history = st.selectbox("Family History of Mental Illness", ["No", "Yes"])
    suicidal_thoughts = st.selectbox("Ever had suicidal thoughts?", ["No", "Yes"])
    financial_stress = st.slider("Financial Stress Level (1-5)", min_value=1.0, max_value=5.0, value=3.0, step=1.0)

with col6:
    if role == "Student":
        academic_pressure = st.slider("Academic Pressure (1-5)", min_value=1.0, max_value=5.0, value=3.0, step=1.0)
        cgpa = st.slider("CGPA", min_value=4.0, max_value=10.0, value=7.5, step=0.1)
        study_satisfaction = st.slider("Study Satisfaction (1-5)", min_value=1.0, max_value=5.0, value=3.0, step=1.0)
        
        # Professional features default to 0 during inference
        work_pressure = 0.0
        job_satisfaction = 0.0
        profession = "Student"
    else:
        work_pressure = st.slider("Work Pressure (1-5)", min_value=1.0, max_value=5.0, value=3.0, step=1.0)
        job_satisfaction = st.slider("Job Satisfaction (1-5)", min_value=1.0, max_value=5.0, value=3.0, step=1.0)
        profession = st.selectbox("Profession", [
            "Software Engineer", "Teacher", "Doctor", "Manager", "HR Manager", 
            "Civil Engineer", "Architect", "Consultant", "Entrepreneur", "Accountant", 
            "Chef", "Pharmacist", "Pilot", "Analyst", "Customer Support"
        ])
        
        # Student features default to 0 during inference
        academic_pressure = 0.0
        cgpa = 0.0
        study_satisfaction = 0.0
st.markdown('</div>', unsafe_allow_html=True)

# Make Prediction
if st.button("Predict Depression Risk"):
    # Create single-record DataFrame
    input_data = pd.DataFrame({
        "Gender": [gender],
        "Age": [float(age)],
        "City": [city],
        "Working Professional or Student": [role],
        "Profession": [profession],
        "Academic Pressure": [float(academic_pressure)],
        "CGPA": [float(cgpa)],
        "Study Satisfaction": [float(study_satisfaction)],
        "Sleep Duration": [sleep_duration],
        "Dietary Habits": [diet],
        "Degree": [degree],
        "Have you ever had suicidal thoughts ?": [suicidal_thoughts],
        "Work Pressure": [float(work_pressure)],
        "Job Satisfaction": [float(job_satisfaction)],
        "Financial Stress": [float(financial_stress)],
        "Family History of Mental Illness": [family_history],
        "Work/Study Hours": [float(sleep_hours)]
    })
    
    # 1. Clean 'Sleep Duration'
    sleep_mapping = {
        'Less than 5 hours': 'Less than 5 hours', '5-6 hours': '5-6 hours',
        '7-8 hours': '7-8 hours', 'More than 8 hours': 'More than 8 hours'
    }
    input_data['Sleep Duration'] = input_data['Sleep Duration'].map(sleep_mapping).fillna('7-8 hours')
    
    # 2. Clean 'Dietary Habits'
    diet_mapping = {
        'Moderate': 'Moderate', 'Unhealthy': 'Unhealthy', 'Healthy': 'Healthy'
    }
    input_data['Dietary Habits'] = input_data['Dietary Habits'].map(diet_mapping).fillna('Moderate')
    
    # 3. Clean Binary Fields
    binary_mapping = {'Yes': 1, 'No': 0}
    input_data['Have you ever had suicidal thoughts ?'] = input_data['Have you ever had suicidal thoughts ?'].map(binary_mapping).fillna(0).astype(int)
    input_data['Family History of Mental Illness'] = input_data['Family History of Mental Illness'].map(binary_mapping).fillna(0).astype(int)
    
    # 4. Get Dummies
    categorical_cols = ['Gender', 'City', 'Working Professional or Student', 'Profession', 'Sleep Duration', 'Dietary Habits', 'Degree']
    encoded_data = pd.get_dummies(input_data, columns=categorical_cols)
    
    # Reindex to match the training features exactly
    final_features = encoded_data.reindex(columns=model_features, fill_value=0)
    
    # Classify
    prediction = model.predict(final_features)[0]
    prob = model.predict_proba(final_features)[0][1]
    
    # Display Result
    st.markdown("### Prediction Results")
    if prediction == 1:
        st.markdown(f"""
        <div class="risk-high">
            <h2>⚠️ Elevated Depression Risk Detected</h2>
            <p style="font-size: 1.5rem; font-weight: bold; margin: 10px 0;">Risk Probability: {prob * 100:.1f}%</p>
            <p>The system predicts that this user profile shows significant signs correlating with depression. Please consider consulting a certified mental health professional.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="risk-low">
            <h2>✅ Low Depression Risk Detected</h2>
            <p style="font-size: 1.5rem; font-weight: bold; margin: 10px 0;">Risk Probability: {prob * 100:.1f}%</p>
            <p>The system predicts a low risk of depression based on the provided lifestyle, occupational, and clinical indicators.</p>
        </div>
        """, unsafe_allow_html=True)
