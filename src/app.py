# cvd_app.py - Cardiovascular Disease Risk Predictor
# CardioScope - Women in Tech Returner Programme Capstone Project
# Final Version with Cost-Sensitive Learning (88.2% Recall)

import streamlit as st
import pandas as pd
import pickle
import os
from PIL import Image

# ============================================================================
# FILE PATH CONFIGURATION
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")
IMAGES_DIR = os.path.join(BASE_DIR, "images")

LR_MODEL_PATH = os.path.join(MODELS_DIR, "lr_model.pkl")
RF_MODEL_PATH = os.path.join(MODELS_DIR, "rf_model.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")
LOGO_PATH = os.path.join(IMAGES_DIR, "Title__250_x_70_px_.png")

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="CardioScope - CVD Risk Assessment",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# BRANDED STYLING - FLAT DESIGN
# ============================================================================
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container with light background */
    .main {
        background: #f5f7f9 !important;
        padding: 0;
    }
    
    .stApp {
        background: #f5f7f9 !important;
    }
    
    section[data-testid="stAppViewContainer"] {
        background: #f5f7f9 !important;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Flat card styling with cream background */
    div[data-testid="stVerticalBlock"] > div:has(div.card-content) {
        background: #f8fafb;
        border-radius: 16px;
        padding: 2.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(137, 169, 186, 0.1);
    }
    
    /* Text colors - dark on light background */
    .stMarkdown, .stText {
        color: #233036;
    }
    
    /* Section headers - should be dark on cream cards */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #233036 !important;
        margin: 2rem 0 1.25rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #89a9ba;
    }
    
    /* Form labels - dark on cream */
    .stSelectbox label, .stNumberInput label, .stSlider label {
        font-weight: 600;
        color: #233036 !important;
        font-size: 0.95rem;
    }
    
    .stSelectbox > div > div, .stNumberInput > div > div > input {
        border-color: #89a9ba !important;
        background-color: #f8fafb !important;
        color: #233036 !important;
    }
    
    .stSelectbox > div > div:focus-within, 
    .stNumberInput > div > div:focus-within {
        border-color: #233036 !important;
        box-shadow: none !important;
    }
    
    /* Slider styling with brand colors */
    .stSlider > div > div > div {
        background-color: rgba(137, 169, 186, 0.3) !important;
    }
    
    .stSlider > div > div > div > div {
        background-color: #89a9ba !important;
    }
    
    /* Button with brand dark color */
    .stButton > button {
        background: #233036;
        color: #f8fafb;
        border: none;
        padding: 1rem 2rem;
        font-size: 1.05rem;
        font-weight: 600;
        border-radius: 8px;
        width: 100%;
        transition: background-color 0.2s;
        margin-top: 2rem;
    }
    
    .stButton > button:hover {
        background: #2d3e45;
        border: none;
    }
    /* Slider tick bar numbers - change background */
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] {
        background-color: #f5f7f9 !important;
        color: #233036 !important;
    }
    
    /* All slider text */
    .stSlider div[data-baseweb="slider"] div {
        background-color: transparent !important;
    }
    /* Metric card for BMI */
    div[data-testid="metric-container"] {
        background: rgba(137, 169, 186, 0.08);
        border: 1px solid rgba(137, 169, 186, 0.2);
        border-radius: 8px;
        padding: 1rem;
    }
    
    div[data-testid="metric-container"] > label {
        color: #89a9ba !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    div[data-testid="metric-container"] > div {
        color: #233036 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Warning/Info boxes with brand colors */
    .stAlert {
        border-radius: 12px;
        border-left-width: 4px;
    }
    
    div[data-baseweb="notification"] {
        background-color: rgba(137, 169, 186, 0.1);
        border-left: 4px solid #89a9ba;
    }
    
    /* Warning box */
    .stWarning {
        background-color: rgba(216, 68, 68, 0.1) !important;
        border-left-color: #d84444 !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(137, 169, 186, 0.1);
        border-radius: 8px;
        font-weight: 600;
        color: #233036;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: rgba(137, 169, 186, 0.15);
    }
    
    .streamlit-expanderContent {
        border: none;
        color: #233036;
    }
    
    /* Column spacing */
    [data-testid="column"] {
        padding: 0 0.5rem;
    }
    
    /* Form container */
    [data-testid="stForm"] {
        background: transparent;
        border: none;
    }
    
    /* Hide Streamlit form border */
    div[data-testid="stForm"] > div {
        border: none !important;
    }
    
    /* Loading spinner with brand color */
    .stSpinner > div {
        border-top-color: #89a9ba !important;
    }
    
    /* Links */
    a {
        color: #89a9ba;
    }
    
    a:hover {
        color: #233036;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS
# ============================================================================


@st.cache_resource
def load_models():
    """Load the trained models and feature list"""
    try:
        with open(LR_MODEL_PATH, "rb") as f:
            lr_model = pickle.load(f)
        with open(RF_MODEL_PATH, "rb") as f:
            rf_model = pickle.load(f)
        with open(FEATURES_PATH, "rb") as f:
            feature_columns = pickle.load(f)
        return lr_model, rf_model, feature_columns
    except FileNotFoundError:
        st.error(
            "⚠️ Model files not found. Please ensure models are in the models/ directory.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

# ============================================================================
# PREPROCESSING FUNCTION
# ============================================================================


def preprocess_input(user_input, feature_columns):
    """Convert user input to match training data format"""
    input_df = pd.DataFrame([user_input])
    input_encoded = pd.get_dummies(input_df)

    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    input_encoded = input_encoded[feature_columns]
    return input_encoded

# ============================================================================
# MAIN APP
# ============================================================================


def main():
    # Load and display logo with HTML for size control
    try:
        import base64
        with open(LOGO_PATH, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()

        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 1rem;">
            <img src="data:image/png;base64,{logo_data}" style="width: 1500px; height: auto;">
        </div>
        """, unsafe_allow_html=True)
    except:
        st.markdown(
            "<h1 style='text-align: center; color: #89a9ba;'>CardioScope</h1>", unsafe_allow_html=True)

    # Framed subtitle
    st.markdown("""
    <div style="background: white; padding: 1.5rem; border-radius: 12px; 
                text-align: center; margin: 1.5rem auto; max-width: 800px;
                border: 1px solid rgba(137, 169, 186, 0.2); box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
        <div style="color: #233036; font-size: 1.1rem; font-weight: 500; line-height: 1.6;">
            AI-Powered Cardiovascular Disease Risk Assessment - Educational demonstration only
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load models
    lr_model, rf_model, feature_columns = load_models()

    if lr_model is None:
        st.stop()

    # Main form
    with st.form("patient_form"):
        # Demographics Section
        st.markdown('<div class="section-header">Demographics</div>',
                    unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            sex = st.selectbox("Sex", ["Female", "Male"])
        with col2:
            age = st.selectbox("Age Category", [
                "Age 18 to 24", "Age 25 to 29", "Age 30 to 34", "Age 35 to 39",
                "Age 40 to 44", "Age 45 to 49", "Age 50 to 54", "Age 55 to 59",
                "Age 60 to 64", "Age 65 to 69", "Age 70 to 74", "Age 75 to 79",
                "Age 80 or older"
            ])
        with col3:
            race = st.selectbox("Race/Ethnicity", [
                "White only, Non-Hispanic",
                "Black only, Non-Hispanic",
                "Hispanic",
                "Other race only, Non-Hispanic",
                "Multiracial, Non-Hispanic"
            ])

        # General Health Section
        st.markdown('<div class="section-header">General Health</div>',
                    unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            general_health = st.selectbox("Overall Health",
                                          ["Excellent", "Very good", "Good", "Fair", "Poor"])
        with col2:
            last_checkup = st.selectbox("Last Medical Checkup", [
                "Within past year (anytime less than 12 months ago)",
                "Within past 2 years (1 year but less than 2 years ago)",
                "Within past 5 years (2 years but less than 5 years ago)",
                "5 or more years ago"
            ])

        # Physical Measurements Section
        st.markdown(
            '<div class="section-header">Physical Measurements</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            height = st.number_input("Height (meters)",
                                     min_value=1.0, max_value=2.5, value=1.70, step=0.01)
        with col2:
            weight = st.number_input("Weight (kg)",
                                     min_value=30.0, max_value=300.0, value=70.0, step=0.5)
        with col3:
            bmi = weight / (height ** 2)
            st.metric("BMI", f"{bmi:.1f}")

        col1, col2 = st.columns(2)
        with col1:
            sleep_hours = st.slider("Sleep Hours per Night", 0, 24, 7)

        col1, col2 = st.columns(2)
        with col1:
            physical_health_days = st.slider(
                "Days of Poor Physical Health (past 30)", 0, 30, 0)
        with col2:
            mental_health_days = st.slider(
                "Days of Poor Mental Health (past 30)", 0, 30, 0)

        # Medical History Section
        st.markdown(
            '<div class="section-header">Medical History</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            had_asthma = st.selectbox("Asthma", ["No", "Yes"])
            had_copd = st.selectbox(
                "COPD/Emphysema/Chronic Bronchitis", ["No", "Yes"])
            had_depressive_disorder = st.selectbox(
                "Depressive Disorder", ["No", "Yes"])

        with col2:
            had_kidney_disease = st.selectbox("Kidney Disease", ["No", "Yes"])
            had_arthritis = st.selectbox("Arthritis", ["No", "Yes"])
            had_diabetes = st.selectbox("Diabetes", [
                "No",
                "Yes",
                "No, pre-diabetes or borderline diabetes",
                "Yes, but only during pregnancy (female)"
            ])

        with col3:
            difficulty_concentrating = st.selectbox(
                "Difficulty Concentrating", ["No", "Yes"])
            difficulty_walking = st.selectbox(
                "Difficulty Walking/Climbing Stairs", ["No", "Yes"])
            difficulty_dressing = st.selectbox(
                "Difficulty Dressing/Bathing", ["No", "Yes"])

        col1, col2, col3 = st.columns(3)
        with col1:
            difficulty_errands = st.selectbox(
                "Difficulty with Errands", ["No", "Yes"])

        # Lifestyle Section
        st.markdown(
            '<div class="section-header">Lifestyle Factors</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            smoker_status = st.selectbox("Smoking Status", [
                "Never smoked",
                "Former smoker",
                "Current smoker - now smokes some days",
                "Current smoker - now smokes every day"
            ])
            physical_activities = st.selectbox(
                "Physical Activity (Past 30 Days)", ["No", "Yes"])

        with col2:
            alcohol_drinkers = st.selectbox("Alcohol Consumer", ["No", "Yes"])

        # Submit button
        submitted = st.form_submit_button("Analyze Risk")

        # Prediction
        if submitted:
            with st.spinner("Analyzing your health data..."):
                try:
                    # Prepare input
                    user_input = {
                        "Sex": sex,
                        "GeneralHealth": general_health,
                        "PhysicalHealthDays": physical_health_days,
                        "MentalHealthDays": mental_health_days,
                        "LastCheckupTime": last_checkup,
                        "PhysicalActivities": physical_activities,
                        "SleepHours": sleep_hours,
                        "AgeCategory": age,
                        "HeightInMeters": height,
                        "WeightInKilograms": weight,
                        "BMI": bmi,
                        "AlcoholDrinkers": alcohol_drinkers,
                        "HadAsthma": had_asthma,
                        "HadCOPD": had_copd,
                        "HadDepressiveDisorder": had_depressive_disorder,
                        "HadKidneyDisease": had_kidney_disease,
                        "HadArthritis": had_arthritis,
                        "HadDiabetes": had_diabetes,
                        "DifficultyConcentrating": difficulty_concentrating,
                        "DifficultyWalking": difficulty_walking,
                        "DifficultyDressingBathing": difficulty_dressing,
                        "DifficultyErrands": difficulty_errands,
                        "SmokerStatus": smoker_status,
                        "RaceEthnicityCategory": race
                    }

                    # Preprocess and predict
                    processed_input = preprocess_input(
                        user_input, feature_columns)

                    lr_pred = lr_model.predict(processed_input)[0]
                    lr_prob = lr_model.predict_proba(processed_input)[0][1]

                    rf_pred = rf_model.predict(processed_input)[0]
                    rf_prob = rf_model.predict_proba(processed_input)[0][1]

                    # Display results
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(
                        '<div class="section-header">Risk Assessment Results</div>', unsafe_allow_html=True)

                    col1, col2 = st.columns(2)

                    # Left: Baseline model (blue-gray)
                    with col1:
                        risk_label = "High Risk" if lr_pred == 1 else "Low Risk"
                        color = "#89a9ba"  # Brand blue-gray for baseline

                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, {color} 0%, #6d8a9a 100%); 
                                    color: white; border-radius: 12px; padding: 2rem; text-align: center;
                                    border: 2px solid rgba(0,0,0,0.05);">
                            <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 1rem;">Logistic Regression</div>
                            <div style="font-size: 3.5rem; font-weight: 800; margin: 0.75rem 0;">{lr_prob*100:.1f}%</div>
                            <div style="font-size: 1.25rem; font-weight: 600; margin-top: 0.5rem;">{risk_label}</div>
                            <div style="background: rgba(255,253,249,0.2); padding: 0.4rem 0.8rem; 
                                        border-radius: 6px; font-size: 0.85rem; margin-top: 0.75rem; display: inline-block;">
                                Baseline Model
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Right: Optimized model (green - recommended)
                    with col2:
                        risk_label = "High Risk" if rf_pred == 1 else "Low Risk"
                        color = "#51cf66"  # Green for recommended model

                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, {color} 0%, #37b24d 100%); 
                                    color: white; border-radius: 12px; padding: 2rem; text-align: center;
                                    border: 2px solid rgba(0,0,0,0.05);">
                            <div style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 1rem;">Random Forest (Optimized)</div>
                            <div style="font-size: 3.5rem; font-weight: 800; margin: 0.75rem 0;">{rf_prob*100:.1f}%</div>
                            <div style="font-size: 1.25rem; font-weight: 600; margin-top: 0.5rem;">{risk_label}</div>
                            <div style="background: rgba(255,253,249,0.2); padding: 0.4rem 0.8rem; 
                                        border-radius: 6px; font-size: 0.85rem; margin-top: 0.75rem; display: inline-block;">
                                ★ Recommended Model
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Info about results
                    st.markdown("""
                    <div class="info-banner">
                        <strong>About These Results:</strong> The Random Forest model uses cost-sensitive learning 
                        to achieve 88.2% recall, exceeding clinical screening thresholds. This model prioritizes 
                        catching potential CVD cases over avoiding false alarms.
                    </div>
                    """, unsafe_allow_html=True)

                    # Expandable sections
                    with st.expander("Understanding Your Results"):
                        st.markdown("""
                        ### Model Evolution Journey
                        
                        **Version 1 (SMOTE only):** 82.4% accuracy, 53.4% recall  
                        ❌ Too conservative - missed 47% of CVD cases
                        
                        **Version 2 (Cost-Sensitive 1:10):** 50.8% accuracy, 93.8% recall  
                        ❌ Too aggressive - 55% false positive rate
                        
                        **Version 3 (Cost-Sensitive 1:5) - Current Model:** ⭐  
                        ✅ 61.4% accuracy, 88.2% recall - optimal balance
                        
                        ---
                        
                        ### Why 88.2% Recall Matters
                        
                        This exceeds the 85% threshold typically required for medical screening tools.
                        The model correctly identifies 88 out of 100 CVD cases, missing only 12.
                        
                        Similar to mammography screening (85% recall, 10% precision), we prioritize
                        catching disease over avoiding false alarms.
                        """)

                    with st.expander("Technical Details"):
                        st.markdown(f"""
                        **Model Architecture:** Random Forest with cost-sensitive learning  
                        **Features:** {len(feature_columns)} (after one-hot encoding)  
                        **Cost Ratio:** 1:5 (CVD cases weighted 5x more)  
                        **Dataset:** 419,656 CDC BRFSS survey responses  
                        
                        **Performance Metrics:**
                        - Accuracy: 61.35%
                        - Recall: 88.22% ✓
                        - Precision: 21.6%
                        - ROC-AUC: 81.19%
                        """)

                    st.warning("""
                    ⚠️ **Educational Use Only**  
                    This model is a proof-of-concept and NOT approved for clinical use. 
                    Always consult qualified healthcare professionals for medical advice.
                    """)

                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
                    st.info("Please check that all fields are filled correctly.")

    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #89a9ba; font-size: 0.9rem;">
        <strong style="color: #233036;">CardioScope</strong> • Women in Tech Returner Programme • Developed by Deirdre O'Connor
    </div>
    """, unsafe_allow_html=True)


# Run the app
if __name__ == "__main__":
    main()
