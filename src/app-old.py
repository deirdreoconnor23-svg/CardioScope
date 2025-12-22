# cvd_app.py - Cardiovascular Disease Risk Predictor
# CardioScope - Women in Tech Returner Programme Capstone Project
# Final Version with Cost-Sensitive Learning (88.2% Recall)

import streamlit as st
import pandas as pd
import pickle

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="CardioScope - CVD Risk Predictor",
    page_icon="❤️",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .risk-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .high-risk {
        background-color: #FFE5E5;
        border: 2px solid #FF4B4B;
    }
    .moderate-risk {
        background-color: #FFF4E5;
        border: 2px solid #FFA500;
    }
    .low-risk {
        background-color: #E5F5E5;
        border: 2px solid #4BFF4B;
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
        with open("lr_model.pkl", "rb") as f:
            lr_model = pickle.load(f)
        with open("rf_model.pkl", "rb") as f:
            rf_model = pickle.load(f)
        with open("feature_columns.pkl", "rb") as f:
            feature_columns = pickle.load(f)
        return lr_model, rf_model, feature_columns
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please make sure lr_model.pkl, rf_model.pkl, and feature_columns.pkl are in the same directory as this app.")
        st.info("Run the save_models code in your notebook first to create these files.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

# ============================================================================
# PREPROCESSING FUNCTION (FIXED TO MATCH TRAINING)
# ============================================================================
def preprocess_input(user_input, feature_columns):
    """Convert user input to match training data format EXACTLY"""
    # Create DataFrame from input
    input_df = pd.DataFrame([user_input])
    
    # One-hot encode ALL categorical columns (matching training preprocessing)
    input_encoded = pd.get_dummies(input_df)
    
    # Add missing columns with 0 (features not present in this input)
    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    # Keep only the columns that were in training (remove any extras)
    input_encoded = input_encoded[feature_columns]
    
    return input_encoded

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Title
    st.markdown('<h1 class="main-title">❤️ CardioScope</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Cardiovascular Disease Risk Prediction using Machine Learning</p>', 
                unsafe_allow_html=True)
    
    # Updated Disclaimer - Reflects improved performance but educational status
    st.info("""
    ℹ️ **EDUCATIONAL PROOF-OF-CONCEPT - Improved Model (Version 3)**

    **Model Performance:**
    • 88.2% recall - exceeds 85% clinical screening threshold ✓
    • Achieved through iterative cost-sensitive learning (class_weight 1:5)
    • 61.4% accuracy, 21.6% precision - appropriate for screening context
    
    **Key Achievement:**
    Through iterative optimization, this model improved from 53.4% recall (too conservative) 
    to 88.2% recall, demonstrating the complete problem-solving process in medical ML.

    **Important Limitations:**
    Despite achieving screening-appropriate metrics, this remains educational-only because it requires:
    • External validation on different populations
    • Prospective clinical studies
    • Comparison to existing screening protocols  
    • Regulatory approval (FDA/CE marking)

    **Never use for medical decisions. Always consult qualified healthcare professionals.**
    """)
    
    # Load models
    lr_model, rf_model, feature_columns = load_models()
    
    if lr_model is None:
        st.stop()
    
    st.markdown("---")
    
    # Create input form
    with st.form("patient_form"):
        st.subheader("📋 Patient Information")
        
        # ========== DEMOGRAPHICS ==========
        st.markdown("#### Demographics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sex = st.selectbox("Sex", ["Male", "Female"])
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
        
        col1, col2 = st.columns(2)
        with col1:
            general_health = st.selectbox("General Health", 
                ["Excellent", "Very good", "Good", "Fair", "Poor"])
        with col2:
            last_checkup = st.selectbox("Last Medical Checkup", [
                "Within past year (anytime less than 12 months ago)",
                "Within past 2 years (1 year but less than 2 years ago)",
                "Within past 5 years (2 years but less than 5 years ago)",
                "5 or more years ago"
            ])
        
        st.markdown("---")
        
        # ========== PHYSICAL MEASUREMENTS ==========
        st.markdown("#### Physical Measurements")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            height = st.number_input("Height (meters)", 
                min_value=1.0, max_value=2.5, value=1.70, step=0.01)
            weight = st.number_input("Weight (kg)", 
                min_value=30.0, max_value=300.0, value=70.0, step=0.5)
        with col2:
            bmi = weight / (height ** 2)
            st.metric("BMI", f"{bmi:.1f}")
            if bmi < 18.5:
                st.caption("🔵 Underweight")
            elif bmi < 25:
                st.caption("🟢 Normal")
            elif bmi < 30:
                st.caption("🟡 Overweight")
            else:
                st.caption("🔴 Obese")
        with col3:
            sleep_hours = st.slider("Sleep Hours (per night)", 0, 24, 7)
        
        col1, col2 = st.columns(2)
        with col1:
            physical_health_days = st.slider(
                "Days of Poor Physical Health (past 30)", 0, 30, 0)
        with col2:
            mental_health_days = st.slider(
                "Days of Poor Mental Health (past 30)", 0, 30, 0)
        
        st.markdown("---")
        
        # ========== MEDICAL CONDITIONS ==========
        st.markdown("#### Medical Conditions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Chronic Diseases**")
            had_diabetes = st.selectbox("Diabetes", [
                "No",
                "Yes",
                "No, pre-diabetes or borderline diabetes",
                "Yes, but only during pregnancy (female)"
            ])
            had_asthma = st.radio("Asthma", ["No", "Yes"], horizontal=True)
            had_copd = st.radio("COPD", ["No", "Yes"], horizontal=True)
        
        with col2:
            st.markdown("**Other Conditions**")
            had_depression = st.radio("Depression", ["No", "Yes"], horizontal=True)
            had_kidney = st.radio("Kidney Disease", ["No", "Yes"], horizontal=True)
            had_arthritis = st.radio("Arthritis", ["No", "Yes"], horizontal=True)
        
        with col3:
            st.markdown("**Functional Difficulties**")
            diff_concentrating = st.radio("Concentrating", ["No", "Yes"], horizontal=True)
            diff_walking = st.radio("Walking", ["No", "Yes"], horizontal=True)
            diff_dressing = st.radio("Dressing/Bathing", ["No", "Yes"], horizontal=True)
            diff_errands = st.radio("Errands", ["No", "Yes"], horizontal=True)
        
        st.markdown("---")
        
        # ========== LIFESTYLE ==========
        st.markdown("#### Lifestyle Factors")
        col1, col2 = st.columns(2)
        
        with col1:
            smoker_status = st.selectbox("Smoking Status", [
                "Never smoked",
                "Former smoker",
                "Current smoker - now smokes some days",
                "Current smoker - now smokes every day"
            ])
            alcohol = st.radio("Alcohol Drinker", ["No", "Yes"], horizontal=True)
        
        with col2:
            physical_activity = st.radio(
                "Physical Activities (past 30 days)", ["No", "Yes"], horizontal=True)
        
        # Submit button
        st.markdown("---")
        submitted = st.form_submit_button("🔍 Predict CVD Risk", 
                                          use_container_width=True)
    
    # ========== MAKE PREDICTION ==========
    if submitted:
        # Collect all inputs
        user_input = {
            'Sex': sex,
            'GeneralHealth': general_health,
            'PhysicalHealthDays': physical_health_days,
            'MentalHealthDays': mental_health_days,
            'LastCheckupTime': last_checkup,
            'PhysicalActivities': physical_activity,
            'SleepHours': sleep_hours,
            'HadAsthma': had_asthma,
            'HadCOPD': had_copd,
            'HadDepressiveDisorder': had_depression,
            'HadKidneyDisease': had_kidney,
            'HadArthritis': had_arthritis,
            'HadDiabetes': had_diabetes,
            'DifficultyConcentrating': diff_concentrating,
            'DifficultyWalking': diff_walking,
            'DifficultyDressingBathing': diff_dressing,
            'DifficultyErrands': diff_errands,
            'SmokerStatus': smoker_status,
            'RaceEthnicityCategory': race,
            'AgeCategory': age,
            'HeightInMeters': height,
            'WeightInKilograms': weight,
            'BMI': bmi,
            'AlcoholDrinkers': alcohol
        }
        
        try:
            # Preprocess input (now correctly matches training)
            input_encoded = preprocess_input(user_input, feature_columns)
            
            # Get predictions
            lr_prob = lr_model.predict_proba(input_encoded)[0][1] * 100
            rf_prob = rf_model.predict_proba(input_encoded)[0][1] * 100
            avg_prob = (lr_prob + rf_prob) / 2
            
            # Display results
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Logistic Regression", f"{lr_prob:.1f}%",
                       delta="37.9% recall")
            col2.metric("Random Forest (Optimized)", f"{rf_prob:.1f}%",
                       delta="88.2% recall ✓")
            col3.metric("Average Risk", f"{avg_prob:.1f}%")
            
            # Risk classification based on Random Forest (the optimized model)
            if rf_prob >= 70:
                risk_class = "high-risk"
                risk_emoji = "🚨"
                risk_text = "VERY HIGH RISK"
                risk_msg = "Immediate medical consultation strongly recommended."
            elif rf_prob >= 50:
                risk_class = "high-risk"
                risk_emoji = "⚠️"
                risk_text = "HIGH RISK"
                risk_msg = "Schedule appointment with healthcare provider soon."
            elif rf_prob >= 30:
                risk_class = "moderate-risk"
                risk_emoji = "ℹ️"
                risk_text = "MODERATE RISK"
                risk_msg = "Regular checkups and healthy lifestyle recommended."
            else:
                risk_class = "low-risk"
                risk_emoji = "✅"
                risk_text = "LOW RISK"
                risk_msg = "Continue maintaining healthy habits!"
            
            st.markdown(f"""
            <div class="risk-box {risk_class}">
                <h2>{risk_emoji} {risk_text}</h2>
                <p style="font-size: 1.2rem;">
                    Estimated CVD Risk (Optimized Model): <strong>{rf_prob:.1f}%</strong>
                </p>
                <p>{risk_msg}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Model explanation
            with st.expander("ℹ️ Understanding the Results - Model Evolution"):
                st.markdown("""
                **Model Evolution Journey:**
                
                **Version 1 (SMOTE only):**
                - 82.4% accuracy, 53.4% recall
                - Problem: Too conservative - missed 47% of CVD cases
                - Real-world test: High-risk patient received only 3.6% risk
                
                **Version 2 (Cost-Sensitive 1:10):**
                - 50.8% accuracy, 93.8% recall
                - Problem: Too aggressive - 55% false positive rate
                - Overcorrected the issue
                
                **Version 3 (Cost-Sensitive 1:5) - Current Model:** ⭐
                - 61.4% accuracy, 88.2% recall
                - Optimal balance - exceeds 85% clinical screening threshold
                - 21.6% precision - appropriate for screening context
                
                ---
                
                **What is Cost-Sensitive Learning?**
                
                We applied `class_weight={0: 1, 1: 5}`, meaning:
                - Missing a CVD case (false negative) is penalized 5x more than a false alarm
                - Reflects real-world screening priorities where catching disease is critical
                - This is why recall improved from 53% to 88%
                
                **Why Lower Accuracy Is Acceptable:**
                
                Accuracy dropped from 82% to 61% because the model now flags more people as 
                at-risk (including some false positives). This is appropriate for screening:
                - Medical screening prioritizes sensitivity (catching cases)
                - Follow-up tests confirm or rule out CVD
                - Similar to mammography (85% recall, 10% precision)
                - Missing disease is more harmful than false alarms
                
                ---
                
                **88.2% Recall Achievement:**
                
                This exceeds the 85% threshold typically required for medical screening tools.
                The model correctly identifies 88 out of 100 CVD cases, missing only 12.
                
                However, this is still educational-only as it requires:
                - External validation on different populations
                - Prospective clinical studies showing health outcomes
                - Regulatory approval and integration with clinical protocols
                """)
            
            # Technical details
            with st.expander("🔧 Technical Details"):
                st.markdown(f"""
                **Input Features:** 24 self-reported health indicators
                
                **Encoded Features:** {len(feature_columns)} (after one-hot encoding)
                
                **Models Compared:**
                - Logistic Regression: Linear relationships, interpretable
                - Random Forest: Non-linear patterns, higher complexity
                
                **Selected Model:** Random Forest with cost-sensitive learning
                - Cost weight ratio: 1:5 (CVD cases weighted 5x more)
                - Handles class imbalance: 88.4% healthy, 11.6% CVD
                - SMOTE oversampling during cross-validation
                
                **Performance Metrics (Random Forest):**
                - Accuracy: 61.35%
                - Recall (Sensitivity): 88.22% ✓
                - Precision: 21.6%
                - ROC-AUC: 81.19%
                - F1-Score: 34.7%
                
                **Confusion Matrix (Test Set):**
                - True Negatives: 33,466
                - False Positives: 40,696 (acceptable for screening)
                - False Negatives: 606 (only 6% of CVD cases missed)
                - True Positives: 9,164
                
                **Dataset:** 419,656 CDC BRFSS survey responses
                - Training: 335,724 samples (80%)
                - Testing: 83,932 samples (20%)
                - Stratified split to preserve class distribution
                """)
            
            # Disclaimer
            st.warning("""
            ⚠️ **EDUCATIONAL USE ONLY**

            While this model achieves 88.2% recall (above screening threshold), it is NOT 
            approved for clinical use. Clinical deployment requires:
            - External validation on diverse populations
            - Prospective studies demonstrating improved health outcomes
            - Regulatory approval (FDA/CE marking)
            - Integration with existing clinical protocols
            
            This demonstrates ML capabilities and limitations in medical applications.
            **Always consult qualified healthcare professionals for medical advice.**
            """)
            
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.info("Please check that your model files are up to date and compatible.")

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("ℹ️ About CardioScope")
    
    st.markdown("""
    ### Project Information
    **CardioScope** is a proof-of-concept machine learning screening tool 
    developed for the Women in Tech Returner Programme capstone project.
    
    Through iterative optimization, we achieved screening-appropriate 
    performance while learning valuable lessons about medical ML deployment.
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### Dataset
    - **Source**: CDC BRFSS Survey
    - **Samples**: 419,656 patient records
    - **Features**: 24 self-reported indicators
    - **Target**: Heart Attack OR Angina OR Stroke
    - **Split**: 80/20 train/test
    - **Class Distribution**: 11.6% CVD, 88.4% healthy
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### Model Evolution
    
    **Version 1 (SMOTE only):**
    - Accuracy: 82.4%
    - Recall: 53.4% ❌
    - Issue: Too conservative
    
    **Version 2 (Weight 1:10):**
    - Accuracy: 50.8%
    - Recall: 93.8%
    - Issue: Too aggressive
    
    **Version 3 (Weight 1:5):** ⭐
    - Accuracy: 61.4%
    - Recall: 88.2% ✓
    - Status: Optimal balance
    
    **Key Learning:**
    Demonstrates precision-recall tradeoff and 
    iterative problem-solving in medical ML.
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### Performance Metrics
    
    **Logistic Regression (Baseline):**
    - Accuracy: 85.31%
    - Recall: 37.90%
    - ROC-AUC: 78.85%
    
    **Random Forest (Optimized):** ⭐
    - Accuracy: 61.35%
    - Recall: 88.22% ✅
    - Precision: 21.6%
    - ROC-AUC: 81.19%
    - Cost-Sensitive: 1:5 weight
    
    **Why This Model?**
    Through cost-sensitive learning, we achieved:
    - 88.2% recall (exceeds 85% clinical threshold)
    - Appropriate precision for screening (21.6%)
    - Prioritizes catching CVD cases over accuracy
    - Similar profile to real screening tools
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### Key Achievements
    
    ✅ Complete ML pipeline deployment
    
    ✅ Identified underprediction through testing
    
    ✅ Diagnosed class imbalance root cause
    
    ✅ Implemented cost-sensitive learning
    
    ✅ Achieved 88% recall (clinical threshold)
    
    ✅ Honest assessment of limitations
    
    ### Limitations
    
    ⚠️ Educational proof-of-concept only
    
    ⚠️ Requires external validation
    
    ⚠️ No prospective studies
    
    ⚠️ No regulatory approval
    
    ⚠️ Self-reported data limitations
    """)
    
    st.markdown("---")
    st.caption("Women in Tech Returner Programme")
    st.caption("Data Analytics & AI Capstone Project")
    st.caption("Developed by Deirdre O'Connor")

# Run the app
if __name__ == "__main__":
    main()
