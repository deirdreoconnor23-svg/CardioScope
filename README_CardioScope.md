
![CardioScope Logo](images/Title__250_x_70_px_.png)

**AI-Powered Cardiovascular Disease Risk Assessment Tool**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Developed as the capstone project for the Technology Ireland Digital Skillsnet - AI Start - Women in Tech Returner Programme*

---

## Project Overview

CardioScope is a machine learning-powered screening tool that predicts cardiovascular disease (CVD) risk based on patient health indicators. Built on 419,656 CDC patient records, the model achieves **88.2% recall** - exceeding the 85% threshold typically required for medical screening tools.

### The Problem

Cardiovascular disease is the leading cause of death globally. Early screening and intervention are critical, but traditional risk assessment methods often miss at-risk patients. This project demonstrates how machine learning can improve screening sensitivity while maintaining practical utility.

### Key Achievement

Through iterative optimization, I improved the model from **53.4% recall** (too conservative, missing 47% of CVD cases) to **88.2% recall** by:
- Identifying underprediction through systematic testing
- Diagnosing class imbalance as the root cause
- Implementing cost-sensitive learning (1:5 weight ratio)
- Validating the optimal balance between recall and precision

This demonstrates complete problem-solving in medical ML: building, testing, diagnosing issues, and iterating to production-ready performance.

---

## Live Demo

**Try the app:** [Launch Instructions](#-installation--usage)



---

## 📈 Model Performance

### Final Model: Random Forest with Cost-Sensitive Learning

| Metric | Score | Clinical Context |
|--------|-------|------------------|
| **Recall (Sensitivity)** | **88.22%** ✅ | Catches 88 out of 100 CVD cases |
| **Accuracy** | 61.35% | Appropriate for screening context |
| **Precision** | 21.6% | Expected trade-off for high recall |
| **ROC-AUC** | 81.19% | Strong discriminative ability |
| **F1-Score** | 34.7% | Balanced composite metric |

### Model Evolution Journey

| Version | Approach | Accuracy | Recall | Issue |
|---------|----------|----------|--------|-------|
| **V1** | SMOTE only | 82.4% | 53.4% ❌ | Too conservative - missed 47% of cases |
| **V2** | Cost-Sensitive (1:10) | 50.8% | 93.8% | Too aggressive - 55% false positive rate |
| **V3** | Cost-Sensitive (1:5) | 61.4% | **88.2%** ✅ | Optimal balance achieved |

### Confusion Matrix (Test Set: 83,932 patients)

|  | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actually Negative** | 33,466 (TN) | 40,696 (FP) |
| **Actually Positive** | 606 (FN) | 9,164 (TP) |

**Key Insight:** Only 606 CVD cases missed out of 9,770 - a 6.2% false negative rate, which is clinically acceptable for a screening tool.

---

## Technical Stack

### Machine Learning
- **Python 3.8+** - Primary programming language
- **scikit-learn** - Model training, preprocessing, evaluation
- **XGBoost** - Gradient boosting implementation (comparison model)
- **Pandas & NumPy** - Data manipulation and numerical computing
- **imbalanced-learn (SMOTE)** - Class imbalance handling

### Data Visualization
- **Matplotlib & Seaborn** - Statistical visualizations
- **Plotly** - Interactive charts (if applicable)

### Web Application
- **Streamlit** - Interactive web interface
- **PIL (Pillow)** - Image processing for logo display

### Development Tools
- **Jupyter Notebook** - Exploratory data analysis and model development
- **Git & GitHub** - Version control and collaboration

---

## 📊 Dataset

**Source:** CDC Behavioral Risk Factor Surveillance System (BRFSS) 2022

- **Total Records:** 419,656 patient surveys
- **Features:** 24 self-reported health indicators
- **Target Variable:** Heart Attack OR Angina OR Stroke
- **Class Distribution:** 11.6% CVD, 88.4% healthy
- **Train/Test Split:** 80/20 stratified split

### Feature Categories

**Demographics:** Age, Sex, Race/Ethnicity, Education, Income  
**Health Status:** General health, Physical health days, Mental health days  
**Medical History:** Diabetes, COPD, Kidney disease, Arthritis, Depression, Asthma  
**Physical Metrics:** BMI, Height, Weight, Sleep hours  
**Lifestyle Factors:** Smoking status, Alcohol consumption, Physical activity  
**Functional Status:** Difficulty walking, concentrating, dressing, errands  
**Preventive Care:** Last checkup time

After one-hot encoding: **49 features**

---

## Methodology

### 1. Data Preprocessing
- Handled missing values and outliers
- Encoded categorical variables (one-hot encoding)
- Feature engineering and selection
- Stratified train-test split to preserve class distribution

### 2. Addressing Class Imbalance
- **SMOTE** (Synthetic Minority Over-sampling Technique) during cross-validation
- **Cost-Sensitive Learning** with 1:5 weight ratio
  - CVD cases weighted 5x more than healthy cases
  - Reflects real-world cost asymmetry: missing a disease is worse than a false alarm

### 3. Model Selection & Comparison

Evaluated three models:

| Model | Accuracy | Recall | Precision | ROC-AUC |
|-------|----------|--------|-----------|---------|
| Logistic Regression | 85.31% | 37.90% | 53.6% | 78.85% |
| Random Forest | **61.35%** | **88.22%** ✅ | 21.6% | **81.19%** |
| XGBoost | 60.12% | 86.45% | 20.8% | 80.34% |

**Winner:** Random Forest with cost-sensitive learning prioritizes catching CVD cases (high recall) over avoiding false alarms.

### 4. Why Lower Accuracy Is Acceptable

In medical screening:
- **Recall > Precision:** It's more important to catch potential cases than to avoid false alarms
- **Follow-up testing confirms:** Positive predictions undergo further clinical validation
- **Real-world parallel:** Mammography has ~85% recall and ~10% precision
- **Cost asymmetry:** Missing a disease has far greater consequences than extra testing

---

## Application Features

### Modern, Professional UI
- Custom brand identity with logo and color scheme
- Clean, flat design aesthetic
- Intuitive form layout with logical grouping
- Real-time BMI calculation
- Interactive sliders for health metrics

### Dual Model Comparison
- **Baseline:** Logistic Regression (for reference)
- **Optimized:** Random Forest (recommended for screening)
- Color-coded risk assessment (green = recommended model)

### Comprehensive Information
- Expandable sections for technical details
- Model evolution explanation
- Performance metrics breakdown
- Educational disclaimers

### Educational Focus
- Clear messaging about proof-of-concept status
- Explanation of precision-recall trade-offs
- Discussion of clinical validation requirements
- Transparent about limitations

---

## Installation & Usage

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/deirdreoconnor23-svg/CardioScope.git
   cd CardioScope
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the application**
   ```bash
   ./launch.sh
   ```
   
   Or manually:
   ```bash
   streamlit run src/app.py
   ```

4. **Access the app**
   
   Open your browser to `http://localhost:8501`

### Project Structure

```
CardioScope/
├── src/
│   └── app.py                      # Main Streamlit application
├── models/
│   ├── lr_model.pkl               # Logistic Regression model
│   ├── rf_model.pkl               # Random Forest model
│   └── feature_columns.pkl        # Feature definitions
├── images/
│   ├── Title__250_x_70_px_.png    # Logo
│   └── [visualizations]           # Model performance charts
├── notebooks/
│   └── CVDClassificationOptimisation1.ipynb  # Model development
├── results/
│   └── [csv files]                # Model metrics and comparisons
├── data/                          # Original dataset (not included)
├── requirements.txt               # Python dependencies
├── launch.sh                      # Convenience launch script
└── README.md                      # This file
```

---

## Screenshots

### Main Interface
*Clean, intuitive form for health data input*

![Main Interface](images/screenshot_main.png)

### Risk Assessment Results
*Dual model comparison with color-coded recommendations*

![Results Display](images/screenshot_results.png)

### Technical Details
*Expandable sections for in-depth metrics*

![Technical Details](images/screenshot_technical.png)

---

## Model Training

To retrain the models from scratch:

1. **Obtain the CDC BRFSS 2022 dataset**
2. **Open the Jupyter notebook:**
   ```bash
   jupyter notebook notebooks/CVDClassificationOptimisation1.ipynb
   ```
3. **Run all cells** to:
   - Load and preprocess data
   - Train multiple models
   - Perform hyperparameter tuning
   - Save models to `models/` directory

---

## Key Learnings & Insights

### Technical Lessons
1. **Accuracy isn't everything:** In imbalanced datasets with asymmetric costs, recall often matters more
2. **Cost-sensitive learning works:** Proper class weighting can dramatically improve minority class detection
3. **Iteration is crucial:** The path from 53% to 88% recall involved testing, diagnosing, and refining
4. **Real-world constraints matter:** Clinical thresholds, follow-up costs, and patient safety drive model design

### ML Best Practices Demonstrated
- ✅ Comprehensive data exploration and visualization
- ✅ Appropriate handling of class imbalance
- ✅ Multiple model comparison with consistent evaluation
- ✅ Clear documentation of methodology and decisions
- ✅ Honest assessment of limitations
- ✅ Production-ready deployment with user-friendly interface

### Medical ML Considerations
- Screening tools prioritize sensitivity (recall) over specificity
- False negatives have higher cost than false positives
- External validation and clinical trials are essential for deployment
- Regulatory approval (FDA/CE marking) is required for medical use
- Self-reported data has inherent limitations

---

## ⚠️ Limitations & Future Work

### Current Limitations
- **Educational only:** Not approved for clinical use
- **No external validation:** Tested only on BRFSS 2022 data
- **Self-reported data:** Potential for recall bias and measurement error
- **US population:** May not generalize to other populations
- **Cross-sectional data:** Cannot establish causality
- **No longitudinal tracking:** Cannot assess prediction over time

### Future Enhancements
- [ ] External validation on independent datasets
- [ ] Prospective clinical study design
- [ ] Integration of biomarker data (cholesterol levels, blood pressure readings)
- [ ] Temporal validation with longitudinal data
- [ ] SHAP or LIME for individual prediction explanations
- [ ] A/B testing against existing screening protocols
- [ ] Multi-language support for broader accessibility
- [ ] Mobile-responsive design optimization
- [ ] API endpoint for integration with health systems

---

## 🎓 About the Developer

**Deirdre O'Connor**

Returning tech professional with 10+ years in IT support (Dell International) and creative production management (Lumen Street Theatre). Currently upskilling in AI/ML through the Women in Tech Returner Programme.

**Background:**
- International IT support experience (Oslo, Malaysia, Moscow)
- BA in Art History and Creative Writing
- Transitioning back to technology with focus on ML/AI

**Connect:**
- LinkedIn: [Deirdre O'Connor](https://www.linkedin.com/in/deirdre-oconnor-327197330)
- GitHub: [@deirdreoconnor23-svg](https://github.com/deirdreoconnor23-svg)
- Email: [Deirdre O'Connor](mailto:deirdreoconnor23@gmail.com)

---

## 🙏 Acknowledgments

- **Technology Ireland Digital Skillsnet** - AI Start Programme
- **Women in Tech Returner Programme** - Training and support
- **CDC BRFSS** - Public dataset
- **Open source community** - Libraries and tools

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚖️ Disclaimer

**IMPORTANT:** CardioScope is an educational proof-of-concept and is **NOT** approved for medical use. 

This tool:
- ❌ Cannot diagnose cardiovascular disease
- ❌ Is not a substitute for professional medical advice
- ❌ Has not undergone clinical validation or regulatory approval
- ❌ Should not be used to make medical decisions

**Always consult qualified healthcare professionals for medical advice, diagnosis, or treatment.**

---

## 📞 Contact & Support

**Questions or feedback?** 
- Open an issue on GitHub
- Connect with me on LinkedIn [Deirdre O'Connor](https://www.linkedin.com/in/deirdre-oconnor-327197330)
- Email: [Deirdre O'Connor](mailto:deirdreoconnor23@gmail.com)

**Interested in collaboration?**
I'm open to opportunities in ML/AI, data science, and related fields!

---

*Built with ❤️ and Python by Deirdre O'Connor | December 2024*
