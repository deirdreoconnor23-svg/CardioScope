
# CardioScope

A machine learning cardiovascular disease screening tool built as a capstone project for the Technology Ireland Digital Skillsnet - AI Start -  Women in Tech Returner Programme.

## Project Overview

CardioScope uses machine learning to predict cardiovascular disease risk based on health and lifestyle factors. The model was trained on a CDC dataset of 419,656 records.

## Demo

▶️ [Watch the demo video](https://youtu.be/d5V3JSzUH5o)


## Key Results

- **Model:** Random Forest with SMOTE and cost-sensitive learning
- **Recall:** 88% (catches 88% of at-risk patients)
- **Precision:** 21% (appropriate trade-off for screening)

The precision-recall trade-off was a deliberate design decision. For a screening tool, missing someone at risk (false negative) is far more dangerous than flagging someone for further tests (false positive).

**Cost-Sensitive Learning**
The model uses cost-sensitive weights to penalise missed diagnoses more heavily than false alarms. 
This reflects the real-world cost asymmetry in healthcare screening - the consequences of missing someone with heart disease far outweigh the inconvenience of additional tests for a healthy person.

## Features

- Data cleaning and preprocessing pipeline
- Exploratory data analysis with visualisations
- Multiple model comparison (Logistic Regression, Random Forest, XGBoost)
- Class imbalance handling with SMOTE
- Streamlit web application for predictions

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Matplotlib, Seaborn

## Files

- `CVDClassificationBaseline.ipynb` - Initial model development
- `CVDClassificationOptimisation1.ipynb` - Hyperparameter tuning
- `CVDClassificationSmote.ipynb` - SMOTE implementation
- `CVDVisualisations.ipynb` - Data exploration and visualisations
- `cvd_app.py` - Streamlit application

## Data Source

CDC Behavioral Risk Factor Surveillance System (BRFSS) 2022 dataset.

## Author

Deirdre O'Connor
