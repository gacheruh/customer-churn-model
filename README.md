# Telco Customer Churn – Project and Web App

## Project Overview
This project tackles the critical business problem of customer churn in the telecommunications sector. Using a dataset of customer information, we built a machine learning model to predict which customers are most likely to cancel their service. The goal is to enable proactive retention strategies, saving costs and increasing customer loyalty.

The project follows a complete data science workflow: from data cleaning and exploratory analysis to model building, evaluation, and interpretation of business insights.

## Business Problem
Customer churn represents a significant loss of revenue for telecom companies. Acquiring a new customer is far more expensive than retaining an existing one. This model identifies high-risk customers, allowing the business to target them with personalized retention campaigns before they decide to leave, thereby optimizing marketing resources and improving ROI.

## Data
The dataset contains information about 7,043 customers, with 20 features including:

- Demographics: gender, age (SeniorCitizen), partner, dependents
- Account information: tenure, contract type, paperless billing, payment method
- Services: phone, multiple lines, internet service, online security, streaming, etc.
- Charges: monthly charges and total charges
- Target variable: `Churn` – whether the customer left within the last month

## Project Steps
1. Data Cleaning & Preprocessing: Handle missing values in `TotalCharges`, encode the target variable, and remove the customer ID.
2. Exploratory Data Analysis (EDA): Analyze target distribution and relationships with key features. Identify churn drivers like contract type, internet service, and payment method.
3. Data Preprocessing for ML: Split data, standardize numeric features, and one-hot encode categorical features via a pipeline.
4. Model Building & Training: Compare models (e.g., Logistic Regression, Random Forest, XGBoost). In this web app, we use Logistic Regression for simplicity and strong baseline performance.
5. Evaluation: Use metrics suited for imbalance (Accuracy, Precision, Recall, F1, ROC-AUC). Review confusion matrices.
6. Results: Logistic Regression performs well with strong ROC-AUC and recall, effectively identifying likely churners.

## Web App
This repository contains a Streamlit app that trains and serves a churn prediction model using the Telco Customer Churn dataset.

### Run locally
1. Ensure you have Python 3.10+.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the app:
   ```bash
   streamlit run app.py
   ```

The app will train a model automatically if `models/churn_pipeline.joblib` is missing. You can retrain from the sidebar.

### Files
- `app.py`: Streamlit UI for single and batch predictions.
- `churn_model.py`: Training, saving, loading, and prediction utilities.
- `requirements.txt`: Python dependencies.
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: Training data.
- `.streamlit/config.toml`: Theme and telemetry settings.
