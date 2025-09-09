# Customer Churn Prediction Model

## Project Overview
This project tackles the critical business problem of **customer churn** in the telecommunications sector. Using a dataset of customer information, I built a **machine learning model** to predict which customers are most likely to cancel their service. The goal is to enable proactive retention strategies, ultimately saving costs and increasing customer loyalty.

The project follows a complete data science workflow: from data cleaning and exploratory analysis to model building, evaluation, and interpretation of business insights.

## Business Problem
Customer churn represents a significant loss of revenue for telecom companies. Acquiring a new customer is far more expensive than retaining an existing one. This model identifies high-risk customers, allowing the business to target them with personalized retention campaigns before they decide to leave, thereby optimizing marketing resources and improving ROI.

## Data
The dataset contains information about 7,043 customers, with 20 features including:

* **Demographics:** gender, age (SeniorCitizen), partner, dependents

* **Account information:** tenure, contract type, paperless billing, payment method

* **Services:** phone, multiple lines, internet service, online security, streaming, etc.

* **Charges:** monthly charges and total charges

* **Target variable:**``` Churn``` - whether the customer left within the last month

## Project Steps
1. **Data Cleaning & Preprocessing:** Handled missing values in ```TotalCharges```, encoded the target variable, and removed the customer ID.

2. **Exploratory Data Analysis (EDA):** Analyzed the distribution of the target variable and investigated the relationship between churn and key features through visualizations. Identified significant churn drivers like contract type, internet service, and payment method.

3. **Data Preprocessing for ML:** Addressed class imbalance in the target variable. Split the data into training and test sets. Built a pipeline to standardize numerical features and one-hot encode categorical features.

4. **Model Building & Training:** Trained and compared three different machine learning algorithms:
  * **Logistic Regression** (with class weighting)

  * **Random Forest Classifier**

  * **XGBoost Classifier**

5. **Model Evaluation:** Evaluated models using a suite of metrics appropriate for imbalanced data: Accuracy, Precision, Recall, F1-Score, and ROC-AUC. Analyzed confusion matrices for each model.

6. **Results & Conclusion:** Logistic Regression emerged as the best model for this task, achieving the highest ROC-AUC score (0.836) and recall, making it the most effective at correctly identifying customers who will churn.

## Key Insights
* The strongest predictors of churn are: **Month-to-month contracts, Fiber optic internet service, and payment by electronic check.**

* Customers who are **senior citizens** or have **no partners or dependents** also show a higher propensity to churn.

* A simpler model (Logistic Regression) can outperform more complex ensemble methods like XGBoost without extensive hyperparameter tuning, highlighting the importance of model comparison.

Technologies Used
* Programming Language: Python

* Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost

* Machine Learning: Logistic Regression, Random Forest, XGBoost, ROC-AUC evaluation, confusion matrices, pipelines, StandardScaler, OneHotEncoder
