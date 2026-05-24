ML Model Comparison Dashbaord

A responsive Streamlit web application that lets users upload any tabular dataset, handle missing data, pre-process categories natively,
and evaluate multiple regression models (`Linear Regression`, `Random Forest`, and `XGBoost`) simultaneously.

Features: 

    1. Dynamic Data Ingestion: Upload any standard CSV file with automated category extraction
    2. Automated Preprocessing Pipeline: Built-in scikit-learn ColumnTransformer execution that spearates numeric and categorical 
                                         columns, handles missing values via imputation and applies one-hot encoding natively with 
                                         zero data leakage
    3. Interactive Hyperparameter Tuning: Adjust parameters for Random Forest and XGBoost through a sidebar control panel
    4. Evaluation Suite: Evaluate models concurrently across multiple metrics, including RMSE, R2 Score and 5-fold Cross Validation R2
                         averages
    5. Explainable AI & Diagnostics: Feature Importance Tracking, Residual Profiling, Learning Dynamics Analysis



Local Setup & Installation:

Currently working on Python 3.11.11

1. Clone this repository:
```bash
   git clone https://github.com/DineshKA1/ml_model_dashboard_v1.git
   cd YOUR_REPOSITORY_NAME

2. Create & Activate a virutal environment
```bash
   python3 -m venv venv
   source venv/bin/activate

3. Install required dependencies
   Create a file named requirements.txt in the directory and add the following contents
   streamlit
   pandas
   numpy
   scikit-learn
   xgboost

   Run the command: pip install -r requirements.txt

4. Launch the dashboard locally by running: streamlit run app.py
