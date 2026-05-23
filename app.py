import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="ML Model Dashboard", layout="wide")

st.title("ML Model Comparison Dashboard")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    for col in df.columns:
        if isinstance(df[col].dtype, pd.StringDtype) or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(object)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Basic Info")
    st.write("Shape:", df.shape)

    target = st.selectbox("Select Target Column", df.columns)

    if st.button("Train Models"):

        X = df.drop(columns=[target])
        y = df[target]
        
        if y.isnull().any():
            st.warning("Target column contains missing values. Dropping those rows for training.")
            valid_idx = y.notnull()
            X = X[valid_idx]
            y = y[valid_idx]

        # Robust numeric vs categorical column selection
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        cat_cols = X.select_dtypes(exclude=[np.number]).columns

        numeric_transformer = SimpleImputer(strategy="mean")

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output="False"))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, cat_cols)
            ]
            
        )

        #X = X.select_dtypes(include=[np.number])


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "Linear Regression": Pipeline([
                ("preprocess", preprocessor),
                ("model", LinearRegression())
            ]),
    
            "Random Forest": Pipeline([
                ("preprocess", preprocessor),
                ("model", RandomForestRegressor(n_estimators=100, random_state=42))
            ]),
    
            "XGBoost": Pipeline([
                ("preprocess", preprocessor),
                ("model", XGBRegressor(n_estimators=100, learning_rate=0.1))
            ])
        }

        results = []

        with st.spinner("Training models..."):
            for name, model in models.items():
                model.fit(X_train, y_train)
                pred = model.predict(X_test)

                rmse = np.sqrt(mean_squared_error(y_test, pred))
                r2 = r2_score(y_test, pred)

                results.append({
                    "Model": name,
                    "RMSE": rmse,
                    "R2 score": r2
                })

        results_df = pd.DataFrame(results)

        st.subheader("Model Comparison")
        st.dataframe(results_df)

        st.bar_chart(results_df.set_index("Model")["RMSE"])
        st.bar_chart(results_df.set_index("Model")["R2 score"])

        col1, col2 = st.columns(2)
        with col1:
            st.write("### RMSE (Lower is better)")
            st.bar_chart(results_df.set_index("Model")["RMSE"])
        with col2:
            st.write("### R2 Score (Higher is better)")
            st.bar_chart(results_df.set_index("Model")["R2 score"])