import streamlit as st
import pandas as pd
import numpy as np
import time 

from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
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

    # stripping out Pandas3.0 StringDtypes problems
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].astype(object)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Basic Info")
    st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")

    target = st.selectbox("Select Target Column", df.columns)

    st.sidebar.header("Model Hyperparameters")

    st.sidebar.subheader("Random Forest")
    rf_n_estimators = st.sidebar.slider("RF Estimators (No. of Trees)", min_value=10, max_value=300, value=100, step=10)
    rf_max_depth = st.sidebar.slider("RF Max Depth", min_value=1, max_value=30, value=15, help="None maps to deep branches if unchecked")

    st.sidebar.markdown("---")

    st.sidebar.subheader("XGBoost Tuning")
    xgb_n_estimators = st.sidebar.slider("XGB Estimators", min_value=1, max_value=15, value=6, step=1)
    xgb_lr = st.sidebar.slider("XGB Learning Rate", min_value=0.01, max_value=0.05, value=0.1, step=0.01)
    xgb_max_depth = st.sidebar.slider("XGB Max Depth", min_value=1, max_value=15, value=6, step=1)


    if st.button("Train & Evaulate Models"):

        X = df.drop(columns=[target])
        y = df[target]
        
        # double check for missing target values
        if y.isnull().any():
            st.warning("Target column contains missing values. Dropping affected rows for training.")
            valid_idx = y.notnull()
            X = X[valid_idx]
            y = y[valid_idx]

        # re-verify and isolate column backends securely
        for col in X.columns:
            if pd.api.types.is_string_dtype(X[col]) or pd.api.types.is_object_dtype(X[col]):
                X[col] = X[col].astype(object)

        # numeric vs categorical column selection
        numeric_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
        cat_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]

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
                ("model", RandomForestRegressor(
                    n_estimators=rf_n_estimators,
                    max_depth=rf_max_depth,
                    random_state=42,
                    n_jobs=-1
                    ))
            ]),
    
            "XGBoost": Pipeline([
                ("preprocess", preprocessor),
                ("model", XGBRegressor(
                    n_estimators=xgb_n_estimators, 
                    learning_rate=xgb_lr,
                    max_depth=xgb_max_depth,
                    random_state=42,
                    n_jobs=-1
                    ))
            ])
        }

        results = []
        trained_models = {}
        test_predictions = {}

        with st.spinner("Executing pipelines & calculating cross-validation"):

            for name, pipeline in models.items():
                
                start_time = time.time()
                pipeline.fit(X_train, y_train)
                end_time = time.time()
                train_time = end_time - start_time

                pred = pipeline.predict(y_test, pred)
                test_predictions[name] = pred

                rmse = np.sqrt(mean_squared_error(y_test, pred))
                r2 = r2_score(y_test, pred)

                cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
                cv_mean = cv_scores.mean()

                results.append({
                    "Model": name,
                    "RMSE": rmse,
                    "R2 score": r2,
                    "CV R2 Mean": cv_mean,
                    "Training time (s)": round(train_time, 4)
                })

                trained_models[name] = pipeline

        results_df = pd.DataFrame(results)

        st.success("Training complete")
        best_model = results_df.sort_values("R2 score", ascending=False).iloc[0]

        st.subheader("Performance Summary")
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric(label="Best Model (Highest R2)", value=best_model["Model"])
        with m_col2:
            st.metric(label="Top R2 Score", value=f"{best_model['R2 score']:.5f}")
        with m_col3:
            st.metric(label="Top CV R2 Mean", value=f"{best_model['CV R2 Mean']:.5f}")

        #st.subheader("Model Comparison")
        #st.dataframe(results_df)

        st.bar_chart(results_df.set_index("Model")["RMSE"])
        st.bar_chart(results_df.set_index("Model")["R2 score"])

        col1, col2 = st.columns(2)
        with col1:
            st.write("### RMSE (Lower is better)")
            st.bar_chart(results_df.set_index("Model")["RMSE"])
        with col2:
            st.write("### R2 Score (Higher is better)")
            st.bar_chart(results_df.set_index("Model")["R2 score"])