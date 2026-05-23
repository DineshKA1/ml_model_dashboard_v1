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
    rf_max_depth = st.sidebar.slider("RF Max Depth", min_value=1, max_value=30, value=15, help="No maps to deep branches if unchecked")

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
        cat_cols = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]

        numeric_transformer = SimpleImputer(strategy="mean")
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
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

                pred = pipeline.predict(X_test)
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

        st.subheader("Model Metrics Comparison Table")
        st.dataframe(results_df.style.highlight_max(subset=['R2 score', 'CV R2 Mean'], color="#26d89c"))

        st.subheader("Analytics")
        g_col1, g_col2, g_col3 = st.columns(3)
        with g_col1:
            st.write("##### R2 score")
            st.bar_chart(results_df.set_index("Model")['R2 score'])
        with g_col2:
            st.write("##### 5-fold CV R2 Mean")
            st.bar_chart(results_df.set_index("Model")['CV R2 Mean'])
        with g_col1:
            st.write("##### Training Time")
            st.bar_chart(results_df.set_index("Model")['Training time (s)'])



        #Feature Importance (FI)
        st.markdown("---")
        st.subheader("Attributes driving predictions")

        try:
            fitted_preprocessor = trained_models["Random Forest"].named_steps["preprocess"]
            encoded_cat_features = fitted_preprocessor.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(cat_cols)
            all_feature_names = list(numeric_cols) + list(encoded_cat_features)

            fi_col1, fi_col2 = st.columns(2)

            with fi_col1:
                st.write("##### Random Forest Priorities")
                rf_fi = trained_models["Random Forest"].named_steps["model"].feature_importances_
                rf_fi_df = pd.DataFrame({"Feature": all_feature_names, "Importance": rf_fi}).sort_values("Importance", ascending=False).head(12)
                st.bar_chart(rf_fi_df.set_index("Feature")["Importance"])

            with fi_col2:
                st.write("##### XGBoost Engine Priorities")
                xgb_fi = trained_models["XGBoost"].named_steps["model"].feature_importances_
                xgb_fi_df = pd.DataFrame({"Feature": all_feature_names, "Importance": xgb_fi}).sort_values("Importance", ascending=False).head(12)
                st.bar_chart(xgb_fi_df.set_index("Feature")["Importance"])

        except Exception as e:
            st.info("Heavily nested feature columns or structural mismathces")

        #error analysis
        st.markdown("---")
        st.subheader("Error & Residual Analysis")
        
        res_col1, res_col2, res_col3 = st.columns(3)

        #building comparative df pairing pred & ground truths

        for idx, (name, col_target) in enumerate(zip(models.keys(), [res_col1, res_col2, res_col3])):

            with col_target:

                st.write(f"##### {name} Residual Profile")
                residual_df = pd.DataFrame({
                    "Actual Val": y_test.values,
                    "Predicted Val": test_predictions[name]
                })

                st.scatter_chart(data=residual_df, x="Actual Val", y="Predicted Val")


        st.subheader("Continuous Evaluation: Learning Curves")

        with st.spinner("Generatiing learning curve profiles..."):

            lc_col1, lc_col2, lc_col3 = st.columns(3)

            train_sizes = np.linspace(0.1, 1.0, 5)

            for name, col_target in zip(models.keys(), [lc_col1, lc_col2, lc_col3]):

                with col_target:

                    st.write(f"##### {name} Learning Dynamics")
                    try:

                        sizes, train_scores, test_scores, = learning_curve(
                            trained_models[name], X_train, y_train,
                            train_sizes=train_sizes, cv=3, scoring="r2", n_jobns=-1
                        )

                        mean_train = train_scores.mean(axis=1)
                        mean_val = test_scores.mean(axis=1)

                        curve_df = pd.DataFrame({
                            "Training Smaples": sizes,
                            "Train Performance": train_scores.mean(axis=1),
                            "Validation Performance": test_scores.mean(axis=1)
                        }).set_index("Training Samples")

                        st.line_chart(curve_df)

                        final_train_score = mean_train[-1]
                        final_val_score = mean_val[-1]
                        score_gap = final_train_score-final_val_score

                        if final_train_score < 0.60 and final_val_score < 0.60:
                            st.error(f"**Underfitting Detected**\n\nBoth scores are low (Train: {final_train_score:.2f}, Val: {final_val_score:.2f}). The model lacks the capacity to capture the underlying patterns.")
                        elif score_gap > 0.15:
                            st.warning(f"**Overfitting Detected**\n\nHigh gap ({score_gap:.2f}) between Train ({final_train_score:.2f}) and Validation ({final_val_score:.2f}). The model is memorizing training details.")
                        elif final_val_score >= 0.80:
                            st.success(f"**Excellent Fit**\n\nHigh validation score ({final_val_score:.2f}) with low variance gap ({score_gap:.2f}).")
                        else:
                            st.info(f"**Stable Fit (Moderate Performance)**\n\nThe gap is healthy ({score_gap:.2f}), but performance is moderate (Val: {final_val_score:.2f}).")


                    except Exception as e:
                        st.info(f"Unable to compute data scaling metrics for {name}")