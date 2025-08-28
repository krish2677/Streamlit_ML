import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Model Training and Explanation", layout="wide")

st.title("🧠 Model Training and Explanation")

# Check if dataframe is loaded
if 'df' not in st.session_state:
    st.warning("Please upload a dataset on the 'Data Upload and Visualization' page first.")
else:
    df = st.session_state['df']

    # --- Model Configuration ---
    st.sidebar.header("1. Model Configuration")
    
    problem_type = st.sidebar.selectbox(
        "Select Problem Type",
        ["Classification", "Regression"]
    )

    st.sidebar.header("2. Feature Selection")
    
    # Get all columns
    all_columns = df.columns.tolist()
    
    target_variable = st.sidebar.selectbox(
        "Select Target Variable",
        all_columns
    )

    feature_columns = st.sidebar.multiselect(
        "Select Feature Columns",
        [col for col in all_columns if col != target_variable],
        default=[col for col in all_columns if col != target_variable]
    )

    st.sidebar.header("3. Model Parameters")
    test_set_size = st.sidebar.slider(
        "Test Set Size (%)", 10, 50, 20, 5
    )

    if problem_type == "Classification":
        model_choice = st.sidebar.selectbox(
            "Select Model",
            ["Random Forest Classifier", "Logistic Regression"]
        )
    else: # Regression
        model_choice = st.sidebar.selectbox(
            "Select Model",
            ["Random Forest Regressor", "Linear Regression"]
        )

    if st.sidebar.button("Train Model"):
        if not feature_columns:
            st.error("Please select at least one feature column.")
        else:
            with st.spinner("Training in progress..."):
                # --- Data Preparation ---
                X = df[feature_columns]
                y = df[target_variable]

                # Add a check for regression target type
                if problem_type == "Regression" and not pd.api.types.is_numeric_dtype(y):
                    st.error(f"The selected target variable '{target_variable}' is not numeric. Please choose a numeric column for regression.")
                    st.stop() # Stop execution if the target is invalid
                
                # Convert categorical variables to numeric using one-hot encoding
                X = pd.get_dummies(X, drop_first=True)
                
                # Update feature names after one-hot encoding
                feature_names = list(X.columns)

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_set_size/100.0, random_state=42
                )

                # --- Model Training ---
                if problem_type == "Classification":
                    if model_choice == "Random Forest Classifier":
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                    else:
                        model = LogisticRegression(max_iter=1000)
                else: # Regression
                    if model_choice == "Random Forest Regressor":
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                    else:
                        model = LinearRegression()
                
                model.fit(X_train, y_train)
                st.session_state['model'] = model
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['feature_names'] = feature_names
                st.session_state['X_train'] = X_train
                st.session_state['y_train'] = y_train
                st.session_state['problem_type'] = problem_type


            st.success("Model trained successfully!")

            # --- Model Performance ---
            st.header("Model Performance")
            y_pred = model.predict(X_test)

            if problem_type == "Classification":
                accuracy = accuracy_score(y_test, y_pred)
                st.metric(label="Accuracy", value=f"{accuracy:.2f}")
            else: # Regression
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.metric(label="Mean Squared Error", value=f"{mse:.2f}")
                st.metric(label="R-squared", value=f"{r2:.2f}")

    # --- LIME Explanation Section ---
    if 'model' in st.session_state:
        st.header("Model Explanation with LIME")
        
        # Retrieve all necessary variables from session state
        model = st.session_state['model']
        X_test = st.session_state['X_test']
        feature_names = st.session_state['feature_names']
        X_train = st.session_state['X_train']
        y_train = st.session_state['y_train']
        problem_type = st.session_state['problem_type']

        # Select an instance from the test set to explain
        idx_to_explain = st.number_input(
            "Select a row index from the test set to explain", 0, len(X_test)-1, 0
        )
        
        if st.button("Explain Prediction"):
            with st.spinner("Generating explanation..."):
                try:
                    # Identify categorical features for LIME.
                    categorical_features_indices = [
                        i for i, col in enumerate(X_train.columns) 
                        if X_train[col].nunique() <= 2 or np.isclose(X_train[col].std(), 0)
                    ]

                    # Convert training data to a numpy array of floats for robustness
                    training_data_np = X_train.values.astype(float)

                    # Create a LIME explainer
                    if problem_type == "Classification":
                        explainer = lime.lime_tabular.LimeTabularExplainer(
                            training_data=training_data_np,
                            feature_names=feature_names,
                            class_names=[str(c) for c in np.unique(y_train)],
                            categorical_features=categorical_features_indices,
                            mode='classification'
                        )
                        predict_fn = model.predict_proba
                    else: # Regression
                        explainer = lime.lime_tabular.LimeTabularExplainer(
                            training_data=training_data_np,
                            feature_names=feature_names,
                            categorical_features=categorical_features_indices,
                            mode='regression'
                        )
                        predict_fn = model.predict

                    instance_to_explain = X_test.iloc[idx_to_explain].values
                    
                    # Get the explanation
                    explanation = explainer.explain_instance(
                        instance_to_explain,
                        predict_fn,
                        num_features=10 # Show top 10 features
                    )

                    # Display the explanation
                    st.write(f"Explanation for prediction on row {idx_to_explain}:")
                    
                    if problem_type == "Classification":
                        st.components.v1.html(explanation.as_html(), height=800, scrolling=True)
                    else:
                        fig = explanation.as_pyplot_figure()
                        st.pyplot(fig)

                    # Add the explanation text
                    st.markdown("""
                    ---
                    **How to Read This Chart:**

                    This chart explains the 'why' behind the prediction for this specific data point.
                    - **<font color='green'>Green bars</font>** show features that pushed the prediction **higher** (or towards the predicted class).
                    - **<font color='red'>Red bars</font>** show features that pushed the prediction **lower** (or away from the predicted class).
                    - The **longer the bar**, the more influential the feature was for this single decision.
                    """, unsafe_allow_html=True)

                except ValueError as e:
                    if 'Domain error in arguments' in str(e):
                        st.error("LIME Explanation Failed: A `ValueError` occurred.")
                        st.warning(
                            "This usually happens when a feature column has no variation "
                            "(i.e., all values are the same) in the training data subset. "
                            "This results in a standard deviation of zero, which LIME cannot handle."
                        )

                        # Find and display the problematic columns
                        problematic_cols = [
                            col for col in X_train.columns if np.isclose(X_train[col].std(), 0)
                        ]
                        if problematic_cols:
                            st.write("Problematic column(s) identified in the current training set:")
                            st.code(problematic_cols)
                            st.info("Consider removing these columns from your feature selection or adjusting the test set size.")
                        else:
                            st.write("Could not automatically identify the problematic column. Please check your data for columns with constant values.")
                    else:
                        # Handle other potential ValueErrors
                        st.error(f"An unexpected error occurred: {e}")
