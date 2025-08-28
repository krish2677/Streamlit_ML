import streamlit as st

st.set_page_config(
    page_title="Automated ML Web App",
    page_icon="🤖",
    layout="wide"
)

st.title("Welcome to the Automated Machine Learning App! 🤖")

st.markdown("""
This interactive web application is designed to streamline your machine learning workflow.
You can easily upload your dataset, visualize it, train a model, and get insights into its predictions.

**Here's how to get started:**

1.  Navigate to the **Data Upload and Visualization** page using the sidebar on the left.
2.  Upload your dataset in either `.xlsx` or `.csv` format.
3.  Explore your data with various interactive charts to understand its structure.
4.  Once you're ready, head over to the **Model Training and Explanation** page.
5.  Select your features, target variable, model type, and train your model to see the results!

This app uses powerful libraries like `pandas`, `scikit-learn`, `plotly`, and `LIME` to provide a seamless experience.
""")

st.info("Please proceed to the **Data Upload and Visualization** page to begin.")
