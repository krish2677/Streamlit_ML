import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Data Upload and Visualization", layout="wide")

st.title("📊 Data Upload and Visualization")

# Function to load data with caching
@st.cache_data
def load_data(file, file_extension):
    if file_extension == '.csv':
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

# File uploader
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file:
    # Get file extension
    file_extension = os.path.splitext(uploaded_file.name)[1]
    
    # Load data
    df = load_data(uploaded_file, file_extension)
    
    # Store the dataframe in session state to be used across pages
    st.session_state['df'] = df
    
    st.success("File uploaded successfully! You can now proceed to the Model Training page.")

    st.header("Data Preview")
    st.write(df.head())

    st.header("Data Visualization")
    
    # Charting Section
    chart_type = st.selectbox("Select chart type", ["Bar Chart", "Scatter Plot", "Histogram", "Box Plot"])

    if chart_type:
        st.subheader(f"Displaying: {chart_type}")
        
        if chart_type in ["Bar Chart", "Scatter Plot", "Box Plot"]:
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("Select X-axis", df.columns, key='x_axis')
            with col2:
                y_axis = st.selectbox("Select Y-axis", df.columns, key='y_axis')
            
            if x_axis and y_axis:
                try:
                    if chart_type == "Bar Chart":
                        fig = px.bar(df, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}")
                    elif chart_type == "Scatter Plot":
                        fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
                    elif chart_type == "Box Plot":
                        fig = px.box(df, x=x_axis, y=y_axis, title=f"Distribution of {y_axis} by {x_axis}")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating chart: {e}")

        elif chart_type == "Histogram":
            hist_col = st.selectbox("Select column for histogram", df.columns)
            if hist_col:
                try:
                    fig = px.histogram(df, x=hist_col, title=f"Distribution of {hist_col}")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating chart: {e}")
else:
    st.info("Please upload a file to get started.")
