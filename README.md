Step 1: Upload Your Data 📂
First, you need to provide the data you want to analyze.
When you first launch the app, you will be on the welcome page. Look at the sidebar on the left.
Click on the "Data Upload and Visualization" page.
Click the "Browse files" button and select an .xlsx or .csv file from your computer.
Once uploaded, you will see a success message and a preview of the first few rows of your data.

Step 2: Visualize and Explore Your Data 📊
Before building a model, it's important to understand your data.
On the same "Data Upload and Visualization" page, scroll down to the "Data Visualization" section.
Use the dropdown menus to select a chart type (e.g., Bar Chart, Scatter Plot).
Select the columns you want to plot on the X-axis and Y-axis.
The chart will automatically update. You can try different combinations to find interesting patterns or relationships in your data.

Step 3: Configure Your Machine Learning Model ⚙️
Now you're ready to set up the model.
In the sidebar on the left, navigate to the "Model Training and Explanation" page.
You will see a sidebar with several configuration options:
Select Problem Type: Choose "Classification" if you are predicting a category (e.g., yes/no, spam/not spam) or "Regression" if you are predicting a number (e.g., price, sales).
Select Target Variable: This is the single column you want the model to learn how to predict.
Select Feature Columns: These are the input columns the model will use to make its prediction. By default, all columns except the target are selected. You can uncheck any you want to exclude.
Test Set Size (%): This slider determines how much of your data is held back to test the model's performance. A good starting point is 20%.
Select Model: Choose the specific algorithm you want to use (e.g., Random Forest).

Step 4: Train the Model and See Performance 🚀
Once your model is configured, it's time to train it.
After setting all the parameters in the sidebar, click the blue "Train Model" button.
The app will process the data and train the model. You will see a "Model trained successfully!" message.
Below that, in the "Model Performance" section, you will see key metrics like Accuracy (for classification) or R-squared (for regression) that tell you how well your model performed on the test data.

Step 5: Explain a Specific Prediction 🧠
This is the most powerful part of the app. You can ask the model why it made a certain prediction.
Scroll down to the "Model Explanation with LIME" section.
You will see a number input box. Enter the row number from the test set that you want to investigate (e.g., 0, 1, 2).
Click the "Explain Prediction" button.
A chart and a text explanation will appear, breaking down the model's reasoning for that single prediction. Green bars show features that supported the prediction, while red bars show features that went against it.
If you encounter the ValueError message here, it means one of your feature columns has no variety in the training data. Simply go back to the sidebar, uncheck that problematic column from your features, and train the model again.

Web app is live at 
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://krishlime.streamlit.app/)
