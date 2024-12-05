# Stock Price Prediction Using Machine Learning

## Project Overview

This project uses machine learning algorithms, specifically **Linear Regression** and **Support Vector Machine (SVM)** [updated], to predict the next day’s stock price based on historical stock data. The goal is to create a predictive model that can estimate tomorrow's stock closing price using the features of previous trading days, such as **Open**, **Close**, **High**, and **Low** prices.

## Features and Target

- **Features**:
  - `Open`: The stock price at the beginning of the trading day.
  - `High`: The highest stock price during the trading day.
  - `Low`: The lowest stock price during the trading day.
  - `Close`: The stock price at the end of the trading day.

- **Target**:
  - `Next Day's Close`: The closing price of the stock for the next trading day (i.e., the target we aim to predict).

## Algorithms Used

### 1. **Linear Regression**
Linear regression is a statistical method used to model the relationship between the dependent variable (next day's stock price) and the independent variables (previous day's stock prices). It provides a simple way to understand the trends in the stock market.

- **Evaluation Metrics**: Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² Score are used to evaluate the performance of the model.

### 2. **Support Vector Machine (SVM)**
Support Vector Machine (SVM) is a more advanced machine learning algorithm that works well with both linear and nonlinear data. In this project, we used **Support Vector Regression (SVR)** with the **RBF kernel** to predict the next day's stock price.

- **Evaluation Metrics**: Similar to Linear Regression, we evaluate SVM using MSE, MAE, and R².

## Steps Involved

### 1. Data Preprocessing
- Loaded and cleaned the historical stock data.
- Handled missing values by dropping rows with any NaN values.
- Generated new features like rolling averages for trend analysis (optional).
- Created the target variable, which is the next day's closing price.

### 2. Feature Scaling
Since SVM is sensitive to the scale of features, we scaled the features using **StandardScaler** from `sklearn`, ensuring that all input features have zero mean and unit variance.

### 3. Model Training
- We split the dataset into training and test sets using **KFold cross-validation** with 10 splits to evaluate the model performance more reliably.
- Trained both **Linear Regression** and **SVM** models on the dataset and evaluated their performance using MSE, MAE, and R² scores.

### 4. Model Evaluation
The models were evaluated using the following metrics:
- **MSE (Mean Squared Error)**: Measures the average squared difference between the predicted and actual values. Lower MSE indicates better performance.
- **MAE (Mean Absolute Error)**: Measures the average absolute difference between the predicted and actual values. Smaller MAE indicates better model performance.
- **R² Score**: A statistical measure that indicates the proportion of variance in the dependent variable explained by the independent variables. R² close to 1.0 indicates a better fit.

### 5. Visualization
- Plotted the evaluation metrics (MSE, MAE, and R²) for both models (Linear Regression and SVM) for comparison.

## Results

The project showed that both models performed similarly in terms of **MSE**, **MAE**, and **R²** scores. The results suggest that the data might not be highly complex and both models could capture the relationship between the features and the target well. 

- **Linear Regression** performed slightly better in MSE and R².
- **SVM** provided comparable results but could require more computational resources.

## How to Run the Code

- `.ipynb` file is given. So anyone can run this on colab.

## Future Improvements

1. **Feature Engineering**: Add more features such as moving averages, volatility measures, or technical indicators like RSI (Relative Strength Index) to improve model accuracy.
2. **Hyperparameter Tuning**: Further optimize the models using techniques like **Grid Search** or **Random Search** for better performance.
3. **Use More Advanced Models**: Experiment with more advanced machine learning models like **Random Forest**, **XGBoost**, or **LSTM (Long Short-Term Memory)** for better predictions, especially when dealing with time-series data.
4. **Time Series Cross-Validation**: Replace KFold with **TimeSeriesSplit** for a more appropriate cross-validation method in time-series data.

## Acknowledgements
- This project was inspired by the increasing use of machine learning in financial market predictions.
- Thanks to the open-source community for providing libraries like `scikit-learn`, `pandas`, and `matplotlib` that made this project possible.

---
