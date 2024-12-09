import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, precision_score
import matplotlib.pyplot as plt
@st.cache_data
# Prepare dataset for training
@st.cache_data
def load_data():
    ticker_symbol = 'AAPL'

    # Fetch the data
    ticker_data = yf.Ticker(ticker_symbol)

    # Get all historical prices
    appleData = ticker_data.history("10Y")

    # Save to a CSV file
    appleData.to_csv("apple_stock_data.csv")

    # Read the CSV file into a DataFrame
    data = pd.read_csv('apple_stock_data.csv')

    data.columns = data.columns.str.strip().str.lower()  # Standardize column names to lowercase
    data['date'] = pd.to_datetime(data['date'], utc=True)  # Ensure 'date' is in datetime format
    data = data.sort_values(by='date')  # Sort by date
    return data

# Create necessary features
apple = load_data()
apple['day'] = apple['date'].dt.day
apple['month'] = apple['date'].dt.month
apple['year'] = apple['date'].dt.year
apple['is_quarter_end'] = np.where(apple['month'] % 3 == 0, 1, 0)
apple['open-close']  = apple['open'].shift(1) - apple['close'].shift(1)
apple['high-low']  = apple['high'].shift(1) - apple['low'].shift(1)
apple['target'] = np.where(apple['close'].shift(-1) > apple['close'], 1, 0)
apple = apple.dropna(subset=['open-close', 'high-low'])
# Select features
features = apple[['open', 'open-close', 'high-low', 'day', 'month', 'year', 'is_quarter_end']]
target = apple['target']
dates = apple['date']

# Scale features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Train-test split
X_train, X_valid, Y_train, Y_valid, dates_train, dates_test = train_test_split(features, target, dates, test_size=0.3, random_state=2022)

logisticRegression = LogisticRegression()
logisticRegression.fit(X_train, Y_train)
SVM = SVC(kernel='poly', probability=True)
SVM.fit(X_train, Y_train)
XGB = XGBClassifier()
XGB.fit(X_train, Y_train)
LogisticRegressionPred = logisticRegression.predict(X_valid)
SVCpred = SVM.predict(X_valid)
XGBpred = XGB.predict(X_valid)

# Create a DataFrame for results
results = pd.DataFrame({
    'Date': dates_test,
    'LogisticRegression': LogisticRegressionPred,
    'SVM': SVCpred,
    'XGBClassifier': XGBpred,
    'actual_close': Y_valid
}).sort_values(by="Date")

# Today's prediction
latest_data = apple.iloc[-1][['open', 'open-close', 'high-low', 'day', 'month', 'year', 'is_quarter_end']].values.reshape(1, -1)
latest_data_scaled = scaler.transform(latest_data)
TodayLogisticRegressionPred = logisticRegression.predict(latest_data_scaled)
TodaySVCpred = SVM.predict(latest_data_scaled)
TodayXGBpred = XGB.predict(latest_data_scaled)
TodayLogisticRegressionResult = 'Up' if TodayLogisticRegressionPred[0] == 1 else 'Down'
TodaySVCResult = 'Up' if TodaySVCpred[0] == 1 else 'Down'
TodayXGBResult = 'Up' if TodayXGBpred[0] == 1 else 'Down'

# Streamlit Interface
st.title("Stock Price Prediction for AAPL (2014-2024 Testing Set)")

# Display today's predictions
st.write("### Today's Predicted Trend")
st.metric(label="Logistic Regression", value=TodayLogisticRegressionResult)
st.metric(label="SVM", value=TodaySVCResult)
st.metric(label="XGB", value=TodayXGBResult)

# Dropdown and input for date selection
st.write("### Search Stock Data by Date")
search_date = st.text_input("Enter a date (YYYY-MM-DD):", "")
selected_date = st.selectbox("Or select a date:", options=results['Date'].dt.strftime('%Y-%m-%d'))
# Display results for the selected date
if search_date:
    try:
        search_date = pd.to_datetime(search_date)
        selected_row = results[results['Date'] == search_date]
        if not selected_row.empty:
            st.write(f"### Results for {search_date.strftime('%Y-%m-%d')}")
            for _, row in selected_row.iterrows():
                st.write(f"**Logistic regression:** {'Up' if row['LogisticRegression'] == 1 else 'Down'}")
                st.write(f"**SVM:** {'Up' if row['SVM'] == 1 else 'Down'}")
                st.write(f"**XGB:** {'Up' if row['XGBClassifier'] == 1 else 'Down'}")
                st.write(f"**Actual Close:** {'Up' if row['actual_close'] == 1 else 'Down'}")
        else:
            st.error("No data found for the entered date.")
    except ValueError:
        st.error("Invalid date format. Please use YYYY-MM-DD.")
elif selected_date:
    selected_row = results[results['Date'].dt.strftime('%Y-%m-%d') == selected_date]
    st.write(f"### Results for {selected_date}")
    for _, row in selected_row.iterrows():
        st.write(f"**Logistic regression:** {'Up' if row['LogisticRegression'] == 1 else 'Down'}")
        st.write(f"**SVM:** {'Up' if row['SVM'] == 1 else 'Down'}")
        st.write(f"**XGB:** {'Up' if row['XGBClassifier'] == 1 else 'Down'}")
        st.write(f"**Actual Close:** {'Up' if row['actual_close'] == 1 else 'Down'}")

# Line chart comparing predicted and actual close prices
st.write("Historical Data")
st.line_chart(apple['close'])