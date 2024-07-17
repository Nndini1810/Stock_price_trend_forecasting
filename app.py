import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import plotly.graph_objs as go

# Custom CSS to set the background image
st.markdown(
    """
    <style>
    .main {
        background-image: url('https://img.freepik.com/free-vector/stock-market-exchange-background-with-chart-diagram-investment_1017-44920.jpg?t=st=1721121913~exp=1721125513~hmac=f6a99de4af711db9b84ffdfb91604b0e64818cace4c548aef7d1f3ab60222e2e&w=1060');
        background-size: cover;
    }
    .title {
        font-size: 48px;
        font-weight: bold;
        font-family: 'Arial';
        color: #ff9800;
        text-shadow: 1px 1px 2px black;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 1. Import libraries

# 2. Define start and today
start = st.sidebar.date_input('Start date', datetime.date(2010, 1, 1))
today = st.sidebar.date_input('End date', datetime.date.today())

# Convert start and today to datetime.datetime objects
start_datetime = datetime.datetime.combine(start, datetime.datetime.min.time())
today_datetime = datetime.datetime.combine(today, datetime.datetime.min.time())

# 3. Set title
st.title('Stock Price Trend Forecasting')

# 4. user_input variable and make df
stock_symbol = st.sidebar.text_input('Enter Stock Symbol', 'AAPL')
df = pd.read_csv(f'https://query1.finance.yahoo.com/v7/finance/download/{stock_symbol}?period1={int(start_datetime.timestamp())}&period2={int(today_datetime.timestamp())}&interval=1d&events=history')

# 5. Describing data
st.subheader('Data from {} to {}'.format(start, today))
st.write(df.describe())

# 6. Visualization of closing price vs time chart
st.subheader('Closing Price vs Time')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
fig1.update_layout(title='Closing Price vs Time', xaxis_title='Years', yaxis_title='Closing Price')
st.plotly_chart(fig1)

# 7. Visualization of closing price vs time chart with 100 MA
st.subheader('Closing Price vs Time with 100 MA')
df['100ma'] = df['Close'].rolling(100).mean()
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
fig2.add_trace(go.Scatter(x=df.index, y=df['100ma'], mode='lines', name='100 MA'))
fig2.update_layout(title='Closing Price vs Time with 100 MA', xaxis_title='Years', yaxis_title='Closing Price')
st.plotly_chart(fig2)

# 8. Visualization of closing price vs time chart with 100 MA and 200 MA
st.subheader('Closing Price vs Time with 100 MA and 200 MA')
df['200ma'] = df['Close'].rolling(200).mean()
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
fig3.add_trace(go.Scatter(x=df.index, y=df['100ma'], mode='lines', name='100 MA'))
fig3.add_trace(go.Scatter(x=df.index, y=df['200ma'], mode='lines', name='200 MA'))
fig3.update_layout(title='Closing Price vs Time with 100 MA and 200 MA', xaxis_title='Years', yaxis_title='Closing Price')
st.plotly_chart(fig3)

# 9. Split data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])

# 10. Use MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
train_close = data_training.values
test_close = data_testing.values
data_training_array = scaler.fit_transform(train_close)

# 11. Split data into x_train and y_train
x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# 12. Load model
model = load_model('model.h5')

# 13. Do testing part
past_100_days = data_training.tail(100).values
final_df = np.concatenate((past_100_days, test_close), axis=0)
input_data = scaler.transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)
y_predicted = scaler.inverse_transform(y_predicted)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# 14. Make final graph of prediction vs actual price
st.subheader('Prediction vs Actual')
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=data_testing.index, y=y_test.flatten(), mode='lines', name='Actual Price', line=dict(color='blue')))
fig4.add_trace(go.Scatter(x=data_testing.index, y=y_predicted.flatten(), mode='lines', name='Predicted Price', line=dict(color='red')))
fig4.update_layout(title='Prediction vs Actual', xaxis_title='Time', yaxis_title='Price')
st.plotly_chart(fig4)


