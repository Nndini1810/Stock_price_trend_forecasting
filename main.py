import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import math
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

st.markdown(
    """
    <style>
    .main {
        background-image: url('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQxtKAxs47KvrBB0333tGR_BAJKmCprya2UdA&s');
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

# Function to fetch historical data
def get_historical(quote, start, end):
    try:
        data = yf.download(quote, start=start, end=end, interval='1m' if (end - start).days <= 5 else '1d')
        df = pd.DataFrame(data)
        if not df.empty:
            df.index = pd.to_datetime(df.index)  # Ensure the index is a datetime object
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Function to create the plot
def create_plot(df, quote, interval):
    fig = go.Figure()

    # Determine the color based on price change
    color = 'green' if df['Close'].iloc[-1] > df['Close'].iloc[0] else 'red'
    fill_color = 'rgba(0, 255, 0, 0.2)' if color == 'green' else 'rgba(255, 0, 0, 0.2)'

    # Add the main trace for the close price
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price', line=dict(color=color, width=2)))

    # Add a second trace for the fill area
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Fill Area', fill='tozeroy', fillcolor=fill_color, line=dict(color='rgba(0, 0, 0, 0)'), showlegend=False, hoverinfo='skip'))

    if interval == '1m':
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # hide weekends
                dict(bounds=[20, 10], pattern="hour")  # hide hours outside of 10am-8pm
            ]
        )

    # Configure date formatting for different intervals
    date_format = "%b %d, %Y" if interval != '1m' else "%H:%M"
    fig.update_layout(title=f'{quote} Close Price Over Time',
                      xaxis_title='Date' if interval == '1d' else 'Time',
                      yaxis_title='Price',
                      hovermode='x unified',
                      yaxis=dict(titlefont=dict(size=14)),
                      xaxis=dict(titlefont=dict(size=14), tickformat=date_format),
                      margin=dict(l=40, r=40, t=40, b=40))

    return fig

# Function to get additional data
def get_additional_data(df):
    try:
        # Current day data
        current_day = df.index[-1]
        current_open = df['Open'].loc[current_day]
        current_high = df['High'].loc[current_day]
        current_low = df['Low'].loc[current_day]
        
        # 52-week high and low
        max_52wk_high = df['High'].rolling(window=252).max().dropna().iloc[-1]
        min_52wk_low = df['Low'].rolling(window=252).min().dropna().iloc[-1]
        
        return {
            'Current Open': current_open,
            'Current High': current_high,
            'Current Low': current_low,
            '52wk High': max_52wk_high,
            '52wk Low': min_52wk_low
        }
    except IndexError:
        return None

# ARIMA Algorithm
def arima_algo(df, quote):
    def parser(x):
        return datetime.strptime(x, '%Y-%m-%d')

    def arima_model(train, test):
        history = [x for x in train]
        predictions = list()
        for t in range(len(test)):
            model = ARIMA(history, order=(6,1,0))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
        return predictions

    df['Price'] = df['Close']
    Quantity_date = df[['Price']]
    Quantity_date.index = Quantity_date.index.map(lambda x: parser(str(x)[:10]))
    Quantity_date['Price'] = Quantity_date['Price'].map(lambda x: float(x))
    Quantity_date = Quantity_date.fillna(Quantity_date.bfill())
    
    quantity = Quantity_date.values
    size = int(len(quantity) * 0.80)
    train, test = quantity[0:size], quantity[size:len(quantity)]
    predictions = arima_model(train, test)
    
    # Plot ARIMA results
    fig_arima = go.Figure()
    fig_arima.add_trace(go.Scatter(x=df.index[-len(test):], y=[x[0] for x in test], mode='lines', name='Actual Price'))
    fig_arima.add_trace(go.Scatter(x=df.index[-len(test):], y=predictions, mode='lines', name='Predicted Price'))
    fig_arima.update_layout(title=f'{quote} ARIMA Model Prediction',
                            xaxis_title='Date',
                            yaxis_title='Price',
                            hovermode='x unified')
    
    arima_pred = predictions[-2]
    error_arima = math.sqrt(mean_squared_error([x[0] for x in test], predictions))
    
    return fig_arima, arima_pred, error_arima


# Linear Regression Algorithm
def LIN_REG_ALGO(df, quote):
    forecast_out = 7
    df['Close after n days'] = df['Close'].shift(-forecast_out)
    df_new = df[['Close', 'Close after n days']]
    
    y = np.array(df_new.iloc[:-forecast_out, -1])
    y = np.reshape(y, (-1, 1))
    X = np.array(df_new.iloc[:-forecast_out, 0:-1])
    X_to_be_forecasted = np.array(df_new.iloc[-forecast_out:, 0:-1])
    
    X_train = X[:int(0.8*len(df)), :]
    X_test = X[int(0.8*len(df)):, :]
    y_train = y[:int(0.8*len(df)), :]
    y_test = y[int(0.8*len(df)):, :]
    
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    X_to_be_forecasted = scaler_X.transform(X_to_be_forecasted)

    
    model = LinearRegression(n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_test_pred = model.predict(X_test)
    y_test_pred = y_test_pred * (1.04)


    error_lr = math.sqrt(mean_squared_error(y_test, y_test_pred))

    forecast_set = model.predict(X_to_be_forecasted)
    forecast_set = forecast_set * (1.04)
    mean = forecast_set.mean()
    lr_pred = forecast_set[0, 0]
    
    # Create plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_test.flatten(), mode='lines', name='Actual Price'))
    fig.add_trace(go.Scatter(y=y_test_pred.flatten(), mode='lines', name='Predicted Price'))
    
    fig.update_layout(title='Linear Regression Predictions',
                      xaxis_title='Time',
                      yaxis_title='Price')

    # Display the plot in Streamlit
    st.plotly_chart(fig)
    
    return df, lr_pred, forecast_set, mean, error_lr

# LSTM Algorithm
def LSTM_ALGO(df, quote):
    # Split data into training set and test set
    dataset_train = df.iloc[0:int(0.8*len(df)), :]
    dataset_test = df.iloc[int(0.8*len(df)):, :]
    training_set = dataset_train.iloc[:, 4:5].values

    # Scale the training set
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    # Create data structure with 60 timesteps and 1 output
    X_train = []
    y_train = []
    for i in range(60, len(training_set_scaled)):
        X_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshape the training data
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    # Compile and fit the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    # Prepare test set for predictions
    dataset_total = pd.concat((dataset_train['Close'], dataset_test['Close']), axis=0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)

    X_test = []
    for i in range(60, inputs.shape[0]):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Make predictions on test data
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    # Plot the results
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[-len(dataset_test):], y=dataset_test['Close'], mode='lines', name='Actual Price'))
    fig.add_trace(go.Scatter(x=df.index[-len(dataset_test):], y=predicted_stock_price.flatten(), mode='lines', name='Predicted Price'))
    fig.update_layout(title=f'{quote} LSTM Model Prediction',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      hovermode='x unified')

    # Calculate error
    error_lstm = math.sqrt(mean_squared_error(dataset_test['Close'], predicted_stock_price))

    # Predicted price for next time step
    lstm_pred = predicted_stock_price[-1][0]

    return fig, lstm_pred, error_lstm

# Function to display additional data
def display_additional_data(additional_data):
    #st.write("### Current Day's Data")
    st.write(f"Open: {additional_data['Current Open']}")
    st.write(f"High: {additional_data['Current High']}")
    st.write(f"Low: {additional_data['Current Low']}")
    
    st.write("### 52-Week Data")
    st.write(f"52-Week High: {additional_data['52wk High']}")
    st.write(f"52-Week Low: {additional_data['52wk Low']}")

# Title and description
st.markdown('<h1 class="title">Stock Price Trend Forecasting</h1>', unsafe_allow_html=True)
st.write("Welcome to the Stock Price Trend Forecasting app. Please enter the stock symbol and date range to visualize the historical data and forecast future trends.")

# Sidebar for inputs
with st.sidebar:
    quote = st.text_input("Enter Stock Symbol (e.g : TCS,AAPL,MSFT)")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2023-12-31"))
    end_date = st.date_input('End Date', value=pd.datetime.today())


# Fetch data button
if st.sidebar.button("Fetch Data"):
    with st.spinner('Fetching data...'):
        df = get_historical(quote, start_date, end_date)
        if not df.empty:
            st.write(f"### Historical Data for {quote}")
            st.dataframe(df)
            
            # Create plot for historical data
            interval = '1m' if (end_date - start_date).days <= 5 else '1d'
            fig = create_plot(df, quote, interval)
            st.plotly_chart(fig)
            
            # Display additional data
            additional_data = get_additional_data(df)
            display_additional_data(additional_data)

            # Calculate forecast data
            forecast_days = 7

            # ARIMA
            st.write("## ARIMA Forecast")
            fig_arima, arima_pred, error_arima = arima_algo(df, quote)
            st.plotly_chart(fig_arima)
            st.write(f"Next time step forecasted price: {arima_pred}")
            st.write(f"Error: {error_arima}")

            # LSTM
            st.write("## LSTM Forecast")
            fig_lstm, lstm_pred, error_lstm = LSTM_ALGO(df, quote)
            st.plotly_chart(fig_lstm)
            st.write(f"Next time step forecasted price: {lstm_pred}")
            st.write(f"Error: {error_lstm}")

            # Linear Regression
            st.write("## Linear Regression Forecast")
            df, lr_pred, forecast_set, mean, error_lr = LIN_REG_ALGO(df, quote)
            #st.write(f"Next 7 days forecasted prices: {forecast_set.flatten()}")
            st.write(f"Mean forecasted price: {mean}")
            st.write(f"Error: {error_lr}")

            # Prepare DataFrame for next 7 days forecast
            future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Forecasted Price': forecast_set.flatten()
            })

            # Display forecast DataFrame
            st.write("### Forecast for Next 7 Days")
            st.dataframe(forecast_df)

            # Plot forecast for the next 7 days
            future_dates_plot = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(x=future_dates_plot, y=forecast_set.flatten(), mode='lines+markers', name='Forecasted Price'))
            fig_forecast.update_layout(title='Next 7 Days Forecast',
                                       xaxis_title='Date',
                                       yaxis_title='Price',
                                       hovermode='x unified')
            st.plotly_chart(fig_forecast)

            # Recommendation
            if forecast_set[-1] > forecast_set[0]:
                st.write("### Recommendation")
                st.write("The forecasted trend is increasing. Consider holding or buying the stock.")
            else:
                st.write("### Recommendation")
                st.write("The forecasted trend is decreasing. Consider selling the stock.")

        else:
            st.error("Failed to fetch data. Please check the stock symbol and date range.")
