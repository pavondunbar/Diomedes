import yfinance as yf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def fetch_data(stock_symbol, start_date, end_date, market_condition):
    df = yf.download(stock_symbol, start=start_date, end=end_date)
    df['Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)
    
    # Adjust based on market condition
    if market_condition == "Bull Market":
        df['Return'] *= 1.1
    elif market_condition == "Bear Market":
        df['Return'] *= 0.9

    return df[['Open', 'High', 'Low', 'Close', 'Volume', 'Return']]

# New function to create lagged features
def create_lagged_features(data, lag=5):
    for i in range(1, lag + 1):
        data[f'Lagged_Return_{i}'] = data['Return'].shift(i)
    data.dropna(inplace=True)
    return data

def build_and_train_model(data, future_days=30, use_lstm=False):
    # Create lagged features
    data = create_lagged_features(data)
    
    # Prepare data
    X = data.drop('Return', axis=1).values
    y = data['Return'].shift(-future_days).values[:-future_days]

    X_train, X_test, y_train, y_test = train_test_split(X[:-future_days], y, test_size=0.2, random_state=42)

    # Normalize data
    scaler_X = MinMaxScaler()
    scaler_Y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    y_train = scaler_Y.fit_transform(y_train.reshape(-1, 1))
    y_test = scaler_Y.transform(y_test.reshape(-1, 1))

    # Build model
    model = Sequential()
    if use_lstm:
        model.add(LSTM(128, input_shape=(X_train.shape[1], 1), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(32))
    else:
        model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer='adam')
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)

    model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), verbose=0,
              callbacks=[early_stop, reduce_lr])

    return model, scaler_X, scaler_Y

def get_allocation_based_on_factors(risk_tolerance, age, goal, market_condition, predicted_return, 
                                    income_level, current_savings, investment_horizon, 
                                    investment_experience, purpose_of_investment, priority):

    allocations = {
        "Stocks": 0.4,
        "Bonds": 0.3,
        "Treasury Notes": 0.1,
        "Real Estate": 0.1,
        "Gold": 0.05,
        "Cash": 0.05
    }
    reasons = []

    # Determine income category based on income_level
    if income_level < 50000:
        income_category = "Low"
    elif 50000 <= income_level <= 200000:
        income_category = "Medium"
    else:
        income_category = "High"

    # Risk tolerance adjustments
    if risk_tolerance == "Conservative":
        allocations["Stocks"] -= 0.15
        allocations["Bonds"] += 0.15
        reasons.append("Given your conservative risk tolerance, stocks allocation was reduced for less volatility.")
    elif risk_tolerance == "Aggressive":
        allocations["Stocks"] += 0.15
        allocations["Bonds"] -= 0.15
        reasons.append("Given your aggressive risk tolerance, stocks allocation was increased for higher potential returns.")

    # Age adjustments
    if age < 35:
        allocations["Stocks"] += 0.05
        allocations["Bonds"] -= 0.05
        reasons.append("As you're younger than 35, stocks allocation was increased to capitalize on long-term growth.")
    elif age > 55:
        allocations["Stocks"] -= 0.05
        allocations["Bonds"] += 0.05
        reasons.append("As you're older than 55, bonds allocation was increased for more stability as retirement approaches.")

    # Goal adjustments
    if goal == "Short-term":
        allocations["Cash"] += 0.1
        allocations["Stocks"] -= 0.1
        reasons.append("With a short-term goal, liquidity is important so cash allocation was increased.")
    else:
        allocations["Real Estate"] += 0.05
        allocations["Treasury Notes"] += 0.05
        reasons.append("With a long-term goal, allocation was increased in assets like real estate and treasury notes for diversification.")

    # Market condition adjustments
    if predicted_return > 0.05:
        allocations["Stocks"] += 0.1
        allocations["Bonds"] -= 0.1
        reasons.append("Given a positive market prediction, stocks allocation was increased for potential growth.")
    else:
        allocations["Stocks"] -= 0.1
        allocations["Bonds"] += 0.1
        reasons.append("Given a less optimistic market prediction, stocks allocation was reduced to protect capital.")

    # Income level adjustments
    if income_category == "Low":
        allocations["Stocks"] -= 0.05
        allocations["Bonds"] += 0.05
        reasons.append("With a lower income level, bonds allocation was increased for safety.")
    elif income_category == "High":
        allocations["Stocks"] += 0.05
        allocations["Real Estate"] += 0.05
        reasons.append("With a higher income level, allocations in stocks and real estate were increased to capitalize on growth and opportunities.")

    # Investment experience adjustments
    if investment_experience == "Novice":
        allocations["Treasury Notes"] += 0.05
        allocations["Stocks"] -= 0.05
        reasons.append("Being new to investments, allocation was increased in safer assets like treasury notes.")
    elif investment_experience == "Expert":
        allocations["Stocks"] += 0.05
        allocations["Bonds"] -= 0.05
        reasons.append("With expert experience, you can manage risk better, hence increased allocation in stocks.")

    # Purpose of investment adjustments
    if purpose_of_investment == "Buying House":
        allocations["Real Estate"] += 0.05
        allocations["Stocks"] -= 0.05
        reasons.append("Investing for buying a house increases allocation in real estate to align with your goal.")
    elif purpose_of_investment == "Retirement":
        allocations["Bonds"] += 0.05
        allocations["Real Estate"] -= 0.05
        reasons.append("Investing for retirement has led to increasing bonds for consistent returns.")

    # Current savings adjustments
    if current_savings < 5000:
        allocations["Cash"] += 0.05
        allocations["Bonds"] -= 0.05
        reasons.append("With lower savings, it's good to keep more cash at hand for emergencies.")
    elif current_savings > 100000:
        allocations["Real Estate"] += 0.05
        allocations["Cash"] -= 0.05
        reasons.append("With higher savings, you can afford to invest more in real estate.")

    # Investment horizon adjustments
    if investment_horizon < 5:
        allocations["Stocks"] -= 0.05
        allocations["Cash"] += 0.05
        reasons.append("With a shorter investment horizon, more liquidity is kept by increasing cash allocation.")
    elif investment_horizon > 20:
        allocations["Stocks"] += 0.05
        allocations["Bonds"] -= 0.05
        reasons.append("For a long-term horizon, stocks offer the best growth potential, hence increased allocation.")

    # Priority adjustments
    if priority == "Safety":
        allocations["Bonds"] += 0.05
        allocations["Stocks"] -= 0.05
        reasons.append("Prioritizing safety leads to higher allocation in bonds.")
    elif priority == "Growth":
        allocations["Stocks"] += 0.05
        allocations["Bonds"] -= 0.05
        reasons.append("Prioritizing growth increases the allocation in stocks for higher returns.")

    return allocations, reasons

if __name__ == "__main__":
    print("Welcome to Diomedes, your personal Financial Advisor AI. Before I can decide how to allocate your investment, please answer a few questions:")

    risk_tolerance = input("What's your risk tolerance (Conservative, Balanced, Aggressive)? ")
    age = int(input("How old are you? "))
    income_level = float(input("What is your annual income? "))
    current_savings = float(input("How much do you currently have in savings? "))
    investment_horizon = int(input("For how many years do you plan to invest this money? "))
    investment_experience = input("How would you describe your investment experience (Novice, Intermediate, Expert)? ")
    purpose_of_investment = input("What's the main purpose of this investment (Retirement, Buying House, Growth, Other)? ")
    priority = input("What's your primary investment priority (Capital Preservation, Growth)? ")
    goal = input("Is your investment goal Short-term or Long-term? ")
    market_condition = input("Current market condition (Bull Market or Bear Market)? ")
    amount = float(input("Enter the dollar amount you'd like to invest: "))

    # Fetch data here
    data = fetch_data("^GSPC", "2010-01-01", "2023-01-01", market_condition)

    # Build and train two models: one with LSTM, another without
    model_1, scaler_X_1, scaler_Y_1 = build_and_train_model(data, use_lstm=False)
    model_2, scaler_X_2, scaler_Y_2 = build_and_train_model(data, use_lstm=True)

    latest_data = scaler_X_1.transform(data.drop('Return', axis=1).iloc[-1:].values)
    predicted_return_1 = scaler_Y_1.inverse_transform(model_1.predict(latest_data))[0][0]
    predicted_return_2 = scaler_Y_2.inverse_transform(model_2.predict(latest_data))[0][0]

    # Average the predictions of the two models
    predicted_return = (predicted_return_1 + predicted_return_2) / 2

    data = fetch_data("^GSPC", "2010-01-01", "2023-01-01", market_condition)
    model, scaler_X, scaler_Y = build_and_train_model(data)

    latest_data = scaler_X.transform(data.drop('Return', axis=1).iloc[-1:].values)
    predicted_return = scaler_Y.inverse_transform(model.predict(latest_data))[0][0]

    # Adjust get_allocation_based_on_factors function to consider new inputs
    allocation_percentages, allocation_reasons = get_allocation_based_on_factors(risk_tolerance, age, goal, market_condition, predicted_return, 
                                                              income_level, current_savings, investment_horizon, investment_experience, 
                                                              purpose_of_investment, priority)

    suggested_allocations = {}
    for investment, percentage in allocation_percentages.items():
        suggested_allocations[investment] = round(amount * percentage, 2)

    print("\nThank you. Based on your input, I, Diomedes, recommend the following investment allocation and my reason(s) below:")
    for investment, allocation in suggested_allocations.items():
        print(f"You should invest ${allocation:.2f} into {investment}.")
 
    print("\nReasons for these allocations:")
    for reason in allocation_reasons:
        print(f"- {reason}")
 
    print("\nDisclaimer:")
    print("Diomedes is suggesting the above to diversify your investment. However, you should still perform due diligence and research before implementing Diomedes's recommendations. You agree also to hold Diomedes harmless against any loss of funds as a result of implementation of its recommendation. Diomedes's recommendations are for information only. Investments are not insured by the FDIC and may lose value.")
