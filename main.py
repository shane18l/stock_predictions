import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score


data = yf.Ticker('AAPL')

stock = data.history(period="max")
print(stock)

# stock.plot.line(y="Close", use_index=True)
# plt.show()

# Removing columns
del stock["Dividends"]
del stock["Stock Splits"]

# Add columns for the Closing price Tomorrow, and if it's greater than today's
stock["Tomorrow"] = stock["Close"].shift(-1)
stock["Target"] = (stock["Tomorrow"] > stock["Close"]).astype(int)


stock = stock.loc["2000-01-02":].copy()


# Random Forest Model
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, 
        random_state =1)

train = stock.iloc[:-100]
test = stock.iloc[-100:]

# Train the model on our train data
predictor_factors = ["Close", "Volume", "High", "Low"]






# Function that builds a prediction
def predict(train, test, predictor_factors, model):
    model.fit(train[predictor_factors], train["Target"])
    # Make predictions
    predictions = model.predict_proba(test[predictor_factors])[:,1]
    predictions[predictions >= 0.6] = 1
    predictions[predictions < 0.6] = 0
    predictions = pd.Series(predictions, index = test.index, name="Predictions")
    combined = pd.concat([test["Target"], predictions], axis = 1)
    return combined

def backtest(data, model, predictor_factors, start=2500, step = 250):
    all_preds = []

    for x in range(start, data.shape[0], step):
        train = data.iloc[0:x].copy()
        test = data.iloc[x:x+step].copy()
        predictions = predict(train, test, predictor_factors, model)
        all_preds.append(predictions)
    return pd.concat(all_preds)

predictions = backtest(stock, model, predictor_factors)
print(predictions["Predictions"].value_counts())
print(precision_score(predictions["Target"], predictions["Predictions"]))
print(predictions["Target"].value_counts() / predictions.shape[0])

# Adding more predictor factors to enhance the model
time_intervals = [2,5,60,250,1000]
new_predictor_factors = []

for interval in time_intervals:
    # Calculate the rolling average for each time interval
    rolling_avg = stock.rolling(interval).mean()
    ratio_column = f"Ratio_Close_{interval}"
    # Column that displays the ratio between the Close price and rolling avg
    stock[ratio_column] = stock["Close"] / rolling_avg["Close"]

    trend_column = f"Trend_{interval}"
    stock[trend_column] = stock.shift(1).rolling(interval).sum()["Target"]

    new_predictor_factors += [ratio_column, trend_column]

stock = stock.dropna()
