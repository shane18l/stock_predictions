import yfinance as yf
import pandas as pd


data = yf.Ticker('AAPL')

stock = data.history(period="max")
print(stock)

stock.plot.line(y="Close", use_index=True)