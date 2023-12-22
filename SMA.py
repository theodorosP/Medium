import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mpl_dates
import datetime
import yfinance
import tensorflow as tf
from datetime import datetime

def get_data(Name, start_date, end_date):
  name = Name
  ticker = yfinance.Ticker(name)
  df = ticker.history(interval="1d",start= start_date, end = end_date )
  df["Date"] = pd.to_datetime(df.index)
  df['Date'] = df['Date'].apply(mpl_dates.date2num)
  df = df.loc[:,['Date', 'Open', 'High', 'Low', 'Close']]
  return df

#this codes gives the snimle moving average
#dataframe is the dataframe with our data.
#window is the lag. How many days we look back
#title is the name of column in the dataframe of the calculated quantity

def get_SMA(dataframe, window, title):
  N = len(dataframe)
  average_values = list()
  for i in range(window, N):
    average_values.append(np.mean(dataframe["Close"][i - window : i]))
  df[title] = np.nan

  for i in range(window, N):
    dataframe[title][i] = average_values[i - window]
  return dataframe

start_date = "2021-03-01"
end_date = datetime.today().strftime('%Y-%m-%d') #data till today
df = get_data("ETE.AT", start_date, end_date)

fig, (ax) = plt.subplots()
date_format = mpl_dates.DateFormatter('%d %b %Y')
ax.plot(df["Date"], df["Close"], label = "Closing Price")
ax.legend(loc = "best")
ax.xaxis.set_major_formatter(date_format)
fig.autofmt_xdate() 
plt.show()

df = get_SMA(df, 10, "SMA_10")
df = get_SMA(df, 20, "SMA_20")

def get_signals(dataframe):
  dataframe["bull"] = np.nan
  dataframe["bear"] = np.nan
  for i in range(0, len(dataframe) - 1):
    if abs(dataframe["SMA_10"][i] - dataframe["SMA_20"][i]) < 0.015  and dataframe["SMA_10"][i + 1] > dataframe["SMA_20"][i + 1]:  # dataframe["SMA_20"][i]:
      dataframe["bull"][i] = dataframe["Close"][i]
      print(dataframe["Close"][i],  dataframe["SMA_10"][i])
      #break
    else:
      dataframe["bear"][i] = dataframe["SMA_20"][i]
  return dataframe

dataset = df.to_numpy()
dataset

plt.figure(figsize = (18,9))

plt.plot(dataset[:, 4], label = "Closing prices")
plt.plot(dataset[:, 5], label = "SMA 10", color = "red")
plt.plot(dataset[:, 6] , label = "SMA 20")
plt.plot(dataset[:, 9], marker = "^", color = "green", label = "Buy")
plt.xlabel("Date")
plt.ylabel("NBG stock (€)")
plt.legend(fontsize=18)
plt.show()


