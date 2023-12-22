import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import matplotlib.dates as mpl_dates
import datetime
import yfinance
from datetime import datetime
import warnings

date = datetime.today().strftime('%Y-%m-%d')
name = "BTC-USD"
ticker = yfinance.Ticker(name)
df = ticker.history(interval="1d",start="2021-03-01",end = date )
df["Date"] = pd.to_datetime(df.index)
df['Date'] = df['Date'].apply(mpl_dates.date2num)
df = df.loc[:,['Date', 'Open', 'High', 'Low', 'Close']]

fig, (ax) = plt.subplots()
date_format = mpl_dates.DateFormatter('%d %b %Y')
ax.plot(df["Date"], df["Close"], label = "Closing Price")
ax.legend(loc = "best")
ax.xaxis.set_major_formatter(date_format)
fig.autofmt_xdate() 
plt.show()

#plot the rolling mean and rolling std
rolling_mean = df["Close"].rolling(window = 12).mean()
rolling_std = df["Close"].rolling(window = 12).std()
plt.plot(df["Close"], color = 'blue', label = 'Original')
plt.plot(rolling_mean, color = 'red', label = 'Rolling Mean')
plt.plot(rolling_std, color = 'black', label = 'Rolling Std')
plt.legend(loc = 'best')
plt.title('Rolling Mean & Rolling Standard Deviation')
plt.show()

#Accept or reject the H0 hypothesis. H0 hypothesis = Time series is not stationary
result = adfuller(df['Close'])
print('ADF Statistic: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))

#timeseries is our dataset in a dataframe form. The function will look for the "Close"
#label and work with this data
def get_stationarity(timeseries):
    
    # rolling statistics
    rolling_mean = timeseries.rolling(window=12).mean()
    rolling_std = timeseries.rolling(window=12).std()
    
    # rolling statistics plot
    original = plt.plot(timeseries["Close"], color='blue', label='Original')
    mean = plt.plot(rolling_mean["Close"], color='red', label='Rolling Mean')
    std = plt.plot(rolling_std["Close"], color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    # Dickey–Fuller test:
    result = adfuller(timeseries["Close"])
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))
    return(result[1])

#This function identifies the best option to make a series stationary
#dataframe = the values in dataframe form
#case = 1, 2, 3, identifies how the data arae changing 

def get_best_stationarity(dataframe, case):
  if case == 1:
    df_log = np.log(dataframe)
    rolling_mean = df_log.rolling(window=12).mean()
    df_log_minus_mean = df_log - rolling_mean
    df_log_minus_mean.dropna(inplace=True) #remove the NA values
    df_log_minus_mean
    p = get_stationarity(df_log_minus_mean)
  elif case == 2:
    df_log = np.log(dataframe)
    rolling_mean_exp_decay = df_log.ewm(halflife=12, min_periods=0, adjust=True).mean()
    df_log_exp_decay = df_log - rolling_mean_exp_decay
    df_log_exp_decay.dropna(inplace=True)
    p = get_stationarity(df_log_exp_decay)
  elif case == 3:
    rolling_mean_exp_decay = df.ewm(halflife=12, min_periods=0, adjust=True).mean()
    df_log_exp_decay = dataframe - rolling_mean_exp_decay
    df_log_exp_decay.dropna(inplace=True)
    p = get_stationarity(df_log_exp_decay)
  elif case == 4:
    df_log_shift = df - df.shift()
    df_log_shift.dropna(inplace=True)
    p = get_stationarity(df_log_shift)
  elif case == 5:
    df_log = np.log(df) 
    df_log_shift = df_log - df_log.shift()
    df_log_shift.dropna(inplace=True)
    p = get_stationarity(df_log_shift)
  return(p)

p_val = list()
for i in range(1, 6):
  p_val.append(get_best_stationarity(df, i))

df_log = np.log(df) 
df_log_shift = df_log - df_log.shift()
df_log_shift.dropna(inplace=True)

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(df_log_shift["Close"])
plot_pacf(df_log_shift["Close"])

#decomposition = seasonal_decompose(df_log_shift["Close"]) 
model = sm.tsa.arima.ARIMA(df_log_shift["Close"], order=(2, 1, 0))
results = model.fit()
df_log_shift["predicted"] = results.fittedvalues
RSS = sum((df_log_shift["predicted"] - df_log_shift["Close"])**2)
rounded_RSS = round(RSS, 3)
plt.title(rounded_RSS)
plt.plot(df_log_shift["Close"])
plt.plot(results.fittedvalues, color='red')

length = int(len(df_log) * 0.66)
train_arima = df_log["Close"][: length]
test_arima = df_log["Close"][length: ]
history = [x for x in train_arima]
predictions = list()
originals = list()
error_list = list()


for i in range(len(test_arima)):
  warnings.filterwarnings("ignore")
  model = sm.tsa.arima.ARIMA(history, order=(2, 1, 0))
  results = model.fit()
  output = results.forecast()
  pred_value = output[0]  
  pred_value = np.exp(pred_value)
  original_value = test_arima[i]
  history.append(original_value)
  original_value = np.exp(original_value)
  error = ((abs(pred_value - original_value)) / original_value) * 100
  error_list.append(error)
  print('predicted = %f,   expected = %f,   error = %f ' % (pred_value, original_value, error), '%')
  predictions.append(float(pred_value))
  originals.append(float(original_value))


