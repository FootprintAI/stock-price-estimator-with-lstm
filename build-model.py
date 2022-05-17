import argparse
import pandas as pd
import numpy as np

# For reading stock data from yahoo
from pandas_datareader.data import DataReader
import yfinance as yf

# For time stamps
from datetime import datetime

from tensorflow.keras import callbacks as keras_callbacks
class LossAndErrorPrintingCallback(keras_callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print("epoch={}".format(epoch))
        print("Training-Loss={:7.6f}".format(logs["loss"]))
        print("Training-MAE={:7.6f}".format(logs["mean_absolute_error"]))
        print("Validation-Loss={:7.6f}".format(logs["val_loss"]))
        print("Validation-MAE={:7.6f}".format(logs["val_mean_absolute_error"]))

def build_model(n: int, epochs: int, X_train, y_train, X_val, y_val, optimizer: str):
    # Now we build a tensorflow model with LSTM
    # the network is not something fancy, it is just a common way to build the model
    # And you can also find a better model online.
    
    # Note the input tensor size should have equal length with the windows size.
    # In this example, we use windows size (n=3), so the input tensor is layers.Input((3, 1)
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras import layers
    
    model = Sequential([layers.Input((n, 1)),
                        layers.LSTM(64),
                        layers.Dense(32, activation='relu'),
                        layers.Dense(32, activation='relu'),
                        layers.Dense(1)])
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error'],
                  )
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, callbacks=[LossAndErrorPrintingCallback()])

# convert dateframe into multiple sequences of array, each sequence is with size $n
def df_to_windowed_df(dataframe, n=3):
    # As we mentioned earier, for LSTM we use a sequence of events (or data) to predict what is the
    # stock price next, so we need to segment a whole year data into multiple sequences with a predefined
    # time-window.
    # It means we are grouping stock prices of three consecutive day into one sequence if n is 3.

    (num_columns, num_rows) = dataframe.shape

    dates = []
    X, Y = [], []

    for index in range(n, num_columns):
        # for example, we are grouping each the role illustrated below
        # df_end_date = 2021-05-13
        # df_end_value = sp(2021-05-13)
        # df_seuqnece = [sp(2021-05-10), sp(2021-05-11), sp(2021-05-12)]

        df_end_date = dataframe.iloc[index]['Date']
        df_end_value = dataframe.iloc[index]['Close']
        df_seuqnece = dataframe.iloc[index-n:index]
        df_values = df_seuqnece['Close'].to_numpy()
        x, y = df_values, df_end_value

        dates.append(df_end_date)
        X.append(x)
        Y.append(y)

    ret_df = pd.DataFrame({})
    ret_df['Target Date'] = dates

    X = np.array(X)
    for i in range(0, n):
        ret_df[f'Target-{n-i}'] = X[:, i]

    ret_df['Target'] = Y

    return ret_df

def windowed_df_to_date_X_y(windowed_dataframe):
    df_as_np = windowed_dataframe.to_numpy()
  
    dates = df_as_np[:, 0]
  
    middle_matrix = df_as_np[:, 1:-1]
    X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))
  
    Y = df_as_np[:, -1]
  
    return dates, X.astype(np.float32), Y.astype(np.float32)


# fetch_data returns a dateframe including columns Date and Close
def fetch_data(last_n_year: int, symbol: str):
    end = datetime.now()
    start = datetime(end.year - last_n_year, end.month, end.day) # get the whole year stock price history

    symbolWithStockPrice = yf.download(symbol, start, end)
    symbolWithStockPrice['Date'] = symbolWithStockPrice.index # add date column
    return symbolWithStockPrice[['Date', 'Close']]


def run(args):
    df = fetch_data(args.lastnyears, args.symbol)
    windowed_df = df_to_windowed_df(df, args.wsize)
    dates, X, y = windowed_df_to_date_X_y(windowed_df)
    q_80 = int(len(dates) * .8)
    q_90 = int(len(dates) * .9)
    
    dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
    dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
    # dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]
    build_model(args.wsize, args.epochs, X_train, y_train, X_val, y_val,
            args.optimizer)

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="stock price estimator with lstm",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--optimizer', type=str, default='adam', dest='optimizer',
                        help='optimizer: Adadelta/Adagrad/Adam/Adamax/Ftrl/SGD/RMSprop')
    parser.add_argument('--epochs', type=int, default=100, dest='epochs', help='epoch: number of iterations run')
    parser.add_argument('--lastnyears', type=int, default=1, dest='lastnyears', help='last n years used')
    parser.add_argument('--symbol', type=str, default='MSFT', dest='symbol', help='symbol: ticket, default: MSFT')
    parser.add_argument('--n', type=int, default=3, dest='wsize', help='n: length of windows')
    args = parser.parse_args()
    run(args)

