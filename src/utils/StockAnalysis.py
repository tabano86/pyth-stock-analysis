# import os
# from importlib import reload
#
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from keras import backend as K
# from keras.layers.core import Dense
# from keras.layers.recurrent import LSTM
# from keras.models import Sequential
# from pandas.plotting import register_matplotlib_converters
# from sklearn.metrics import r2_score
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.python.keras.callbacks import EarlyStopping
#
# from src.utils.StockRestService import get_json_data
#
# register_matplotlib_converters()
# # disable chained assignments
# pd.options.mode.chained_assignment = None
#
#
# def configure_regressor() -> Sequential:
#     regr = Sequential()
#     regr.add(Dense(12, input_dim=1, activation='relu'))
#     regr.add(Dense(1))
#     regr.compile(loss='mean_squared_error', optimizer='adam')
#     return regr
#
#
# def plot(real_dataset: [], predicted_data_set: [], ticker: str):
#     plt.plot(real_dataset, color='black', label=ticker + ' Stock Price')
#     plt.plot(predicted_data_set, color='green', label='Predicted ' + ticker + ' Stock Price')
#     plt.title(ticker + ' Stock Price Prediction')
#     plt.xlabel('Date')
#     plt.xticks(rotation='60')
#     plt.autoscale()
#     plt.grid(b=True, which='both')
#     plt.subplots_adjust(bottom=0.23)
#     plt.ylabel(ticker + ' Stock Price')
#     plt.legend()
#     plt.show()
#
#
# def set_keras_backend(backend):
#     if K.backend() != backend:
#         os.environ['KERAS_BACKEND'] = backend
#         reload(K)
#         assert K.backend() == backend
#
#
# def run_analysis(ticker: str = 'MSFT', epochs_num: int = 100, forecast_days: int = 30):
#     response = get_json_data(ticker)
#     df = response['close']
#     split_date = np.datetime64('today') - np.timedelta64(90, 'D')
#     train = pd.DataFrame(df.loc[:split_date])
#     test = pd.DataFrame(df.loc[split_date:])
#     scaler = MinMaxScaler(feature_range=(-1, 1))
#     train_sc = scaler.fit_transform(train)
#     test_sc = scaler.transform(test)
#     X_train = train_sc[:-1]
#     y_train = train_sc[1:]
#     X_test = test_sc[:-1]
#     y_test = test_sc[1:]
#     nm_model = configure_regressor()
#     early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
#     history = nm_model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=1, callbacks=[early_stop], shuffle=False)
#
#     y_pred_test_nn = nm_model.predict(X_test)
#     y_train_pred_nn = nm_model.predict(X_train)
#     print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred_nn)))
#     print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test_nn)))
#
#     train_sc_df = pd.DataFrame(train_sc, columns=['Y'], index=train.index)
#     test_sc_df = pd.DataFrame(test_sc, columns=['Y'], index=test.index)
#
#     for s in range(1, 2):
#         train_sc_df['X_{}'.format(s)] = train_sc_df['Y'].shift(s)
#         test_sc_df['X_{}'.format(s)] = test_sc_df['Y'].shift(s)
#
#     X_train = train_sc_df.dropna().drop('Y', axis=1)
#     y_train = train_sc_df.dropna().drop('X_1', axis=1)
#
#     X_test = test_sc_df.dropna().drop('Y', axis=1)
#     y_test = test_sc_df.dropna().drop('X_1', axis=1)
#
#     X_train = X_train.values
#     y_train = y_train.values
#
#     X_test = X_test.values
#     y_test = y_test.values
#
#     X_train_lmse = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
#     X_test_lmse = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
#
#     print('Train shape: ', X_train_lmse.shape)
#     print('Test shape: ', X_test_lmse.shape)
#
#     lstm_model = Sequential()
#     lstm_model.add(
#         LSTM(7, input_shape=(1, X_train_lmse.shape[1]), activation='relu', kernel_initializer='lecun_uniform',
#              return_sequences=False))
#     lstm_model.add(Dense(1))
#     lstm_model.compile(loss='mean_squared_error', optimizer='adam')
#
#     early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
#     lstm_model.fit(X_train_lmse, y_train, epochs=1, batch_size=1, verbose=1, shuffle=False,
#                    callbacks=[early_stop])
#
#     y_pred_test_lstm = lstm_model.predict(X_test_lmse)
#     y_train_pred_lstm = lstm_model.predict(X_train_lmse)
#     print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred_lstm)))
#     print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test_lstm)))
#     plt.figure(figsize=(10, 6))
#     plt.plot(y_test, label='True')
#     plt.plot(y_pred_test_lstm, label='LSTM')
#     plt.title("LSTM's Prediction")
#     plt.xlabel('Observation')
#     plt.ylabel('Adj Close scaled')
#     plt.legend()
#     plt.show()
#     #
#     #
#     #
#     #
#     #
#     # # normalize data
#     # scaler = MinMaxScaler(feature_range=(0, 1))
#     # df_close = scaler.fit_transform(df)
#     #
#     # # split data into train and test
#     # train_size = int(len(df_close) * 0.7)
#     # test_size = len(df_close) - train_size
#     #
#     # # df_train, df_test = df_close[0:train_size, :], df_close[train_size:len(df_close), :]
#     #
#     # df_train = df.loc[:split_date]
#     # df_test = df.loc[split_date:]
#     #
#     # scaler = MinMaxScaler(feature_range=(-1, 1))
#     # train_sc = scaler.fit_transform(df_train)
#     # test_sc = scaler.transform(df_test)
#     #
#     # X_train_lmse = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
#     # X_test_lmse = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
#     # # need to now convert the data into time series looking back over a
#     # # period of days...e.g. use last 7 days to predict price
#     # # def create_ts(ds, input_series):
#     # #     x, y = [], []
#     # #     for i in range(len(ds) - input_series - 1):
#     # #         item = ds[i:(i + input_series), 0]
#     # #         x.append(item)
#     # #         y.append(ds[i + input_series, 0])
#     # #     return np.array(x), np.array(y)
#     #
#     # series = 7
#     #
#     # train_x, train_y = create_ts(df_train, series)
#     # test_x, test_y = create_ts(df_test, series)
#     # print(train_x)
#     # print(train_x[0])
#     #
#     # # reshape into  LSTM format - samples, steps, features
#     # train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
#     # test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
#     #
#     # # build the model
#     # model = Sequential()
#     # model.add(LSTM(4, input_shape=(series, 1)))
#     # model.add(Dense(1))
#     # model.compile(loss='mse', optimizer='adam')
#     # # fit the model
#     # model.fit(train_x, train_y, epochs=10, batch_size=32)
#     #
#     # # test this model out
#     # train_predictions = model.predict(train_x)
#     # test_predictions = model.predict(test_x)
#     # # unscale predictions
#     # train_predictions = scaler.inverse_transform(train_predictions)
#     # test_predictions = scaler.inverse_transform(test_predictions)
#     # train_y = scaler.inverse_transform([train_y])
#     # test_y = scaler.inverse_transform([test_y])
#     #
#     # # lets calculate the root mean squared error
#     # train_score = math.sqrt(mean_squared_error(train_y[0], train_predictions[:, 0]))
#     # test_score = math.sqrt(mean_squared_error(test_y[0], test_predictions[:, 0]))
#     # print('Train score: %.2f rmse', train_score)
#     # print('Test score: %.2f rmse', test_score)
#
#     # lets plot the predictions on a graph and see how well it did
#     # train_plot = np.empty_like(df_close)
#     # train_plot[:, :] = np.nan
#     # train_plot[series:len(train_predictions) + series, :] = train_predictions
#     #
#     # test_plot = np.empty_like(df_close)
#     # test_plot[:, :] = np.nan
#     # test_plot[len(train_predictions) + (series * 2) + 1:len(df_close + 30) - 1, :] = test_predictions
#     # test_plot = pd.DataFrame(test_plot, columns=['close'])
#     # train_plot = pd.DataFrame(train_plot, columns=['close'])
#     # test_plot['date'] = response.index
#     # train_plot['date'] = response.index
#     # test_plot = test_plot.set_index(['date'], drop=True)
#     # train_plot = train_plot.set_index(['date'], drop=True)
#     # # plot on graph
#     # for i in range(0, len(df)):
#     #     test_plot['date'][i] = response.index[i]
#     #     train_plot['date'][i] = response.index[i]
#
#     # plt.plot(scaler.inverse_transform(df_close), label='inv trans')
#     # plt.plot(train_plot, label='train')
#     # plt.plot(test_plot, label='test')
#     # plt.plot(response['close'], label='act')
#     # plt.legend()
#     # plt.show()
#     # training_set
#     # sc = MinMaxScaler(feature_range=(0, 1))
#     # train_sc = sc.fit_transform(training_set)
#     # #
#     # train_sc_df = pd.DataFrame(train_sc, columns=['close'], index=train.index)
#     #
#     # for s in range(1, 2):
#     #     train_sc_df['X_{}'.format(s)] = train_sc_df['y'].shift(s)
#     #
#     # X_train = train_sc_df.dropna().drop('y', axis=1)
#     # y_train = train_sc_df.dropna().drop('X_1', axis=1)
#     #
#     # X_train = X_train.values
#     # y_train = y_train.values
#     #
#     # X_train_lmse = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
#     #
#     # # # create training set
#     # # training_set = df
#     # # sc = MinMaxScaler(feature_range=(0, 1))
#     # # training_set_scaled = sc.fit_transform(training_set)
#     # # X_train = []
#     # # y_train = []
#     # # for i in range(60, len(df)):
#     # #     X_train.append(training_set_scaled[i - 60:i, 0])
#     # #     y_train.append(training_set_scaled[i, 0])
#     # # X_train, y_train = np.array(X_train), np.array(y_train)
#     # # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#     # # regressor = configure_regressor(X_train)
#     # # r = regressor.fit(X_train, y_train, epochs=epochs_num, batch_size=32)
#     #
#     # # Predicting
#     # # dataset_total = pd.concat((training_set['close'], df['close']), axis=0)
#     # # inputs = dataset_total[len(dataset_total) - len(df) - 60:].values
#     # # inputs = inputs.reshape(-1, 1)
#     # # inputs = sc.transform(inputs)
#     # # X_test = []
#     # # for i in range(60, 60 + forecast_days):
#     # #     X_test.append(inputs[i - 60:i, 0])
#     # # X_test = np.array(X_test)
#     # # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#     # predicted_stock_price = regressor.predict(X_train_lmse)
#     # predicted_stock_price = pd.DataFrame(sc.inverse_transform(predicted_stock_price), columns=['close'])
#     # predicted_stock_price['date'] = np.datetime64('today')
#     #
#     # for i in range(0, len(predicted_stock_price)):
#     #     predicted_stock_price['date'][i] = np.datetime64('today') + np.timedelta64(i, 'D')
#     #
#     # predicted_stock_price = predicted_stock_price.set_index(['date'], drop=True)
#     # df = df[(df.index > np.datetime64('today') - np.timedelta64(90, 'D'))]
#     # plot(df, predicted_stock_price, ticker)
#     # plt.show()
#
#
# run_analysis(ticker="AMD", epochs_num=1, forecast_days=30)
