#!/usr/bin/env python
# coding: utf-8

import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

THRESHOLD = 0.7
PREDICTION_PERIOD = 60


def load_data(file_name='./test6610.xlsx',
              data_tab='EndOfDay',
              names_tab='Names'):
    xls = pd.ExcelFile(file_name)
    data_df = pd.read_excel(xls, data_tab)
    names_df = pd.read_excel(xls, names_tab)
    return (data_df, names_df)


def extract_target_data(data_df, names_df, target_name='ת"א-125'):
    # Finding sec_num of 'ת"א-125'
    target_sec_num = names_df.sec_num[names_df.sec_name ==
                                      target_name].values[0]
    target = data_df[data_df.sec_num == target_sec_num]
    target.set_index('date', inplace=True)
    return target


def choose_stocks(df, df_names, y, choose_amount=10):
    # filter all stocks exclude indices
    stocks = df_names.sec_num[df_names.sec_num >= 1000].tolist()
    basis_correlations = {}
    closing_correlations = {}
    # extracting data for each stock
    for stock in stocks:
        x = df[df.sec_num == stock]
        x.set_index('date', inplace=True)
        # filter out stocks with less than THRESHOLD% from target observations
        if len(df[df.sec_num == stock]) > (THRESHOLD * len(y)):
            # calculating correlations with target variable
            basis_corr = x.basis_price.corr(y.closing_price)
            closing_corr = x.closing_price.corr(y.closing_price)
            # saving correlations' absolut value to dictionaries
            basis_correlations[stock] = abs(basis_corr)
            closing_correlations[stock] = abs(closing_corr)
    # extracting top 10 stocks with absolut value correlations to lists
    basis_most_correlated_with_target = sorted(basis_correlations,
                                               key=basis_correlations.get,
                                               reverse=True)[:choose_amount]
    closing_most_correlated_with_target = sorted(closing_correlations,
                                                 key=closing_correlations.get,
                                                 reverse=True)[:choose_amount]
    most_correlated_with_target = [
        X for X in basis_most_correlated_with_target
        if X in closing_most_correlated_with_target
    ]
    return most_correlated_with_target


def plot_chosen_stocks_basis_price(y, df, df_names, target_label='TA-125'):
    # plot 10 chosen stocks
    plt.figure(figsize=(15, 9))
    y.basis_price.plot(label=target_label, linewidth=4, color='black')
    for correlated in most_correlated_with_target:
        x = df[df.sec_num == correlated]
        x.set_index('date', inplace=True)
        name = df_names.sec_name[df_names['sec_num'] ==
                                 correlated].values[0][::-1]
        x.basis_price.plot(label=name)
    plt.title('Chosen stocks basis price', fontsize=22)
    plt.xlabel('DATE', fontsize=14)
    plt.ylabel('PRICE', fontsize=14)
    plt.legend(title='Basis price')
    plt.show()

    
def preprocess_data_for_NN(df, most_correlated_with_target, y):
    X = pd.DataFrame(
        index=y.index
    )  # create the X dataframe with columns for each stock price
    for correlated in most_correlated_with_target:
        x = df[df.sec_num == correlated].copy()
        x.set_index('date', inplace=True)
        x.rename(columns={
            "basis_price": f"{correlated} basis_price",
            "closing_price": f"{correlated} closing_price"
        },
                 inplace=True)
        tmp_basis = x[f"{correlated} basis_price"]
        tmp_closing = x[f"{correlated} closing_price"]
        X = X.join(tmp_basis)
        X = X.join(tmp_closing)
    dataset = X.join(y.closing_price)
    dataset.rename(columns={
        "closing_price": "TA-125 closing_price",
    },
                   inplace=True)
    # Scale the all of the data to be values between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    stacking_range = dataset.shape[1] - 1
    # Compute the number of rows to train the model on
    training_data_len = math.ceil(len(scaled_data) * .8)
    # Create the scaled training data set
    train_data = scaled_data[0:training_data_len, :]
    x_train = []
    # Split the data into x_train and y_train data sets
    for i in range(PREDICTION_PERIOD, len(train_data)):
        x_train.append(train_data[i - PREDICTION_PERIOD:i, :-1])
    x_train = np.array(x_train)
    y_train = train_data[PREDICTION_PERIOD:, -1]
    test_data = scaled_data[training_data_len:, :]
    # Create the x_test and y_test data sets
    y_test = dataset.iloc[(training_data_len + PREDICTION_PERIOD):, -1]
    x_test = []
    for i in range(PREDICTION_PERIOD, len(test_data)):
        x_test.append(test_data[i - PREDICTION_PERIOD:i, :-1])
    x_test = np.array(x_test)
    return (x_train, y_train, x_test, y_test, scaler, stacking_range)


def build_lstm(x_train):
    model = Sequential()
    model.add(
        LSTM(units=128,
             return_sequences=True,
             input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dense(units=18, activation='relu'))
    model.add(Dense(units=9))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_model(model, x_train, y_train):
    early_stop = EarlyStopping(monitor="loss", patience=10, mode='min')
    model_save = ModelCheckpoint('.mdl_stocks_pred.hdf5',
                                 save_best_only=True,
                                 monitor='loss',
                                 mode='min')
    model.fit(x_train,
              y_train,
              batch_size=1,
              epochs=100,
              callbacks=[early_stop, model_save],
              verbose=False)
    return model


def inverse_scaler(predictions, scaler):
    # preparing shape for scaler
    stacked = predictions
    for i in range(stacking_range):
        stacked = np.hstack((stacked, predictions))
    # Undo scaling
    y_pred = scaler.inverse_transform(stacked)
    # extracting prediction in original scale
    y_pred = y_pred[:, -1]
    return y_pred


def calculate_rmse(y_pred, y_test):
    rmse = np.sqrt(np.mean(((y_pred - y_test)**2)))
    return rmse


if __name__ == "__main__":
    # Loading the tables from Excel file
    df, df_names = load_data()
    # Extracting target variable data frame
    y = extract_target_data(df, df_names)
    # Finding 10 variables with highest absolute correlation to target variable
    most_correlated_with_target = choose_stocks(df, df_names, y)
    plot_chosen_stocks_basis_price(y, df, df_names)
    # preparing the data for the model
    x_train, y_train, x_test, y_test, scaler, stacking_range = preprocess_data_for_NN(
        df, most_correlated_with_target, y)
    # Building the model and printing model summary
    model = build_lstm(x_train)
    print(model.summary())
    # Training the model on the train set
    model = train_model(model, x_train, y_train)
    # Getting the models predicted values
    predictions = model.predict(x_test)
     # scaling prediction to actual values
    y_pred = inverse_scaler(predictions, scaler)
    # Calculate the loss function, Root Mean Squre Error and printing it
    rmse = calculate_rmse(y_pred, y_test)
    print('\nRoot Mean Square Error - ', round(rmse, 2))

