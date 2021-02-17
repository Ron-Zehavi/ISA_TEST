# ISA_TEST

Test requirements - 
given - test6610.xlsx file containing open/close prices of securities and measures trading in Tel Aviv stock exchange market
mission - choose 10 stocks and predict next day price
limits - 200 lines of clean code

Solution - 

I chose to use Long short-term memory (LSTM) which is an artificial recurrent neural network (RNN) architecture used in the field of deep learning.
Unlike standard feedforward neural networks, LSTM has feedback connections. It can not only process single data points,
but also entire sequences of data like stock prices from 30 or 60 days before the day we want to predict.

LSTM models are able to store information over a period of time. 
This characteristic is extremely useful when we deal with Time-Series.
When using an LSTM, the model decides what information will be stored and what discarded.
Or in our case to determine the optimal weight of each observation in the sequence before the target day.

weaknesses -
1) Data set is a bit small to provide good results, usually, to train the deep neural network effectively we need to have at least hundreds of thousands of observations.
2) reality is too chaotic to predict with sufficient accuracy to support trading decisions with the current knowledge (otherwise one could become the richest man alive in a very short time). 
3) Because of the mission limits I aimed just to focus on accuracy, but in order to support real trade, it is also important to decide the trade direction (long/short) and taking it into consideration in the loss function.
