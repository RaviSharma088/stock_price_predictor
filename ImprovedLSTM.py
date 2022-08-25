import lstm, time #helper libraries
import pandas as pd
import visualize as vs
import lstm
import stock_data as sd
#Build an improved LSTM model

# stocks = pd.read_csv('google_preprocessed.csv')
def neuralnetwork(stocks):
    stocks_data = stocks.drop(['Item'], axis =1)

    print(stocks_data.head())

    #Split train and test data sets and Unroll train and test data for improved lstm model
    X_train, X_test,y_train, y_test = sd.train_test_split_lstm(stocks_data, 5)

    unroll_length = 50
    X_train = sd.unroll(X_train, unroll_length)
    X_test = sd.unroll(X_test, unroll_length)
    y_train = y_train[-X_train.shape[0]:]
    y_test = y_test[-X_test.shape[0]:]

    print("x_train", X_train.shape)
    print("y_train", y_train.shape)
    print("x_test", X_test.shape)
    print("y_test", y_test.shape)


    # Set up hyperparameters
    batch_size = 512
    epochs = 20

    # build improved lstm model
    model = lstm.build_improved_model( X_train.shape[-1],output_dim = unroll_length, return_sequences=True)

    start = time.time()
    #final_model.compile(loss='mean_squared_error', optimizer='adam')
    model.compile(loss='mean_squared_error', optimizer='adam')
    print('compilation time : ', time.time() - start)

    #Train improved LSTM model

    model.fit(X_train, y_train, batch_size=batch_size,epochs=epochs,verbose=2,validation_split=0.05)

    #Make prediction on improved LSTM model

    predictions = model.predict(X_test, batch_size=batch_size)

    #plot the results

    vs.plot_lstm_prediction(predictions, y_test)
