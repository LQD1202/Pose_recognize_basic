import numpy as np
import pandas as pd

from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split

bodyswing_df = pd.read_csv("SWING.txt")
handswing_df = pd.read_csv("HANDSWING.txt")

x = []
y = []
no_of_timsteps = 10

dataset = bodyswing_df.iloc[:, 1:].values
n_sample = len(dataset)
for i in range(no_of_timsteps, n_sample):
    x.append(dataset[i-no_of_timsteps:i, :])
    y.append(1)

datasets = handswing_df.iloc[:, 1:].values
n_samples = len(datasets)
for i in range(no_of_timsteps, n_samples):
    x.append(dataset[i-no_of_timsteps:i, :])
    y.append(0)

x, y = np.array(x), np.array(y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(no_of_timsteps, x.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True, input_shape=(no_of_timsteps, x.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True, input_shape=(no_of_timsteps, x.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True, input_shape=(no_of_timsteps, x.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer="adam", metrics=['accuracy'], loss = "binary_crossentropy")
model.fit(X_train, y_train, epochs=16, batch_size=32, validation_data=(X_test, y_test))
model.save("model.h5")


