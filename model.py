import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

file_path = 'KAG_energydata_complete.csv'
df = pd.read_csv(file_path)

features = ['T_out', 'RH_out', 'T1', 'RH_1', 'lights']
target = 'Appliances'

X = df[features].values
y = df[target].values.reshape(-1, 1)

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_scaled = x_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)

def create_sequences(X_data, y_data, seq_length):
    X_seq = []
    y_seq = []

    for i in range(len(X_data) - seq_length):
        X_seq.append(X_data[i:i + seq_length])
        y_seq.append(y_data[i + seq_length])

    return np.array(X_seq), np.array(y_seq)

seq_length = 10

X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)

split_idx = int(len(X_seq) * 0.8)

X_train = X_seq[:split_idx]
X_test = X_seq[split_idx:]
y_train = y_seq[:split_idx]
y_test = y_seq[split_idx:]
 
model = Sequential()
model.add(Input(shape=(seq_length, X_train.shape[2])))
model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(
    optimizer='adam',
    loss='mse'
)

model.summary()

history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=2, shuffle=False)