from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, GRU, Input, Dense, Concatenate, Conv1D, Attention
from tensorflow.keras.models import Model

# 读取数据
df = pd.read_csv('data/binance/BTCUSDT_5m.csv')

df["date"] = df["date"].apply(lambda x: int(datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timestamp()))

# 计算价格差
df["price_diff"] = df["close"] - df["open"]

# 计算对数收益率
df["log_return"] = np.log(1 + df["price_diff"] / df["open"])

# 选择需要的列并将数据转换为numpy array
data = df[["date", "price_diff", "log_return"]].to_numpy()

X = data[:, :-1]  # 除去最后一列的所有列
y = data[:, -1]  # 最后一列（log return）

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def create_compound_model(input_shape):
    input_layer = Input(shape=input_shape)

    # LSTM
    lstm_layer = LSTM(units=64)(input_layer)

    # GRU
    gru_layer = GRU(units=64)(input_layer)

    # Self Attention
    attention_layer = Attention()([input_layer, input_layer])
    attention_dense = Dense(units=64, activation='relu')(attention_layer)

    # 1D CNN
    conv1d_layer = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)

    # Concatenate layers
    concat_layer = Concatenate()([lstm_layer, gru_layer, attention_dense, conv1d_layer])

    # Output layer
    output_layer = Dense(1, activation='linear')(concat_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model


input_shape = (X_train.shape[1], 1)
model = create_compound_model(input_shape)

# Reshape data for LSTM, GRU and 1D CNN input
X_train_reshaped = X_train.reshape(-1, X_train.shape[1], 1)
X_test_reshaped = X_test.reshape(-1, X_test.shape[1], 1)

model.fit(X_train_reshaped, y_train, epochs=100, batch_size=32, validation_data=(X_test_reshaped, y_test))

loss, mae = model.evaluate(X_test_reshaped, y_test)
print("Mean Absolute Error on Test Set:", mae)

predictions = model.predict(X_test_reshaped)
