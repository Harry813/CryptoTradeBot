import os
import time
import numpy as np
import pandas as pd
from binance import Client
from ta import add_all_ta_features
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, GRU, Conv1D, Dense, Dropout, Flatten, MaxPooling1D
from keras.callbacks import ModelCheckpoint
from config import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import ccxt
import pandas as pd
import numpy as np
import time
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

API_KEY = BINANCE_API_KEY
API_SECRET = BINANCE_SECRET_KEY
SYMBOLS = ['BTCUSDT', 'ETHUSDT']  # 用户决定的虚拟货币
TIMEFRAME = '1m'  # K线时间周期
TRADE_INTERVAL = 5 * 60  # 交易间隔（秒）
UPDATE_INTERVAL = 10 * 60  # 更新模型间隔（秒）
MODEL_PATH = 'models'
HIST_DATA_LENGTH = 1000
SEQUENCE_LENGTH = 60
BUY_THRESHOLD = 65
SELL_THRESHOLD = -35

client = Client(API_KEY, API_SECRET)


def get_historical_data (symbol, interval):
    # 获取历史数据并计算技术指标
    klines = client.get_historical_klines(symbol, interval, f"{HIST_DATA_LENGTH} day ago UTC")
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                       'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                       'taker_buy_quote_asset_volume', 'ignore'])
    df = df.astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    return add_all_ta_features(df, open='open', high='high', low='low', close='close', volume='volume')


def create_model ():
    # 创建混合模型
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(SEQUENCE_LENGTH, len(INPUT_COLUMNS))))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(GRU(50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(50))
    model.add(Dropout(0.2))

    model.add(Dense(2))

    model.compile(optimizer='adam', loss='mse')
    return model


def update_model (symbol):
    # 更新并保存模型
    data = get_historical_data(symbol, TIMEFRAME)
    X, y = preprocess_data(data)
    model = create_model()
    checkpoint = ModelCheckpoint(f'{MODEL_PATH}/{symbol}_best_model.h5', monitor='val_loss', save_best_only=True)
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, callbacks=[checkpoint])


def load_saved_model (symbol):
    # 加载已保存的模型
    return load_model(f'{MODEL_PATH}/{symbol}_best_model.h5')


def preprocess_data (data):
    # 数据预处理
    input_data = data[INPUT_COLUMNS].values
    scaler = MinMaxScaler()
    input_data = scaler.fit_transform(input_data)

    X = []
    y = []

    for i in range(SEQUENCE_LENGTH, len(input_data)):
        X.append(input_data[i - SEQUENCE_LENGTH:i])
        y.append([data.iloc[i]['short_score'], data.iloc[i]['long_score']])

    return np.array(X), np.array(y)


def calculate_short_long_scores (data):
    # 根据均线策略，动量策略，均值回归策略，返回短线和长线评分
    short_score = (data['rsi_14'] - 50) * 2
    long_score = (data['macd'] - data['macd_signal']) * 10
    return short_score, long_score


def calculate_scores_with_model (symbol, model, data):
    # 使用模型计算评分
    X, _ = preprocess_data(data)
    recent_data = X[-1].reshape(1, SEQUENCE_LENGTH, len(INPUT_COLUMNS))
    short_score, long_score = model.predict(recent_data)[0]
    return short_score * 100, long_score * 100


def execute_trade (symbol, score, buy_threshold, sell_threshold):
    # 根据评分执行交易
    trade_amount = abs(score) / 100
    if score > 0:
        # 买入
        pass
    elif score < 0:
        # 卖出
        pass


INPUT_COLUMNS = ['rsi_14', 'macd', 'macd_signal']


def main ():
    # 初始化 Binance API
    client = Client(API_KEY, API_SECRET)

    # 训练或加载模型
    models = {}
    for symbol in SYMBOLS:
        model_path = f'{MODEL_PATH}/{symbol}_best_model.h5'
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        if not os.path.exists(model_path):
            update_model(symbol)
        models[symbol] = load_saved_model(symbol)

    # 进行交易和更新模型
    while True:
        for symbol in SYMBOLS:
            data = get_historical_data(symbol, TIMEFRAME)
            data['short_score'], data['long_score'] = zip(*data.apply(calculate_short_long_scores, axis=1))
            short_score, long_score = calculate_scores_with_model(symbol, models[symbol], data)
            final_score = calculate_final_score(short_score, long_score)
            execute_trade(symbol, final_score, BUY_THRESHOLD, SELL_THRESHOLD)

        time.sleep(TRADE_INTERVAL)

        # 更新模型
        for symbol in SYMBOLS:
            update_model(symbol)
            models[symbol] = load_saved_model(symbol)


if __name__ == '__main__':
    main()
