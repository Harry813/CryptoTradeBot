import os
from typing import Any, Dict, List
from datetime import datetime, timezone, timedelta

from config import *
from settings import *

from exchange.exchange import Exchange


# 2. 数据预处理
def preprocess_data ():
    # 读取数据
    # 数据预处理
    # 保存到本地
    pass


# 3. 特征工程
def feature_engineering ():
    # 读取数据
    # 特征工程
    # 保存到本地
    pass


if __name__ == '__main__':
    exc = Exchange(
        name="binance",
        api_key=BINANCE_API_KEY,
        secret_key=BINANCE_SECRET_KEY
    )
    exc.download_data(["1m", "5m", "15m", "1h"],
                      datetime(2021, 1, 1, tzinfo=timezone.utc),
                      datetime(2021, 12, 31, tzinfo=timezone.utc))
