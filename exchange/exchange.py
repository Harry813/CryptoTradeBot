import os
import logging
import time
from typing import Any, Dict, List
from datetime import datetime, timezone, timedelta

import ccxt
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from config import PAIRLIST
from exchange.common import SUPPORTED_EXCHANGES
from settings import DOWNLOAD_PATH
from utils.common import parse_timeframe, get_path

logger = logging.getLogger(__name__)


class Exchange:
    def __init__ (self, name, api_key, secret_key, config: Dict[str, Any] = None):
        self._name = name
        self._api_key = api_key
        self._secret_key = secret_key

        self.validate()

        self.exchange = getattr(ccxt, self._name.lower())({
            'apiKey': self._api_key,
            'secret': self._secret_key,
            **(config or {})
        })

    def validate (self):
        if self._name not in SUPPORTED_EXCHANGES:
            raise ValueError(f'Unsupported exchange: {self._name}')

        if not self._api_key or not self._secret_key:
            raise ValueError('API key or secret key is missing')

    def fetch_ticker (self, symbol):
        return self.exchange.fetch_ticker(symbol)

    def fetch_ohlcv (self, symbol, timeframe, limit=1000, since: int = None, params=None):
        if params is None:
            params = {}

        try:
            df = pd.DataFrame(self.exchange.fetch_ohlcv(symbol, timeframe, limit, since, params),
                              columns=["date", "open", "high", "low", "close", "volume"])
            df["date"] = pd.to_datetime(df["date"], unit="ms")
            df.set_index('date', inplace=True)
            df = df.sort_index(ascending=True)
            return df
        except ccxt.ExchangeError as e:
            raise e

    def _download (self, start: datetime, end: datetime, timeframe: str,
                   interval: int, time_scale: int, pair: str) -> DataFrame:

        timestamps = []

        div, mod = divmod(end - start, timedelta(seconds=1000 * parse_timeframe(timeframe)))

        if mod != 0:
            div += 1

        since = start
        for i in range(div):
            timestamps.append({
                "since": self.exchange.parse8601(since.isoformat()),
                "limit": interval if i != div - 1 else int(mod / timedelta(seconds=time_scale))
            })

        data = []
        for timestamp in tqdm(timestamps,
                              desc=f'\033[32mDownloading {pair} {timeframe:>3} data from {self._name}\033[0m'):
            data.append(self.fetch_ohlcv(pair, timeframe, timestamp["since"], timestamp["limit"]))
            time.sleep(self.exchange.rateLimit / 1000)
        return pd.concat(data)

    def download_data (self, timeframes: List[str], start: datetime,
                       end: datetime = datetime.now(timezone.utc),
                       data_path: str = DOWNLOAD_PATH, pairlist: List[str] = PAIRLIST,
                       interval: int = 1000):
        if end < start:
            raise ValueError('End time must be greater than start time')
        if end > datetime.now(timezone.utc):
            raise ValueError('End time must be less than now')

        download_path = get_path(data_path)
        if not os.path.exists(download_path):
            os.makedirs(download_path)
        if not os.path.exists(f'{download_path}/{self._name}'):
            os.makedirs(f'{download_path}/{self._name}')

        for pair in pairlist:
            pair_path = pair.replace('/', '')
            for timeframe in timeframes:
                try:
                    time_scale = parse_timeframe(timeframe)
                except ValueError:
                    raise ValueError(f'Unsupported timeframe: {timeframe}')

                file_name = f'{download_path}/{self._name}/{pair_path}_{timeframe}.csv'

                self._download(
                    start=start,
                    end=end,
                    timeframe=timeframe,
                    interval=interval,
                    time_scale=time_scale,
                    pair=pair
                ).to_csv(file_name)

    def __str__ (self):
        return f'Exchange({self._name})'

    @property
    def name (self):
        return self._name
