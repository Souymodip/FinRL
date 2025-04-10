import pandas as pd
from backtesting import Backtest, Strategy
from load_data import download_data

from finrl import config_tickers
from finrl.config import (
    INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
)

class MyStrategy(Strategy):
    def init(self):
        # Example: 20-day simple moving average of the close price
        self.sma = self.I(lambda: df['Close'].rolling(20).mean())

    def next(self):
        # Simple logic: buy when price crosses above the SMA, sell otherwise
        if self.data.Close[-1] < self.sma[-1] and self.data.Close[0] > self.sma[-1]:
            self.buy()
        elif self.data.Close[-1] > self.sma[-1] and self.data.Close[0] < self.sma[-1]:
            self.sell()


def set_dates():
    global TRAIN_START_DATE, TRAIN_END_DATE, TEST_START_DATE, TEST_END_DATE, TRADE_START_DATE, TRADE_END_DATE
    TRAIN_START_DATE = '2018-01-01'
    TRAIN_END_DATE = '2024-12-31'
    # TEST_START_DATE = '2023-01-01'
    # TEST_END_DATE = '2023-12-31'
    TRADE_START_DATE = '2025-02-01'
    TRADE_END_DATE = '2025-12-31'

if __name__ == '__main__':
    set_dates()
    df = download_data(TRAIN_START_DATE, TRADE_END_DATE, config_tickers.DOW_3_TICKER)
    # Convert 'date' column to datetime objects first
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['tic'] == 'AAPL']
    df = df.rename(columns={
        'date': 'Date',
        'open': 'Open',
        'high': 'High',
        'low':  'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    # Set the 'Date' column as the index
    df.set_index('Date', inplace=True)
    # import pdb; pdb.set_trace() # Removed the debugger line
    # df.index = pd.DatetimeIndex(df.Date) # This is now handled by set_index


    bt = Backtest(df, MyStrategy, cash=10000, commission=0.002)
    import pdb; pdb.set_trace()
    result = bt.run()
    print(result)
    bt.plot()
