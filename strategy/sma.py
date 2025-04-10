import pandas as pd
from scipy.stats import norm
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
def set_dates():
    global TRAIN_START_DATE, TRAIN_END_DATE, TEST_START_DATE, TEST_END_DATE, TRADE_START_DATE, TRADE_END_DATE
    TRAIN_START_DATE = '2018-01-01'
    TRAIN_END_DATE = '2023-12-31'
    # TEST_START_DATE = '2023-01-01'
    # TEST_END_DATE = '2023-12-31'
    TRADE_START_DATE = '2024-02-01'
    TRADE_END_DATE = '2025-12-31'

class SMAStrategy:
    def __init__(self, short_window=20, long_window=50):
        self.short_window = short_window
        self.long_window = long_window
        self.lag = 100

    def fit(self, series):
        # Ensure we do not modify the original dataframe directly

        sma_short = series.rolling(window=self.short_window, min_periods=1).mean()
        sma_long = series.rolling(window=self.long_window, min_periods=1).mean()
        df = pd.DataFrame({
            'short' : sma_short,
            'long' : sma_long,
            'diff' : sma_short - sma_long
        })
        return df

    def play(self, _df, start_date, val_column, tics):        
        def play_tic(tic):
            tic_df = _df[_df['tic'] == tic]
            future_df = tic_df[tic_df['date'] >= start_date]
            N = len(tic_df) - len(future_df)

            # import pdb; pdb.set_trace()
            value_df = tic_df[val_column]
            trade_signals = []
            for k in range(len(future_df)-1):
                # print(f'{N} < {N+k-1} < {len(tic_df)}')
                date = tic_df.iloc[N+k].date # next day #
                open = tic_df.iloc[N+k].open
                close = tic_df.iloc[N+k].close
                high = tic_df.iloc[N+k].high
                low = tic_df.iloc[N+k].low
                vol = tic_df.iloc[N+k].volume

                h_df = value_df[:N+k] # history till (inclusive) previous day
                h_df = self.fit(h_df)
                curr_diff = h_df.iloc[-1]['diff']
                prev_diff = h_df.iloc[-2]['diff']

                std_diff = h_df['diff'].tail(self.lag).std() 
                assert std_diff > 0, f'Standard deviation cannot be zero. {std_diff}'

                curr_z = curr_diff / std_diff
                prev_z = prev_diff / std_diff

                curr_z_p = 2 * norm.sf(abs(curr_z))
                prev_z_p = 2 * norm.sf(abs(prev_z))

                confidence = 1 - curr_z_p*prev_z_p

                # import pdb; pdb.set_trace()
                ### strategy ####
                ## Sell
                # - Previous day: SMA20 < SMA50
                # - Current day: SMA20 > SMA50
                ## Buy
                # - Previous day: SMA20 > SMA50
                # - Current day: SMA20 < SMA50
                
                data = {
                    'Date' : date, 'Open' : open, 'Close' : close, 'High':high, 'Low' : low, 'Volume' : vol,
                    'sell': prev_diff < 0 and curr_diff > 0,
                    'buy' : prev_diff > 0 and curr_diff < 0,
                    'confidence' : confidence
                }
                trade_signals.append(data)
            return pd.DataFrame(trade_signals)
        res = dict()
        for tic in tics:
            res[tic] = play_tic(tic)
        return res
    
    def transaction(self, df: pd.DataFrame, init_cash: float, brokerage: callable):
        """
        Processes transactions on a day-by-day basis for fractional stock amounts using a simplified linear brokerage fee.
        
        Brokerage fee model:
            fee = m * (trade_value) + a
        where trade_value = (number_of_shares * close_price)
        
        For a buy signal:
        - The maximum shares that can be bought with available cash 'C' at price 'p' is:
            q_max = (C - a) / (p * (1 + m))
            provided C > a.
        - Actual shares purchased = q_max * confidence.
        
        For a sell signal:
        - Sell (shares held) * confidence.
        - Revenue is computed as:
            revenue = (sell_shares * p) - [sell_shares * p * m + a]
        
        Returns a DataFrame of daily trading records containing the updated cash and share holdings.
        """
        cash = init_cash
        shares = 0.0
        trade_records = []
        
        # Determine brokerage parameters assuming linearity:
        # Let fee(x) = m*x + a. Then a = brokerage(0) and m = brokerage(1) - brokerage(0)
        a = brokerage(0)
        m = brokerage(1) - brokerage(0)
        
        # Ensure DataFrame is sorted by date (assumes Date is comparable)
        df_sorted = df.sort_values('Date')
        
        for i, row in df_sorted.iterrows():
            date = row['Date']
            close_price = row['Close']
            action = 'hold'
            
            # Buy: if buy signal is True and cash > additive fee
            if row['buy'] and cash > a:
                # Maximum shares that can be bought (fractional allowed)
                q_max = (cash - a) / (close_price * (1 + m))
                shares_to_buy = q_max * min(row['confidence'], 1.0)
                trade_value = shares_to_buy * close_price
                fee = trade_value * m + a
                total_cost = trade_value + fee
                
                # Check if sufficient cash is available
                if cash >= total_cost:
                    cash -= total_cost
                    shares += shares_to_buy
                    action = f'buy {shares_to_buy:.4f}'
                else:
                    action = 'buy signal but insufficient cash after fee'
            
            # Sell: if sell signal is True and we hold some shares
            elif row['sell'] and shares > 0:
                shares_to_sell = shares * min(row['confidence'], 1.0)
                trade_value = shares_to_sell * close_price
                fee = trade_value * m + a
                revenue = trade_value - fee
                
                # Only execute sell if revenue is positive after fees
                if revenue > 0:
                    cash += revenue
                    shares -= shares_to_sell
                    action = f'sell {shares_to_sell:.4f}'
                else:
                    action = 'sell signal but trade not profitable after fee'
            asset = close_price * shares + cash
            # Record the transaction details
            trade_records.append({
                'Date': date,
                'Open': row['Open'],
                'Close': close_price,
                'High': row['High'],
                'Low': row['Low'],
                'Volume': row['Volume'],
                'Action': action,
                'Shares_Held': shares,
                'Available_Cash': cash,
                'Confidence': row['confidence'],
                'asset' : asset
            })
        
        return pd.DataFrame(trade_records)


def plot_transactions(trans_df, tic):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    
    # Ensure the 'Date' column is in datetime format.
    if not pd.api.types.is_datetime64_any_dtype(trans_df['Date']):
        trans_df['Date'] = pd.to_datetime(trans_df['Date'])
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.grid()
    # Define the width of the candle in days (since one unit in date coordinates equals one day)
    candle_width = 0.8
    
    # Draw candlestick candles: a rectangle is plotted for each row where the bottom and height 
    # depend on the Open and Close values.
    for idx, row in trans_df.iterrows():
        date = row['Date']
        # Convert date to numeric format so that the rectangle width is interpreted in days.
        date_num = mdates.date2num(date)
        open_price = row['Open']
        close_price = row['Close']
        lower = min(open_price, close_price)
        height = abs(close_price - open_price)
        # Color: red if the day's close is lower than the open, green otherwise.
        color = 'red' if close_price < open_price else 'green'
        candle = Rectangle((date_num - candle_width/2, lower), candle_width, height,
                           color=color, alpha=0.7, zorder=2)
        ax1.add_patch(candle)
    
    # Set x-axis limits based on the date range
    ax1.set_xlim(trans_df['Date'].min(), trans_df['Date'].max())
    ax1.set_ylabel('Price', color='black')
    
    # Format the x-axis with proper date formatting.
    ax1.xaxis_date()
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    
    # Plot markers for trade actions where the Action is not 'hold'
    for idx, row in trans_df.iterrows():
        if row['Action'] != 'hold':
            date = row['Date']
            close_price = row['Close']
            asset_value = row['asset']
            
            if row['Action'].startswith('buy'):
                marker = '^'  # upward triangle for buy
                marker_color = 'orange'
                offset = (0, 8)
            elif row['Action'].startswith('sell'):
                marker = 'v'  # downward triangle for sell
                marker_color = 'blue'
                offset = (0, -12)
            else:
                continue  # skip any unknown actions
            
            # Draw the marker at the Close price
            ax1.scatter(date, close_price, color=marker_color, marker=marker, s=100, zorder=5)
            # Annotate the marker with the asset value
            ax1.annotate(f'{asset_value:.2f}', (date, close_price),
                         textcoords="offset points", xytext=offset,
                         ha='center', color=marker_color, fontsize=9)

    
    first_row = trans_df.iloc[0]
    first_date = first_row['Date']
    first_close = first_row['Close']
    first_asset = first_row['asset']
    ax1.annotate(f'{first_asset:.2f}', (first_date, first_close),
                 textcoords="offset points", xytext=(10, 0),
                 ha='left', va='center', color='purple', fontsize=10)
    
    last_row = trans_df.iloc[-1]
    last_date = last_row['Date']
    last_close = last_row['Close']
    last_asset = last_row['asset']
    ax1.annotate(f'{last_asset:.2f}', (last_date, last_close),
                 textcoords="offset points", xytext=(-20, 0),
                 ha='left', va='center', color='purple', fontsize=10)
    
    # Add a legend combining the line plots
    # ax1.legend(loc='upper left')
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='orange',
               markersize=10, label='Buy Signal'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='blue',
               markersize=10, label='Sell Signal')
    ]
    ax1.legend(handles=legend_elements, loc='upper left')

    plt.title(f"[{tic}] (Buy/Sell) and Asset: {last_asset:.2f}")
    plt.show()


def test():
    set_dates()
    df = download_data(TRAIN_START_DATE, TRADE_END_DATE, config_tickers.DOW_3_TICKER)

    # Create an instance of SMAStrategy and generate SMAs
    sma_strategy = SMAStrategy()
    val_column='close'
    tics=config_tickers.DOW_3_TICKER
    # df_with_sma = sma_strategy.fit(df, val_column)
    res = sma_strategy.play(df, TRADE_START_DATE, val_column, tics=tics)
    transactions = {}

    for tic in tics:
        m, a = 0.01, 1
        trans = sma_strategy.transaction(res[tic], init_cash=10000, brokerage=lambda x: m*x + a)
        transactions[tic] = trans
        plot_transactions(trans, tic)




    # print(df_with_sma.head(25))


# Example usage:
if __name__ == "__main__":
    pass
