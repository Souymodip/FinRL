import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

def plot_candlestick(stock_df, trade_df, tic):
    # Filter stock data for the given ticker and ensure the date is datetime
    df = stock_df[stock_df['tic'] == tic].copy()
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6), tight_layout=True)
    ax.xaxis_date()  # Tell matplotlib that the x-axis contains dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()  # Auto-format date labels
    ax.grid()
    # Set the width of the candlestick (in days)
    width = 0.6

    # Plot each row as a candlestick
    for _, row in df.iterrows():
        date_num = mdates.date2num(row['date'])
        o, c, h, l = row['open'], row['close'], row['high'], row['low']
        color = 'green' if c >= o else 'red'
        
        # Plot high-low line
        ax.plot([date_num, date_num], [l, h], color='black', alpha=0.4)
        
        # Plot candle body as a rectangle
        lower = min(o, c)
        height = abs(c - o)
        rect = Rectangle((date_num - width/2, lower), width, height, color=color)
        ax.add_patch(rect)
    
    # Overlay trade text (assuming dates match exactly between DataFrames)
    if trade_df is not None and not trade_df.empty:
        init_cap = trade_df['capital'].iloc[0]
        trade_df['date'] = pd.to_datetime(trade_df['date'])
        for _, row in trade_df.iterrows():
            e_date = row['date']
            capital = (row['capital'] - init_cap)
            text_color = 'green' if capital >= 0 else 'red'
            match = df[df['date'] == e_date]
            if not match.empty:
                # Set y position to the high value of that day
                y_pos = match.iloc[0]['high']
                e_date_num = mdates.date2num(e_date)
                ax.text(e_date_num, y_pos, f'{capital:.2f}', color=text_color,
                        ha='center', va='bottom', fontsize=10)
    
    # Set titles and labels
    ax.set_title(f'{tic}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    plt.show()


def test():
    # ----- Test Example -----

    # Sample stock trading data
    stock_data = {
        'date': ['2024-07-03', '2024-07-04', '2024-07-05', '2024-07-08', '2024-07-09'],
        'tic': ['AAPL', 'AAPL', 'AAPL', 'AAPL', 'AAPL'],
        'open': [150, 152, 151, 153, 155],
        'close': [152, 151, 153, 154, 156],
        'high': [153, 153, 154, 155, 157],
        'low': [149, 150, 150, 152, 154],
        'volume': [100000, 110000, 105000, 115000, 120000]
    }
    df_stock = pd.DataFrame(stock_data)

    # Sample portfolio data with portfolio on specific dates
    portfolio_data = {
        'date': ['2024-07-04', '2024-07-09'],
        'portfolio': [2.5, -1.2]
    }
    df_portfolio = pd.DataFrame(portfolio_data)

    # Plot the candlestick chart for the ticker 'AAPL'
    plot_candlestick(df_stock, df_portfolio, 'AAPL')

if __name__ == "__main__":
    test()
