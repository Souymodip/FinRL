from playground.backtesting import test_per_ticker as test
from strategy.arima import test as test_arima
from strategy.sma import test as test_sma
from news import gather
import pandas as pd

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
    TRAIN_START_DATE = '2015-01-01'
    TRAIN_END_DATE = '2019-12-31'
    # TEST_START_DATE = '2023-01-01'
    # TEST_END_DATE = '2023-12-31'
    TRADE_START_DATE = '2020-02-01'
    TRADE_END_DATE = '2020-12-31'


import matplotlib.pyplot as plt

def plot_ticker_data(date_value, date_sentiment, ticker):
    # Filter the DataFrame by the given ticker value
    # df_ticker = merged_df[merged_df['tic'] == ticker].copy()
    
    # Create a figure and a primary axis (for the close price)
    fig, ax1 = plt.subplots(1, 1, figsize=(13, 8), tight_layout=True)
    
    # Plot the close value on the primary y-axis (left side)
    ax1.plot(date_value['date'], date_value['value'], color='blue', label='Close')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Close', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Create a secondary axis sharing the same x-axis for the sentiment scores
    ax2 = ax1.twinx()

    # Plot the sentiment scores on the secondary y-axis (right side)
    ax2.plot(date_sentiment['date'], date_sentiment['sentiment'], '--', color='orange', label="sentiment", alpha=0.5)
    ax2.plot(date_sentiment['date'], date_sentiment['sma'], 'r-', label="sma 15")
    # ax2.plot(df_ticker['date'], df_ticker['neutral_score'], 'o', color='gray', label='Neutral Score')
    # ax2.plot(df_ticker['date'], df_ticker['positive_score'], 'o', color='green', label='Positive Score')
    # ax2.plot(df_ticker['date'], df_ticker['negative_score'], 'o', color='red', label='Negative Score')
    ax2.set_ylabel('Sentiment Scores')
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    # ax2.legend()
    
    # Add a title and format the date labels for better readability
    plt.title(f"{ticker} - Close Price and Sentiment Scores Over Time")
    fig.autofmt_xdate()
    
    plt.show()


def gather_test():
    sect_dict = gather.get_sectors()
    print(f'------------------- SECTORES -----------------')
    for sect in sect_dict.keys():
        print(sect)
    print(f'----------------------------------------------')
    sector = 'Health Care' # 'Information Technology'
    sect_tics = sect_dict[sector]
    print(f'Sector Stock: {sect_tics}')
    df_sentiment = gather.organize_sentiment()
    df_history = download_data(TRAIN_START_DATE, TRADE_END_DATE, sect_tics)
    df_merged = gather.merge_sentiment(df_history=df_history, df_sentiment=df_sentiment)
    print(df_merged.columns.to_list())

    sentiment = df_merged[['date', 'tic', 'neutral_score', 'positive_score', 'negative_score']]
    sentiment_grouped = sentiment.groupby(['date'])[['neutral_score', 'positive_score', 'negative_score']].sum().reset_index()
    sentiment = sentiment_grouped['positive_score'] -  sentiment_grouped['negative_score'] + 0.5 * sentiment_grouped['neutral_score']
    
    ticker='CNC'
    assert ticker in set(sect_tics), f'{ticker} not in {sect_tics}'

    value = df_merged[df_merged['tic'] == ticker]
    date_value = pd.DataFrame({
        'date' : value['date'],
        'value' : value['close']
    })
    date_sentiment = pd.DataFrame({
        'date' : sentiment_grouped['date'],
        'sentiment' : sentiment
    })
    date_sentiment['date'] = pd.to_datetime(date_sentiment['date'])
    date_sentiment = date_sentiment.sort_values('date')
    date_sentiment.set_index('date', inplace=True)
    date_sentiment['sma'] = date_sentiment['sentiment'].rolling(window=15).mean()
    date_sentiment = date_sentiment.reset_index()

    plot_ticker_data(date_value, date_sentiment,  ticker=ticker)


if __name__ == "__main__":
    # test_arima()
    # test_sma()
    set_dates()
    gather_test()
    
    