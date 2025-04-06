import pandas as pd
import numpy as np
import itertools
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
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
    TRAIN_END_DATE = '2022-12-31'
    TEST_START_DATE = '2023-01-01'
    TEST_END_DATE = '2023-12-31'
    TRADE_START_DATE = '2024-01-01'
    TRADE_END_DATE = '2024-12-31'


def data_processing(df, indicators=INDICATORS, vix=True, turbulence=True):
    fe = FeatureEngineer(
                    use_technical_indicator=True,
                    tech_indicator_list = indicators,
                    use_vix=vix,
                    use_turbulence=turbulence,
                    user_defined_feature = False)
    processed = fe.preprocess_data(df)
    list_ticker = processed["tic"].unique().tolist()
    list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
    combination = list(itertools.product(list_date,list_ticker))
    processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
    processed_full = processed_full[processed_full['date'].isin(processed['date'])]
    processed_full = processed_full.sort_values(['date','tic'])
    processed_full = processed_full.fillna(0)
    return processed_full


def download_data(start_date, end_date, ticker_list):
    df = YahooDownloader(start_date = start_date,
                    end_date = end_date,
                    ticker_list = ticker_list).fetch_data()
    
    return df


