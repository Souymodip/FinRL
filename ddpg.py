# from finrl import config
from finrl import config_tickers
import os
import numpy as np
from finrl.main import check_and_make_directories
from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
)
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
import pandas as pd
import itertools
from viz.viz import plot_ticker_plt as plot_ticker
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
import warnings

def set_dates():
    global TRAIN_START_DATE, TRAIN_END_DATE, TEST_START_DATE, TEST_END_DATE, TRADE_START_DATE, TRADE_END_DATE
    TRAIN_START_DATE = '2018-01-01'
    TRAIN_END_DATE = '2022-12-31'
    TEST_START_DATE = '2023-01-01'
    TEST_END_DATE = '2023-12-31'
    TRADE_START_DATE = '2024-01-01'
    TRADE_END_DATE = '2024-12-31'

def data_processing(df):
    fe = FeatureEngineer(
                    use_technical_indicator=True,
                    tech_indicator_list = INDICATORS,
                    use_vix=True,
                    use_turbulence=True,
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

def experiment_ddpg(training = False):
    check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])
    set_dates()
    print("Experiment started")
    print('Training Interval: ', TRAIN_START_DATE, ' ~ ', TRAIN_END_DATE)
    print('Testing Interval: ', TEST_START_DATE, ' ~ ', TEST_END_DATE)
    print('Trade Interval: ', TRADE_START_DATE, ' ~ ', TRADE_END_DATE)

    df = YahooDownloader(start_date = TRAIN_START_DATE,
                     end_date = TRADE_END_DATE,
                     ticker_list = config_tickers.DOW_3_TICKER).fetch_data()
    plot_ticker(df.tail(100), 'MSFT')
    import pdb; pdb.set_trace()
    processed_full = data_processing(df)
    # mvo_df = processed_full.sort_values(['date','tic'],ignore_index=True)[['date','tic','close']]
    # plot mvo_df
    print(processed_full.columns)
    train = data_split(processed_full, TRAIN_START_DATE,TRAIN_END_DATE)
    trade = data_split(processed_full, TRADE_START_DATE,TRADE_END_DATE)


    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
    print(f"----------------------------------------------\n", 
          f"Training Stock Dimension: {stock_dimension}, State Space: {state_space}\n",
          "----------------------------------------------")
    
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension # broker commission
    num_stock_shares = [0] * stock_dimension # number of shares currently held
    
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }
    e_train_gym = StockTradingEnv(df = train, **env_kwargs)
    env_train, state_init = e_train_gym.get_sb_env()

    ##### DDPG #####
    agent = DRLAgent(env = env_train)
    model_ddpg = agent.get_model("ddpg")
    folder = 'ddpg3'
    tmp_path = RESULTS_DIR + '/' + folder

    if training:
        model_ddpg.set_logger(configure(tmp_path, ["stdout", "csv", "tensorboard"]))
        trained_ddpg = agent.train_model(model=model_ddpg, tb_log_name='ddpg', total_timesteps=100000)

        # save model
        model_ddpg.save(TRAINED_MODEL_DIR)
    else:
        if not os.path.exists(f'{TRAINED_MODEL_DIR}/{folder}.zip'):
            warnings.warn(f'Path [{TRAINED_MODEL_DIR}/{folder}.zip] does not exists')
            exit()
        
        # load model
        model_ddpg.load(TRAINED_MODEL_DIR + '/' + folder)

    data_risk_indicator = processed_full[(processed_full.date<TRAIN_END_DATE) & (processed_full.date>=TRAIN_START_DATE)]
    insample_risk_indicator = data_risk_indicator.drop_duplicates(subset=['date'])

    ##### In-sample backtest #####
    e_trade_gym = StockTradingEnv(df = trade, turbulence_threshold = 70, risk_indicator_col='vix', **env_kwargs)

    df_account_value_ddpg, df_actions_ddpg = DRLAgent.DRL_prediction(model=model_ddpg, environment = e_trade_gym)

    closing_prices = np.array([
        trade[trade['tic'] == tick]['close'].to_numpy() for tick in df_actions_ddpg.columns.tolist()
    ]).transpose()

    # weights = trade.sort_values(['date','tic'],ignore_index=True)[['date','tic','open']].iloc[:3]['open'].to_numpy()
    # weights = weights / weights.sum()
    # initial_value = (1000000 * weights).astype(int)

    # closing_counts = 
    # counts = (closing_prices * initial_count[None,:])

    # last_day = closing_prices[-1]
    # last_day_valuation = (last_day * initial_count).sum()

    # import pdb; pdb.set_trace()

    naive_df = pd.DataFrame(index=df_account_value_ddpg.index)
    # naive_df['account_value'] = naive_df.index.get_indexer(df_account_value_ddpg.index)
    naive_df['account_value'] = trade['close'].iloc[0]
    # import pdb; pdb.set_trace()

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(15, 7), tight_layout=True)
    ticks = df_actions_ddpg.columns.tolist()
    for i, tick in enumerate(ticks):
        axs[0].plot(df_actions_ddpg.index, df_actions_ddpg[[tick]], label=tick)
    # import pdb; pdb.set_trace()
    axs[1].plot(df_account_value_ddpg.index, df_account_value_ddpg[['account_value']], label='account_value')
    axs[0].legend()
    axs[1].legend()
    print(f'Saving plot to {RESULTS_DIR}/{folder}.png')
    plt.savefig(f'{RESULTS_DIR}/{folder}.png', dpi=300)
    import pdb; pdb.set_trace()
    

if __name__ == "__main__":
    experiment_ddpg()
