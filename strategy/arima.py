import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning, ConvergenceWarning
import matplotlib.pyplot as plt
from load_data import download_data, data_processing, set_dates
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from finrl import config_tickers
# import math
from statsmodels.stats.diagnostic import acorr_ljungbox
from tqdm import tqdm

import matplotlib
from strategy.viz import test as viz_test, plot_candlestick

matplotlib.use('TkAgg')

warnings.simplefilter('ignore', InterpolationWarning)
warnings.simplefilter('ignore', FutureWarning) # Suppress Prophet and Pandas future warnings
warnings.simplefilter('ignore', ConvergenceWarning)
# warnings.simplefilter('ignore', UserWarning)

# Use relative import assuming arima.py and super.py are in the same directory 'strategy'
from .super import Strategy

class Arima(Strategy):
    """
    Trading strategy based on ARIMA forecasting after time series stationarization.

    Inherits from the Strategy base class.
    """
    def __init__(self, p: int = 14, d: int = 1, q: int = 14, 
                 lowess_frac: float = 0.1, diff_order: int = 1, tics: list[str] = []):
        """
        Initializes the ARIMA strategy.

        Parameters:
        -----------
        detrending_method : str, optional
            Method for making the series stationary ('diff_stationary', 'lowess_stationary', 'prophet_stationary').
            Defaults to 'diff_stationary'.
        p : int, optional
            Order of the AR part of ARIMA. Defaults to 5.
        q : int, optional
            Order of the MA part of ARIMA. Defaults to 5.
        forecast_horizon : int, optional
            Number of steps ahead to forecast. Defaults to 1.
        lowess_frac : float, optional
            Fraction of data used when smoothing with LOWESS. Defaults to 0.1. Only used if detrending_method='lowess_stationary'.
        diff_order : int, optional
            Order of differencing. Defaults to 1. Only used if detrending_method='diff_stationary'.
        """
        self.p = p
        self.d = d 
        self.q = q

        self.lowess_frac = lowess_frac
        self.diff_order = diff_order

        assert tics is not None and len(tics) > 0, "tics must be provided"
        self.tics = tics

        print('\n\n----------------------------------------------------------------')
        print(f"\t ARIMA parameters: p={self.p}, d={self.d}, q={self.q}")
        print('----------------------------------------------------------------\n\n')

    # def _check_stationarity(self, series: pd.Series):
    #     """Performs ADF and KPSS tests for stationarity."""
    #     try:
    #         # Ensure sufficient data points after dropping NaNs
    #         series_clean = series.dropna()
    #         if len(series_clean) < 10: # Heuristic threshold
    #              print(f"Warning: Insufficient data ({len(series_clean)} points) for stationarity check.")
    #              return 1.0, 0.0 # Assume non-stationary

    #         adf_result = adfuller(series_clean, regression='c', autolag='AIC')
    #         # Use 'auto' lags for KPSS, handle potential warnings
    #         with warnings.catch_warnings():
    #              warnings.simplefilter("ignore") # Ignore warnings from kpss test e.g. about short series
    #              kpss_result = kpss(series_clean, regression='c', nlags='auto')
    #         return adf_result[1], kpss_result[1] # p-value for ADF, p-value for KPSS
    #     except Exception as e:
    #         print(f"Warning: Stationarity check failed: {e}")
    #         return 1.0, 0.0 # Assume non-stationary if check fails

    # def _prophet_stationalization(self, series: pd.Series):
    #     """Detrends a time series using Prophet."""
    #     model = Prophet()
    #     prophet_df = pd.DataFrame({
    #         'ds': series.index,
    #         'y': series.values
    #     }).dropna() # Prophet requires non-NaN 'y'

    #     if prophet_df.empty:
    #          raise ValueError("Series is empty after preparing for Prophet.")

    #     model.fit(prophet_df)
    #     forecast = model.predict(prophet_df)
    #     # Align forecast index with the original series index used in fitting
    #     forecast.index = prophet_df.index

    #     yhat = forecast['yhat']
    #     residuals = prophet_df['y'].values - yhat.values
    #     residuals_series = pd.Series(residuals, index=prophet_df.index)

    #     # Reindex residuals and yhat to match original series index, filling gaps if necessary
    #     residuals_series = residuals_series.reindex(series.index)
    #     yhat = yhat.reindex(series.index)

    #     def prediction_lambda(forcast_df):
    #         n_future = len(forcast_df)
    #         prophet_forcast_df = model.make_future_dataframe(periods=n_future, freq=series.index.freq)
    #         # assuming additive model
    #         y = forcast_df.values + prophet_forcast_df['trend'].values + prophet_forcast_df['seasonality'].values
    #         df = pd.DataFrame(y, index=forcast_df.index)
    #         return df

    #     return residuals_series, yhat, prediction_lambda

    # def _de_trending_by_differencing(self, series: pd.Series):
    #     """Detrends a time series by differencing."""
    #     diff_series = series.copy()
    #     for _ in range(self.diff_order):
    #         diff_series = diff_series.diff()

    #     def prediction_lambda(forcast_df):
    #         return forcast_df.cumsum() + series.iloc[-1]
    #     return diff_series.dropna(), prediction_lambda

    # def _de_trending_by_lowess(self, series: pd.Series):
    #     """Detrends a time series using LOWESS."""
    #     series_clean = series.dropna()
    #     if series_clean.empty:
    #         raise ValueError("Series is empty after dropping NaNs for LOWESS.")

    #     # Convert DatetimeIndex to numeric representation for LOWESS
    #     numeric_index = series_clean.index.to_julian_date()
    #     lowess_result = lowess(series_clean.values, numeric_index, frac=self.lowess_frac)
    #     trend_series = pd.Series(lowess_result[:, 1], index=series_clean.index)

    #     # Align and subtract, interpolate trend for missing values in original series index
    #     aligned_original, aligned_trend = series.align(trend_series, join='left')
    #     aligned_trend = aligned_trend.interpolate(method='time').ffill().bfill()

    #     diff_series = aligned_original - aligned_trend

    #     def prediction_lambda(forcast_df):
    #         degree = 2  # Set the degree of the polynomial; adjust as needed
    #         model = Pipeline([
    #             ('poly', PolynomialFeatures(degree=degree)),
    #             ('linear', LinearRegression())
    #         ])
    #         ts = np.arange(len(aligned_trend))
    #         model.fit(ts[:, np.newaxis], aligned_trend.values)
    #         future_ts = np.arange(len(aligned_trend), len(aligned_trend) + len(forcast_df))
    #         future_trend = model.predict(future_ts[:, np.newaxis])
    #         return forcast_df + future_trend
    #     return diff_series.dropna(), aligned_trend, prediction_lambda

    # def _stationarize(self, series: pd.Series, debug: bool = False):
    #     """Applies the chosen detrending method and returns the stationary series and trend component."""
    #     stationary_series, trend_component = None, None
    #     if self.detrending_method == 'diff_stationary':
    #         stationary_series, prediction_lambda = self._de_trending_by_differencing(series)
    #         trend_component = {'type': 'difference', 'order': self.diff_order, 'prediction_lambda': prediction_lambda}
    #     elif self.detrending_method == 'lowess_stationary':
    #         stationary_series, trend, prediction_lambda = self._de_trending_by_lowess(series)
    #         trend_component = {'type': 'lowess', 'trend_series': trend, 'prediction_lambda': prediction_lambda}
    #     elif self.detrending_method == 'prophet_stationary':
    #         stationary_series, yhat, prediction_lambda = self._prophet_stationalization(series)
    #         trend_component = {'type': 'prophet', 'yhat_series': yhat, 'prediction_lambda': prediction_lambda}
    #     else:
    #         raise ValueError("Invalid detrending method") # Should be caught in init
        
    #     if debug and stationary_series is not None:
    #         fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    #         axs[0].plot(series, label='Original Series')
    #         if trend_component['type'] == 'difference':
    #             axs[0].plot(stationary_series, label='Stationarized Series')
    #             axs[0].legend()
    #         elif trend_component['type'] == 'lowess':
    #             axs[0].plot(trend_component['trend_series'], label='Trend Component')
    #             axs[0].legend()
    #         elif trend_component['type'] == 'prophet':
    #             axs[0].plot(trend_component['yhat_series'], label='Trend Component')
    #             axs[0].legend()
    #         axs[1].plot(trend_component['trend_series'], label='Trend Component')
    #         axs[1].legend()
    #         plt.show()

    #     return stationary_series, trend_component

    def _preprocess_data(self, df: pd.DataFrame, tic: str, value_col: str = 'close'):
        tic_df = df[df['tic'] == tic][['date', value_col]]
        if not isinstance(tic_df.index, pd.DatetimeIndex):
            tic_df['date'] = pd.to_datetime(tic_df['date'])
            tic_df = tic_df.set_index('date', drop=True) # Keep date column if needed
        
        # if tic_df.index.inferred_freq is None:
        #     inferred_freq = pd.infer_freq(tic_df.index)
        #     if inferred_freq:
        #         tic_df = tic_df.asfreq(inferred_freq)
        #         # Forward fill NaNs introduced by asfreq
        #         tic_df = tic_df.ffill()#.bfill() # Forward fill is usually safer for financial data
        #         print(f"Info: Inferred frequency '{inferred_freq}' for {tic} and applied ffill.")
        #     else:
        #         print(f"Warning: Could not infer frequency for {tic}. ARIMA/Prophet might be unreliable. Skipping ticker.")
        #         # return None # Skip if frequency cannot be determined
        return tic_df
    
    def _check_white_noise(self, series: pd.Series, lags):
        """
        Performs the Ljung-Box test for white noise on the series residuals.
        Null Hypothesis (H0): The data are independently distributed (i.e., white noise).
        Returns the p-value of the test. A small p-value (< 0.05) suggests
        rejecting the null hypothesis, meaning the series is likely not white noise.
        """
        try:
            series_clean = series.dropna()
            if len(series_clean) < lags + 2: # Need enough data points for the test
                print(f"Warning: Insufficient data ({len(series_clean)} points) for Ljung-Box test with {lags} lags.")
                return 1.0 # Assume white noise if not enough data to test

            # Perform the Ljung-Box test
            lb_test_result = acorr_ljungbox(series_clean, lags=[lags], return_df=True)
            p_value = lb_test_result['lb_pvalue'].iloc[0]
            return p_value
        except Exception as e:
            print(f"Warning: Ljung-Box test failed: {e}")
            return 1.0 # Assume white noise if test fails

    def fit(self, series: pd.DataFrame, debug: bool = False):
        """
        Fit the strategy to the data.
        """
 
        ticker_models = None    
        # Ensure we have enough data after potential cleaning
        if len(series) < (self.p + self.q + 10): # Heuristic check
            print(f"Warning: Insufficient data for {series} after processing ({len(series)} points). Skipping ticker.")
            return ticker_models

        try:
            # Check stationarity after transformation (optional, for logging)
            # model = pm.auto_arima(series.values, 
            #           start_p=1, start_q=1,
            #           max_p=14, max_q=7,
            #           d=None,            # Let auto_arima determine the best order of differencing
            #           seasonal=False,    # Set to True if your data is seasonal
            #           stepwise=True,     # Enables stepwise search for faster performance
            #           suppress_warnings=True,
            #           error_action='ignore')  # Avoids halting on non-critical errors

            # print(model.summary())
            # model = ARIMA(series.values, order=model.order) 
            model = ARIMA(series.values, order=(self.p, self.d, self.q)) 
            model_fit = model.fit()

            residue = model_fit.resid
            residue_df = pd.DataFrame(residue, index=series.index)

            lags = self.p + self.q
            p_value = self._check_white_noise(residue_df, lags=lags)

            if debug:
                print('----------------------------------------------------------------')
                print(f"Ljung-Box p-value for residuals (lags={lags}): {p_value:.4f}. \n",
                    f"Reject H0. i.e. Residue is not white noise ? {p_value < 0.05}")
                print('----------------------------------------------------------------')

            if debug:
                print(model_fit.summary())
                arima_fit = model_fit.fittedvalues    
                df_arima = pd.DataFrame(arima_fit, index=series.index)
                
                fig, ax = plt.subplots(1, 1, figsize=(12, 6), tight_layout=True)
                ax.plot(series, label='Original Series')
                ax.plot(df_arima, label='ARIMA Fit')
                ax.grid(True)
                ax.legend()
                fig.suptitle('ARIMA Model Fit')
                plt.show()

            ticker_models = {'arima_model': model_fit}
                            
        except Exception as e:
            print(f"Error processing ticker: {e}")

        return ticker_models, p_value
    
    def play(self, df: pd.DataFrame, value_col: str = 'close', rounds = 5, use_last_n=-1) -> pd.DataFrame:
        # We fit till TRADE_START_DATE, and predict from TRADE_START_DATE to TRADE_END_DATE
        N = len(df[df['date'] >= TRADE_START_DATE])

        results = {}
        for tic in self.tics: # do analysis for each ticker
            tic_df = df[df['tic'] == tic]
            strategy = {
                'buy_volume' : [],
                'sell_volume' : []
            }
            
            predictions = []
            dates = []
            targets = []
            confidences = []

            rounds = rounds if rounds > 0 else N

            start = 0 if use_last_n < 0 else max(0, len(tic_df) - N - use_last_n)
            pbar = tqdm(range(rounds), desc=f'{tic}')
            for d in pbar:
                train_tic_df = self._preprocess_data(tic_df.iloc[start:-N+d], tic, value_col)
                if train_tic_df is None:
                    continue
            
                last_14_days_train = train_tic_df[value_col].iloc[-14:]
                price_std = last_14_days_train.std()


                ticker_models, residue_p_value = self.fit(train_tic_df, debug=False)
                if ticker_models is None:
                    assert False, f'model was not constructed.'

                date = tic_df.iloc[-N + d]['date']
                dates.append(date)
                pred = ticker_models['arima_model'].forecast(1)[0]
                target = tic_df.iloc[-N + d][value_col]
                diff = target - pred
                targets.append(target)

                predictions.append(pred)
                if diff > price_std * 0.5:
                    strategy['sell_volume'].append(1)
                    strategy['buy_volume'].append(0)
                elif diff < - price_std * 0.5:
                    strategy['sell_volume'].append(0)
                    strategy['buy_volume'].append(1)
                else:
                    strategy['sell_volume'].append(0)
                    strategy['buy_volume'].append(0)
                
                confidences.append(np.clip(abs(diff)/price_std, 0.0, 1.0))

                pbar.set_postfix({"date": str(date).split(' ')[0], "Diff": diff, "Conf": confidences[-1], "Chi-p_value":residue_p_value})

                # print('----------------------------------------------------------------')
                # print(f"{str(date).split(' ')[0]} [{tic}] {train_tic_df.shape}. confidence: {confidences[-1]}. diff:{diff}, price std:{price_std}")
                # print('----------------------------------------------------------------')
                # print(f'{date}: ')

            output_df = pd.DataFrame({
                'date': pd.to_datetime(dates),
                'target': targets,
                'prediction': predictions,
                'buy': strategy['buy_volume'],
                'sell': strategy['sell_volume'],
                'confidence': confidences
            })
            
            # buy = pd.Series(strategy['buy_volume'], dtype=bool)
            # sell = pd.Series(strategy['sell_volume'], dtype=bool)
            # fig, ax = plt.subplots(1,1, figsize=(13,6), tight_layout=True)
            # ax.plot(output_df['date'], output_df['target'], label='target')
            # ax.plot(output_df['date'], output_df['prediction'], label='prediction')
            # ax.plot(output_df['date'][buy], output_df['target'][buy], 'r^')
            # ax.plot(output_df['date'][sell], output_df['target'][sell], 'gv')
            # ax.legend()
            # plt.show()
            # import pdb; pdb.set_trace()
            # print(df['date'].iloc[-N:])

            results[tic] = output_df
        return results
    
    def simulate_trading(self, df_signals: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
        """
        Simulates a trading strategy based on buy/sell signals and confidence.

        Args:
            df_signals (pd.DataFrame): DataFrame with trading signals and price data.
                Required columns:
                - 'date': The date of the data point (should be sortable).
                - 'target': The price of the asset at that date (used for transactions).
                - 'buy': Binary signal (1 for buy, 0 otherwise).
                - 'sell': Binary signal (1 for sell, 0 otherwise).
                - 'confidence': The fraction (0 to 1) of available capital/holdings
                            to use for the transaction.
                Constraint: 'buy' and 'sell' cannot both be 1 for the same date.
            initial_capital (float): The starting amount of cash.

        Returns:
            pd.DataFrame: A DataFrame with the same index as df_signals, containing:
                - 'holdings': The number of shares held at the end of each day (int).
                - 'capital': The total portfolio value (cash + holdings value)
                            at the end of each day (float).
                - 'cash': The cash available at the end of each day (float). # Added for clarity
        """
        # --- Input Validation (Basic) ---
        required_cols = ['date', 'target', 'buy', 'sell', 'confidence']
        if not all(col in df_signals.columns for col in required_cols):
            raise ValueError(f"Input DataFrame missing one or more required columns: {required_cols}")

        if not pd.api.types.is_numeric_dtype(df_signals['target']):
            raise TypeError("Column 'target' must be numeric.")
        if not pd.api.types.is_numeric_dtype(df_signals['buy']):
            raise TypeError("Column 'buy' must be numeric (0 or 1).")
        if not pd.api.types.is_numeric_dtype(df_signals['sell']):
            raise TypeError("Column 'sell' must be numeric (0 or 1).")
        if not pd.api.types.is_numeric_dtype(df_signals['confidence']):
            raise TypeError("Column 'confidence' must be numeric.")

        if (df_signals['buy'] + df_signals['sell'] > 1).any():
            raise ValueError("Buy and Sell signals cannot both be 1 on the same day.")
        if (df_signals['confidence'] < 0).any() or (df_signals['confidence'] > 1).any():
            raise ValueError("Confidence must be between 0 and 1.")
        if initial_capital < 0:
            raise ValueError("Initial capital cannot be negative.")

        # --- Initialization ---
        # Sort by date to ensure chronological processing
        df = df_signals.sort_values(by='date').copy()
        df.reset_index(drop=True, inplace=True) # Ensure default integer index for iloc

        holdings = 0
        cash = initial_capital
        results = [] # List to store daily results


        brokerage_mult = 0.0005
        brokerage_add = 1

        ## holding strategy ##
        row = df.iloc[1]
        price = row['target']
        holding_stock_count = (initial_capital - brokerage_add) / (price * (1 + brokerage_mult))
        assert holding_stock_count * price < initial_capital, f'Initial:{initial_capital}. holding_strategy_capital: {holding_stock_count * price}. price: {price}'

        # --- Simulation Loop ---
        # auto regressive. Hence starting from 1 and use previous days signals.
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            price = row['target']
            buy_signal = prev_row['buy'] # signal generated from previous day should be used now.
            sell_signal = prev_row['sell'] # signal generated from previous day should be used now.
            confidence = prev_row['confidence']

            shares_to_buy = 0
            shares_to_sell = 0
            
            # -------- Trading Logic --------
            # If buy signal then buy amount proportionate to confidence. 
            # Similarly if sell signal then sell amount proportionate to the confidence
            # -------------------------------
            #  
            if buy_signal == 1 and price > 0: # Check price > 0 to avoid division by zero
                # Buy with 'confidence' percent of available cash
                amount_to_invest = cash * confidence
                shares_to_buy = amount_to_invest / price
                
                actual_cost = (brokerage_mult + 1)* shares_to_buy * price + brokerage_add 
                if actual_cost > cash:
                    shares_to_buy = (cash - brokerage_add) / (price * (1 + brokerage_mult))
                    actual_cost = (brokerage_mult + 1)* shares_to_buy * price + brokerage_add 

                if shares_to_buy > 1e-3:
                    cash -= actual_cost
                    holdings += shares_to_buy

            elif sell_signal == 1 and holdings > 0:
                shares_to_sell = holdings * confidence 
                
                if shares_to_sell > 1e-3:
                    fee = brokerage_mult * shares_to_sell * price + brokerage_add
                    cash += shares_to_sell * price - fee
                    holdings -= shares_to_sell

            # --- Record State ---
            # Calculate total capital at the end of the day
            current_total_capital = cash + (holdings * price)
            print(f'{str(row['date']).split(' ')[0]}:buy:{buy_signal}, sell:{sell_signal}, confidence:{confidence} -> Hold: {holdings}. Current: {current_total_capital}')

            results.append({
                'date': row['date'],
                'holdings': holdings,
                'cash': cash,
                'arima_strategy_capital': current_total_capital,
                'holding_strategy_capital' : holding_stock_count * price,
                'shares_to_sell' : shares_to_sell,
                'shares_to_buy' : shares_to_buy,
                'price' : price
            })

        # --- Create Output DataFrame ---
        output_df = pd.DataFrame(results)
        # Set date as index if desired (matches original structure often)
        # output_df.set_index('date', inplace=True)

        return output_df


from finrl.config import (
    INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
)


def plot_strategy(df, tic):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    buy = pd.Series(df['buy'], dtype=bool)
    sell = pd.Series(df['sell'], dtype=bool)

    # Plot 'a' and 'b' on the left y-axis (ax1)
    ax1.plot(df['date'], df['target'], '-', label='target',)
    ax1.plot(df['date'], df['prediction'], '--', label='prediction')
    ax1.plot(df['date'][buy], df['target'][buy], 'r^', label='buy')
    ax1.plot(df['date'][sell], df['target'][sell], 'gv', label='sell')

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Target vs Predition', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.legend(loc='upper left')
    ax1.grid()

    # Create a second y-axis for 'c'
    # ax2 = ax1.twinx()
    # import pdb; pdb.set_trace()
    ax2.grid()
    ax2.plot(df['date'], df['arima_strategy_capital'], '--', label='arima')
    ax2.plot(df['date'], df['holding_strategy_capital'], '-', label='holding capital')
    ax2.set_ylabel('arima vs holding', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right')
    fig.suptitle(f'{tic}')
    plt.show()

def set_dates():
    global TRAIN_START_DATE, TRAIN_END_DATE, TEST_START_DATE, TEST_END_DATE, TRADE_START_DATE, TRADE_END_DATE
    TRAIN_START_DATE = '2018-01-01'
    TRAIN_END_DATE = '2024-12-31'
    # TEST_START_DATE = '2023-01-01'
    # TEST_END_DATE = '2023-12-31'
    TRADE_START_DATE = '2025-02-01'
    TRADE_END_DATE = '2025-12-31'

def test():
    set_dates()
    df = download_data(TRAIN_START_DATE, TRADE_END_DATE, config_tickers.DOW_3_TICKER)
    df = data_processing(df)
    tics = config_tickers.DOW_3_TICKER #['MSFT', 'META']
    arima = Arima(p=15, d=1, q=5, tics=tics)
    res = arima.play(df, rounds=-1, use_last_n=200)

    df['date'] = pd.to_datetime(df['date'])
    
    for tic in tics:
        p_df = res[tic].copy()
        out = arima.simulate_trading(p_df, 10000)

        p_df['arima_strategy_capital'] = out['arima_strategy_capital']
        p_df['holding_strategy_capital'] = out['holding_strategy_capital']

        print('----------------------------------------------------------------')
        print(f'\t\t {tic}')
        print('----------------------------------------------------------------')
        print(out)
        plot_strategy(p_df, tic)
        
        last_date = out['date'].iloc[0]
        first_date = out['date'].iloc[-1]
        _df = df[(df['date'] >= last_date) & (df['date'] <= first_date)]
        trade_df = out.copy()
        trade_df['capital'] = out['arima_strategy_capital']
        plot_candlestick(stock_df=_df, trade_df=trade_df, tic=tic)
        # import pdb; pdb.set_trace()



if __name__ == '__main__':
    test()


