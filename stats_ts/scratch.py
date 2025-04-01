import yfinance as yf
import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.nonparametric.smoothers_lowess import lowess
import numpy as np
from prophet import Prophet
# inclue pacf and acf
from statsmodels.tsa.stattools import pacf, acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# import arima
from statsmodels.tsa.arima.model import ARIMA

import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning
warnings.simplefilter('ignore', InterpolationWarning)

def get_data(symbol):
    df = pd.read_csv(os.path.join('YFin', f'{symbol}.csv'), index_col=0, parse_dates=True)
    return df


def check_stationarity(df):
    adf_result = adfuller(df, regression='c', autolag='AIC', maxlag=30)
    print(f'adfuller: p-value: {adf_result[1]}. {"Stationary" if adf_result[1] < 0.05 else "Non-Stationary"}')

    kpss_result = kpss(df, regression='c', nlags=30)
    print(f'kpss: p-value: {kpss_result[1]}. {"Non-Stationary" if kpss_result[1] < 0.05 else "Stationary"}')
    return adf_result[1], kpss_result[1]
    

def prophet_stationalization(df, debug=False):
    model = Prophet()
    prophet_df = pd.DataFrame({
        'ds': df.index,
        'y': df.values[:,0]
    })
    model.fit(prophet_df)
    forecast = model.predict(prophet_df)
    trend = forecast['trend']
    yhat = forecast['yhat']
    residuals = prophet_df['y'] - yhat

    adf_result, kpss_result = check_stationarity(residuals)
    if debug:
        fig, ax = plt.subplots(2, 1, figsize=(15, 6), tight_layout=True)
        ax[0].plot(prophet_df['ds'], prophet_df['y'], label='Original Data')
        ax[0].plot(prophet_df['ds'], yhat, label='Forecast')
        ax[0].plot(prophet_df['ds'], forecast['yhat_lower'], label='yhat_lower')
        ax[0].plot(prophet_df['ds'], forecast['yhat_upper'], label='yhat_upper')
        ax[1].plot(prophet_df['ds'], residuals,  '--', label='Residuals/Stationary Data')
        ax[0].legend()
        ax[1].legend()
        fig.suptitle(f'Prophet: ADF p-value: {adf_result}. KPSS p-value: {kpss_result}.')
        plt.show()

    return residuals, yhat, trend


def de_trending_by_differencing(df, debug=False, order=1):
    # Apply differencing 'order' times
    diff_df = df.copy()
    for _ in range(order):
        diff_df = diff_df.diff().dropna()
    
    if debug:
        fig, ax = plt.subplots(2, 1, figsize=(15, 6), tight_layout=True)
        ax[0].plot(df.index, df, label='Original Data')
        ax[1].plot(df.index[1:], diff_df, label='Differenced Data')
        ax[0].legend()
        ax[1].legend()
        adf_result, kpss_result = check_stationarity(diff_df)
        fig.suptitle(f'Differencing: ADF p-value: {adf_result}. KPSS p-value: {kpss_result}.')
        plt.show()
    return diff_df

def de_trending_by_lowess(df, debug=False):
    lowess_df = lowess(df.values[:,0], df.index, frac=0.1)
    diff_df = df.copy()
    diff_df.iloc[:,0] = df.iloc[:,0] - lowess_df[:, 1] 
    if debug:
        fig, ax = plt.subplots(2, 1, figsize=(15, 6), tight_layout=True)
        ax[0].plot(df.index, df, label='Original Data')
        ax[0].plot(df.index, lowess_df[:, 1], label='Trend')
        ax[1].plot(df.index, diff_df, label='De-trended Data')
        ax[0].legend()
        ax[1].legend()
        adf_result, kpss_result = check_stationarity(diff_df)
        fig.suptitle(f'LOESS: ADF p-value: {adf_result}. KPSS p-value: {kpss_result}.')
        plt.show()
    return diff_df

def analyze(_df):
    # check if the data is stationary
    diff_stationary = de_trending_by_differencing(_df, order=1)
    diff_lowess_stationary = de_trending_by_lowess(_df)
    prophet_stationary, yhat, trend = prophet_stationalization(_df)

    fig, ax = plt.subplots(4, 1, figsize=(15, 8), tight_layout=True)
    ax[0].plot(diff_stationary, label='Differenced Data')
    adf_result, kpss_result = check_stationarity(diff_stationary)
    ax[0].set_title(f'ADF p-value: {adf_result}. KPSS p-value: {kpss_result}.')
    ax[1].plot(diff_lowess_stationary, label='Lowess Differenced Data')
    adf_result, kpss_result = check_stationarity(diff_lowess_stationary)
    ax[1].set_title(f'ADF p-value: {adf_result}. KPSS p-value: {kpss_result}.')
    ax[2].plot(prophet_stationary, label='Prophet Stationalized Data')
    adf_result, kpss_result = check_stationarity(prophet_stationary)
    ax[2].set_title(f'ADF p-value: {adf_result}. KPSS p-value: {kpss_result}.')
    ax[3].plot(_df, label='Original Data')
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[3].legend()
    plt.show()

    # import pdb; pdb.set_trace()

    # plt.rcParams.update({'figure.figsize':(10,6), 'figure.dpi':120})
    fig, ax = plt.subplots(3, 1, figsize=(15, 6), tight_layout=True)    
    plot_pacf(diff_stationary, ax=ax[0]); ax[0].set_title('Differenced Data')
    plot_pacf(diff_lowess_stationary, ax=ax[1]); ax[1].set_title('Lowess Differenced Data')
    plot_pacf(prophet_stationary, ax=ax[2]); ax[2].set_title('Prophet Stationalized Data')
    plt.show()

    stationary = diff_stationary
    sd, yhat, trend = prophet_stationalization(stationary, debug=True)
    # Convert to numpy array and create DataFrame
    stationary = pd.DataFrame(sd.to_numpy(), index=stationary.index)
    
    lag = 14
    
    # Fit model on the series till 30 days before the end
    stationary_lag = stationary.iloc[:-lag]
    fig, ax = plt.subplots(2, 1, figsize=(15, 6), tight_layout=True)
    plot_pacf(stationary_lag, ax=ax[0], lags=365)
    plot_acf(stationary_lag, ax=ax[1], lags=365)
    ax[0].set_title('PACF of Differenced Data')
    ax[1].set_title('ACF of Differenced Data')
    plt.show()


    model = ARIMA(np.array(stationary_lag.values).flatten(), order=(lag, 0, lag))
    model_fit = model.fit(method_kwargs={'maxiter': 1000, 'disp': True})
    print(model_fit.summary())

    arima_fit = model_fit.fittedvalues    
    df_arima = pd.DataFrame(arima_fit, index=stationary_lag.index)
    fig, ax = plt.subplots(1, 1, figsize=(15, 6), tight_layout=True)
    ax.plot(stationary, label='Stationary Data')
    ax.plot(df_arima, label='ARIMA Fit')
    ax.legend()
    fig.suptitle('ARIMA Model Fit')
    plt.show()


    # Get predictions for next  days
    forecast = model_fit.forecast(steps=lag)
    last_date = stationary_lag.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=lag, freq='B')
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame(forecast, index=forecast_dates)
    
    # Plot the results
    fig, ax = plt.subplots(1, 1, figsize=(15, 6), tight_layout=True)
    ax.plot(stationary, label='Stationary Data')
    ax.plot(df_arima, label='ARIMA Fit')
    ax.plot(forecast_df, label='Forecast', linestyle='--', color='red')
    ax.legend()
    fig.suptitle('ARIMA Model Fit and Forecast')
    plt.show()
    
    print("\nForecast for next 30 days:")
    print(forecast_df)

    yhat_lag = yhat.iloc[-lag:]
    forecast_df.iloc[:,0] = forecast_df.iloc[:,0] + yhat_lag.to_numpy()
    
    # Get the last value of the original series before forecasting
    last_original_value = _df.iloc[-(lag+1)].values[0]  # -lag because we forecasted lag days ahead

    # Recover the original values by cumulative sum of differences
    recovered_forecast = last_original_value + forecast_df.cumsum()

    fig, ax = plt.subplots(1, 1, figsize=(15, 6), tight_layout=True)
    ax.plot(_df, label='Original Data')
    ax.plot(recovered_forecast, label='Recovered Forecast', linestyle='--', color='red')
    ax.legend()
    fig.suptitle('Original Data and Recovered Forecast')
    plt.show()


if __name__ == '__main__':
    df = yf.download('AAPL', auto_adjust=True, start='2018-01-01')
    # Set the frequency of the index to business days ('B')
    # df.index = pd.DatetimeIndex(df.index)

    # Set the frequency to business days ('B')
    # df = df.asfreq('B')
    # df.index = pd.DatetimeIndex(df.index).to_period('B').to_timestamp()
    analyze(df['Close'])

    