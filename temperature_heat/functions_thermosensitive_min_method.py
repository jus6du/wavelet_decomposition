from datetime import timedelta, date
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

 
def charge_min(time_series, year): #récupère tous les minimums des couples (jour, heure) d'une time series
    charge_min = pd.DataFrame()
    first_january = pd.Timestamp(year=year, month=1, day=1)

    index = pd.date_range(start=first_january, periods=len(time_series), freq='h')
    time_series.index = index
    min_by_day_hour = time_series.groupby([time_series.index.dayofweek, time_series.index.hour]).min()

    return min_by_day_hour

def thermosensitive_ts(time_series, year):
    charge_min_matrix=charge_min(time_series, year)
    therm_ts = time_series.copy()
    for item in therm_ts.index:
            therm_ts.loc[item,:] -= charge_min_matrix.loc[(item.dayofweek, item.hour),:]
    return therm_ts

def non_thermosensitive_ts(time_series,year):
    charge_min_matrix=charge_min(time_series, year)
    non_therm_ts = time_series.copy()
    for item in non_therm_ts.index:
            non_therm_ts.loc[item,:] = charge_min_matrix.loc[(item.dayofweek, item.hour),:]
    return non_therm_ts

def weighted_mean(ts):
      ## Mise en place d'une moyenne glissante sur la journée non pondérée pour les données de fit du modèle
    rolling_window = ts.rolling(window='24H')
    weight = pd.Series(range(1,25))
    weighted_mean = rolling_window.apply(lambda x: (x*weight).sum()/weight.sum(), raw=True)
    return weighted_mean


def piecewise_linear_regression(temperatures, electricity_consumption):
    # Fit linear regression for temperatures below 18°C
    indices_below_18 = temperatures < 18
    X_below_18 = temperatures[indices_below_18]
    y_below_18 = electricity_consumption[indices_below_18]
    coefficients_below_18 = np.polyfit(X_below_18, y_below_18, 1)

    # Fit linear regression for temperatures above 22°C
    indices_above_22 = temperatures > 22
    X_above_22 = temperatures[indices_above_22]
    y_above_22 = electricity_consumption[indices_above_22]
    coefficients_above_22 = np.polyfit(X_above_22, y_above_22, 1)

    return coefficients_below_18, coefficients_above_22

def plot_regression(temperatures, electricity_consumption, coefficients_below_18, coefficients_above_22):
     # Generate temperatures for plotting
    temperatures_plot = np.linspace(min(temperatures), max(temperatures), 100)

    # Predict electricity consumption based on temperature
    # Predict electricity consumption based on temperature
    predicted_consumption_below_18 = coefficients_below_18[0] * temperatures_plot + coefficients_below_18[1]
    predicted_consumption_below_18[temperatures_plot >= 18] = None

    predicted_consumption_above_22 = coefficients_above_22[0] * temperatures_plot + coefficients_above_22[1]
    predicted_consumption_above_22[temperatures_plot <= 22] = None

    # Plot the data and predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(temperatures, electricity_consumption, color='blue', label='Actual Data')
    plt.plot(temperatures_plot, predicted_consumption_below_18, color='red', linestyle='--', label='Prediction Below 18°C')
    plt.plot(temperatures_plot, predicted_consumption_above_22, color='green', linestyle='--', label='Prediction Above 22°C')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Electricity Consumption')
    plt.title('Electricity Consumption Prediction based on Temperature')
    plt.legend()
    plt.grid(True)
    plt.show()
    return