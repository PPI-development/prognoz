import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from data_loader import load_pest_data, load_weather_data


def forecast_population(pest_name, region_name):
    """
    Выполняет прогноз численности вредителей с использованием моделей ARIMA и RandomForest.

    :param pest_name: Название вредителя (например, "locustitalian" или "breadbeetle").
    :param region_name: Название региона (например, "акмолинская область").
    :return: JSON-объект с прогнозом на 10 лет.
    """
    pest_data = load_pest_data(pest_name)
    if pest_data is None:
        return None

    pest_data.columns = [col.lower() for col in pest_data.columns]
    if "region" not in pest_data.columns:
        first = pest_data.columns[0]
        pest_data = pest_data.rename(columns={first: "region"})

    # Приводим данные в длинный формат
    df_long = pest_data.melt(id_vars=["region"], var_name="year", value_name="population")
    df_long["year"] = df_long["year"].astype(int)
    df_long["population"] = pd.to_numeric(df_long["population"], errors="coerce")
    df_long.dropna(subset=["population"], inplace=True)

    # Выбираем данные по региону
    df_region = df_long[df_long["region"].str.lower() == region_name.lower()]
    if df_region.empty:
        return None

    # Получаем последний год с данными
    last_year = df_region["year"].max()

    # Загружаем погодные данные
    weather_df = load_weather_data(region_name)
    if weather_df is None:
        return None

    # Объединяем данные по году
    merged = pd.merge(df_region, weather_df, on="year", how="inner")
    if merged.shape[0] < 5:
        return None  # Недостаточно данных для прогноза

    # Выбираем погодные факторы
    features = ["temperature_2m", "relative_humidity_2m", "precipitation"]
    X = merged[features].values
    y = merged["population"].values

    # --- ARIMA ПРОГНОЗ ---
    try:
        arima_model = sm.tsa.ARIMA(y, order=(1, 1, 1))
        arima_fitted = arima_model.fit()
        arima_forecast = arima_fitted.forecast(steps=10)
    except:
        arima_forecast = np.full(10, y[-1])  # Если ошибка, берем последнее значение

    # --- RANDOM FOREST ПРОГНОЗ ---
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_scaled, y)

        avg_features = np.mean(X_scaled, axis=0).reshape(1, -1)
        rf_forecast = rf_model.predict(avg_features)
        rf_forecast = np.full(10, rf_forecast[0])
    except:
        rf_forecast = np.full(10, y[-1])  # Если ошибка, берем последнее значение

    # Усредняем прогноз ARIMA и RandomForest
    final_forecast = (arima_forecast + rf_forecast) / 2
    final_forecast = np.maximum(final_forecast, 0)  # Убираем отрицательные значения

    forecast_years = list(range(last_year + 1, last_year + 11))

    return {"years": forecast_years, "population": final_forecast.tolist()}
