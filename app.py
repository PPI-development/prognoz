#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import requests
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, render_template, request, jsonify
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from plotly.offline import plot
import plotly.graph_objs as go
from plotly.subplots import make_subplots

app = Flask(__name__)

# =======================
# 1. Конфигурация и константы
# =======================

# Корневая папка с данными
data_dir = "data"

# Папка для погодных данных; сюда будут сохраняться JSON-файлы, полученные через Open-Meteo
weather_dir = os.path.join(data_dir, "weather")
os.makedirs(weather_dir, exist_ok=True)

# Конфигурация вредителей (ключи – идентификаторы вредителей, например, "breadbeetle_ZKO" и "locust_italian")
from collections import OrderedDict
config = OrderedDict()

config['breadbeetle_ZKO'] = {
    'order': (0, 1, 1),
    'seasonal_order': (1, 0, 1, 2),
    's_month': 1,
    'e_month': 3,
    'train_size': 0.8,
    'pest': 'breadbeetle',
    'area': 'ZKO',
    'population_type': 'adult',
    'target_var': 'Weighted sum',
    'features': ['QV2M', 'T2MDEW', 'T2MWET'],
    'thresholds': [2.5, 2.5, 2.5],
    'imputer_features': None,
    'coords': [51.14, 51.22],
    'json': "weather_data_ZKO.json"
}

config['locust_italian'] = {
    'order': (1, 1, 1),
    'seasonal_order': (0, 1, 1, 12),
    's_month': 5,
    'e_month': 9,
    'train_size': 0.8,
    'pest': 'locust_italian',
    'area': 'KZ',
    'population_type': 'adult',
    'target_var': 'population',
    'features': [
        "temperature_2m", "relative_humidity_2m", "precipitation",
        "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm",
        "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm"
    ],
    'thresholds': [10, 10, 0.5, 0, 0, 0, 0],
    'imputer_features': None,
    'coords': [45.0, 70.0],
    'json': "weather_data_KZ.json"
}

# Словарь для отображения имен вредителей и регионов (используется при построении графиков)
name_dict = {
    'breadbeetle_ZKO': {'pest': 'Хлебные жуки', 'area': 'Западно-Казахстанская Область'},
    'locust_italian': {'pest': 'Итальянский прус', 'area': 'Казахстан'}
}

# =======================
# 2. Вспомогательные функции
# =======================

def robust_geomean(data):
    return np.exp(np.mean(np.log(1 + np.array(data)))) - 1

def effective_sum(temps, thresh=5):
    temps = np.array(temps)
    return np.sum(temps[temps > thresh])

def get_monthly_avg(data_obj, year, month, average=True, expectation=np.nanmean):
    year = str(year)
    month = str(month).zfill(2)
    temps = []
    for day in range(1, 32):
        key = year + month + str(day).zfill(2)
        if key in data_obj:
            temps.append(data_obj[key])
    if average:
        return expectation(temps)
    return temps

def get_daily_val(data_obj, year, month, day):
    year = str(year)
    month = str(month).zfill(2)
    day = str(day).zfill(2)
    key = year + month + day
    return data_obj.get(key, None)

def calc_hitrate(actual, pred, backtest=True, percentage=True):
    actual, pred = list(actual), list(pred)
    hits = []
    for i in range(1, len(actual)):
        act_movement = np.sign(actual[i] - actual[i - 1])
        if backtest:
            pred_movement = np.sign(pred[i] - actual[i - 1])
        else:
            pred_movement = np.sign(pred[i] - pred[i - 1])
        hits.append(1 if act_movement == pred_movement else 0)
    hitrate = np.sum(hits) / (len(actual) - 1)
    return hitrate * 100 if percentage else hitrate

def calc_normbias(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(2 * (y_pred - y_true) / (y_pred + y_true))

def calc_mpe(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(100 * (y_pred - y_true) / y_true)

# =======================
# 3. Функция загрузки погодных данных через Open-Meteo
# =======================

def download_openmeteo(coords, jsonname, start_date, end_date):
    lat, lon = coords
    # Определяем список ежедневных переменных (можно изменить по необходимости)
    daily_vars = "temperature_2m,relative_humidity_2m,precipitation,soil_temperature_0_to_7cm,soil_temperature_7_to_28cm,soil_moisture_0_to_7cm,soil_moisture_7_to_28cm"
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": daily_vars,
        "timezone": "auto"
    }
    response = requests.get(base_url, params=params, timeout=30)
    content = response.json()
    filepath = os.path.join(weather_dir, jsonname)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(content, f)
    return filepath

# =======================
# 4. Загрузчики моделей
# =======================

class AR_ModelLoader:
    def __init__(self, order, s_order, s_month, e_month, train_size=0.8, scale=True, yearly=True, include_test=True):
        self.order = order
        self.s_order = s_order
        self.s_month = s_month
        self.e_month = e_month
        self.train_size = train_size
        self.scale = scale
        self.yearly = yearly
        self.include_test = include_test

    def load_data(self, pest, area, population_type, target_var, dirpath=data_dir, check_temp_first=False):
        self.pest = pest
        self.area = area
        self.population_type = population_type
        self.target_var = target_var
        self.dirpath = dirpath
        filepath = os.path.join(dirpath, f"{pest}_{area}.xlsx")
        if self.include_test:
            filepath = filepath.replace("_train", "")
        sheet_name = 0 if population_type == 'adult' else 1
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        if 'Date' in df.columns:
            df.index = pd.to_datetime(df['Date'])
        df['y'] = df[target_var].astype(float)
        self.df = df[['y']]
        return self.df

    def extract_features(self, factors, thresholds=[], expectation=np.nanmean, agg_strategy='effective_sum', only_last_year=False):
        self.factors = factors
        self.thresholds = thresholds
        # Формирование пути к JSON-файлу с погодными данными (определяется по конфигурации)
        key = self.pest + '_' + self.area
        if key in config:
            jsonpath = os.path.join(weather_dir, config[key]['json'])
        else:
            jsonpath = os.path.join(weather_dir, "default_weather.json")
        with open(jsonpath, encoding='utf-8') as f:
            data = json.load(f)
        df = self.df
        X = []
        # Для каждого временного момента берем агрегированные погодные признаки за предыдущий год
        for i in range(len(df)):
            if not only_last_year:
                year = df.index[i].year - 1
            else:
                year = df.index[i].year
            factor_vals = []
            for idx, factor in enumerate(factors):
                # Пример: если ключ factor присутствует в data['daily'], усредняем значения за месяц s_month (для упрощения)
                if 'daily' in data and factor in data['daily']:
                    data_obj = data['daily'][factor]
                    avg_val = get_monthly_avg(data_obj, year, self.s_month)
                else:
                    avg_val = np.nan
                factor_vals.append(avg_val)
            X.append(factor_vals)
        self.X = np.array(X)
        return self.X

    def crossval(self, conf_level=0.9, verbose=False, save_plots=False):
        df = self.df
        train_size = int(self.train_size * len(df))
        X = self.X
        y = df['y'].values
        y_pred = []
        for i in range(train_size, len(X)):
            X_train = X[:i]
            y_train = y[:i]
            X_test = X[i:i + 1]
            mod = sm.tsa.statespace.SARIMAX(y_train, order=self.order, seasonal_order=self.s_order, exog=X_train)
            results = mod.fit(disp=False)
            fcast = results.get_forecast(steps=1, exog=X_test)
            pred = fcast.predicted_mean.values[0]
            y_pred.append(pred)
        return y[:train_size], y[train_size:], y_pred

    def predict(self, n=1):
        X = self.X
        y = self.df['y'].values
        mod = sm.tsa.statespace.SARIMAX(y, order=self.order, seasonal_order=self.s_order, exog=X)
        results = mod.fit(disp=False)
        fcast = results.get_forecast(steps=n, exog=X[-1:])
        return fcast

class R_ModelLoader:
    def __init__(self, train_size=0.8, include_test=True, random_state=42, scale=False, regressor='rf', n_estimators=100, kernel='rbf', C=0.75):
        self.train_size = train_size
        self.include_test = include_test
        self.scale = scale
        if regressor == 'rf':
            self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        elif regressor == 'svr':
            self.model = SVR(kernel=kernel, C=C)

    def load_data(self, pest, area, population_type, target_var, dirpath=data_dir, check_temp_first=False):
        self.pest = pest
        self.area = area
        self.population_type = population_type
        self.target_var = target_var
        self.dirpath = dirpath
        filepath = os.path.join(dirpath, f"R_{pest}_{area}.xlsx")
        if self.include_test:
            filepath = filepath.replace("_train", "")
        sheet_name = 0 if population_type == 'adult' else 1
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        if 'Date' in df.columns:
            df.index = pd.to_datetime(df['Date'])
        df['y'] = df[target_var].astype(float)
        self.df = df[['y']]
        return self.df

    def extract_features(self, factors, thresholds=[], expectation=np.nanmean, agg_strategy='effective_sum', only_last_day=False):
        self.factors = factors
        self.thresholds = thresholds
        key = self.pest + '_' + self.area
        if key in config:
            jsonpath = os.path.join(weather_dir, config[key]['json'])
        else:
            jsonpath = os.path.join(weather_dir, "default_weather.json")
        with open(jsonpath, encoding='utf-8') as f:
            data = json.load(f)
        df = self.df
        X = []
        for i in range(1, len(df)):
            # Здесь вместо реального расчета, генерируются случайные значения – для демонстрации
            dummy_vals = [np.random.random() * 10 for _ in factors]
            X.append(dummy_vals)
        self.X = np.array(X)
        return self.X

    def crossval(self, verbose=False, save_plots=False):
        X = self.X
        y = self.df['y'].values[1:]
        split = int(self.train_size * len(X))
        y_pred = []
        for i in range(split, len(X)):
            X_train = X[:i]
            y_train = y[:i]
            self.model.fit(X_train, y_train)
            pred = self.model.predict(X[i:i + 1])[0]
            y_pred.append(pred)
        return y[:split], y[split:], y_pred

    def predict(self):
        X = self.X
        y = self.df['y'].values[1:]
        self.model.fit(X, y)
        fcast = self.model.predict(X[-1:])[0]
        return fcast

# =======================
# 5. Функции построения графиков (Plotly)
# =======================

def plot_data(df, pestarea_comb, population_type, add_var=None, var_name=""):
    info = name_dict.get(pestarea_comb, {"pest": pestarea_comb, "area": ""})
    pest = info["pest"]
    area = info["area"]
    if add_var is not None:
        fig = make_subplots(rows=2, cols=1)
        trace1 = go.Scatter(x=df.index, y=df['y'], marker=dict(color="blue"), name=f'Популяция {population_type}')
        trace2 = go.Scatter(x=df.index, y=add_var, marker=dict(color="green"), name=var_name)
        fig.add_trace(trace1, row=1, col=1)
        fig.add_trace(trace2, row=2, col=1)
        fig.update_layout(title=f"Объект: {pest}, Область: {area}", height=900, width=1200)
    else:
        fig = go.Figure(data=[go.Scatter(x=df.index, y=df['y'], marker=dict(color="blue"))],
                        layout=dict(title=f"Объект: {pest}, Область: {area}", height=900, width=1200))
    return plot(fig, output_type='div')

# =======================
# 6. Функции для итальянского пруса (locust)
# =======================

def load_locust_data():
    locust_file = os.path.join(data_dir, "locustitalian.csv")
    if os.path.exists(locust_file):
        df = pd.read_csv(locust_file)
        df.columns = [col.lower() for col in df.columns]
        if 'region' not in df.columns:
            df = df.rename(columns={df.columns[0]: 'region'})
        return df
    return None

def forecast_population_locust(region_name):
    """
    Прогноз для итальянского пруса: на основе исторических данных из locustitalian.csv
    и погодных данных (CSV-файл для выбранного региона, например, "region_name.csv" в weather_dir).
    Прогноз выполняется методом наименьших квадратов.
    """
    df = load_locust_data()
    if df is None:
        return None
    df.columns = [col.lower() for col in df.columns]
    if 'region' not in df.columns:
        df = df.rename(columns={df.columns[0]: 'region'})
    df_long = pd.melt(df, id_vars=['region'], var_name='year', value_name='population')
    df_long['year'] = df_long['year'].astype(int)
    df_long['population'] = pd.to_numeric(df_long['population'], errors='coerce')
    df_long = df_long.dropna(subset=['population'])
    current_year = datetime.now().year
    df_long = df_long[(df_long['year'] >= 1935) & (df_long['year'] <= current_year)]
    df_long['region'] = df_long['region'].str.strip().str.lower()
    region_key = region_name.lower().strip()
    if region_key in df_long['region'].unique():
        df_region = df_long[df_long['region'] == region_key]
    else:
        df_region = df_long
    df_region = df_region.sort_values('year')
    last_year = df_region['year'].max()
    weather_file = os.path.join(weather_dir, f"{region_key}.csv")
    if os.path.exists(weather_file):
        weather_df = pd.read_csv(weather_file)
    else:
        return None
    merged = pd.merge(df_region, weather_df, on="year", how="inner")
    if len(merged) < 5:
        return None
    features = ["temperature_2m", "relative_humidity_2m", "precipitation",
                "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm",
                "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm"]
    X = merged[features].values
    y = merged["population"].values
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    avg_features = merged[features].mean().values
    forecast_years = list(range(last_year + 1, last_year + 11))
    forecast_val = float(np.dot(avg_features, coeffs))
    forecast_list = [forecast_val] * 10
    return {"years": forecast_years, "population": forecast_list}

# =======================
# 7. Flask маршруты
# =======================

@app.route("/")
def index():
    current_year = datetime.now().year
    years = list(range(1935, current_year + 1))
    pests = list(config.keys())
    # Собираем уникальные регионы из конфигураций (если конфигурация содержит ключ 'area')
    regions = sorted(list(set(conf['area'] for conf in config.values() if 'area' in conf)))
    return render_template("index.html", regions=regions, pests=pests, years=years)


@app.route("/data", methods=["POST"])
def get_data():
    try:
        pest = request.json.get("pest")
        region_filter = request.json.get("region")
        year_filter = request.json.get("year")
        if pest == "locust_italian":
            df = load_locust_data()
            if df is None:
                return jsonify({"error": "Данные locust не найдены"}), 404
            df.columns = [col.lower() for col in df.columns]
            if 'region' not in df.columns:
                df = df.rename(columns={df.columns[0]: 'region'})
            df_long = pd.melt(df, id_vars=["region"], var_name="year", value_name="population")
            df_long["year"] = df_long["year"].astype(int)
            df_long["population"] = pd.to_numeric(df_long["population"], errors="coerce")
            df_long = df_long.dropna(subset=["population"])
            current_year = datetime.now().year
            df_long = df_long[(df_long["year"] >= 1935) & (df_long["year"] <= current_year)]
            df_long["region"] = df_long["region"].str.strip().str.lower()
            if region_filter:
                region_filter_lc = region_filter.lower().strip()
                if region_filter_lc in df_long["region"].unique():
                    df_region = df_long[df_long["region"] == region_filter_lc]
                else:
                    df_region = df_long
            else:
                df_region = df_long
            if year_filter:
                df_chart = df_region[df_region["year"] == int(year_filter)]
            else:
                df_chart = df_region.sort_values("year")
            chart_data = {"years": df_chart["year"].tolist(), "population": df_chart["population"].tolist()}
            df_map = df_region.groupby("region", as_index=False)["population"].sum()
            map_data = {"regions": df_map["region"].tolist(), "population": df_map["population"].tolist()}
            return jsonify({"chart": chart_data, "map": map_data})
        else:
            conf = config.get(pest)
            if conf is None:
                return jsonify({"error": "Вредитель не найден"}), 404
            model = AR_ModelLoader(conf['order'], conf['seasonal_order'], conf['s_month'], conf['e_month'], conf['train_size'])
            df = model.load_data(conf['pest'], conf['area'], conf['population_type'], conf['target_var'])
            X = model.extract_features(conf['features'], conf['thresholds'])
            chart_data = {"years": [d.year for d in df.index], "population": df['y'].tolist()}
            map_data = {"regions": [conf['area']], "population": [float(df['y'].sum())]}
            return jsonify({"chart": chart_data, "map": map_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/forecast", methods=["POST"])
def forecast():
    pest = request.json.get("pest")
    region = request.json.get("region")
    if pest == "locust_italian":
        forecast_data = forecast_population_locust(region)
        if forecast_data is None:
            return jsonify({"error": "Недостаточно данных для прогноза"}), 400
        return jsonify(forecast_data)
    else:
        conf = config.get(pest)
        if conf is None:
            return jsonify({"error": "Вредитель не найден"}), 404
        model = AR_ModelLoader(conf['order'], conf['seasonal_order'], conf['s_month'], conf['e_month'], conf['train_size'])
        df = model.load_data(conf['pest'], conf['area'], conf['population_type'], conf['target_var'])
        X = model.extract_features(conf['features'], conf['thresholds'])
        fcast = model.predict(n=1)
        next_year = df.index[-1].year + 1
        forecast_value = fcast.predicted_mean.values[0]
        return jsonify({"years": [next_year], "population": [forecast_value]})

# =======================
# 8. Запуск приложения
# =======================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask Pest Forecasting App")
    parser.add_argument("--port", type=int, default=int(os.getenv("FLASK_PORT", 5090)),
                        help="Порт для запуска сервера")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port, debug=True)
