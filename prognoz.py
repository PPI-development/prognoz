from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
from datetime import datetime
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

app = Flask(__name__)

# Путь к файлам с данными заселения
data_dir = "data"
locust_data_file = os.path.join(data_dir, "locustitalian.csv")
bako_data_file = os.path.join(data_dir, "bakozhogalm.csv")
bread_data_file = os.path.join(data_dir, "breadbeetle.csv")
colorado_data_file = os.path.join(data_dir, "coloradobeetle.csv")
gesen_data_file = os.path.join(data_dir, "gesenmuha.csv")
gold_data_file = os.path.join(data_dir, "goldnematoda.csv")
moroco_data_file = os.path.join(data_dir, "locustmoroco.csv")

# Папка с погодными данными
pogoda_dir = os.path.join(data_dir, "pogoda")

# Список новых названий регионов (ключи – в нижнем регистре)
all_regions = {
    "акмолинская область": [50.0, 69.0],
    "актюбинская область": [48.5, 54.0],
    "алматинская область": [43.0, 76.0],
    "восточно-казахстанская область": [49.5, 82.0],
    "жамбылская область": [42.5, 69.0],
    "западно-казахстанская область": [50.5, 57.0],
    "карагандинская область": [49.8, 66.9],
    "костанайская область": [53.0, 63.6],
    "кызылординская область": [44.0, 65.5],
    "мангистауская область": [43.5, 51.5],
    "павлодарская область": [52.0, 77.0],
    "северо-казахстанская область": [53.2, 69.0],
    "туркестанская область": [42.0, 70.0],
    "нур-султан": [51.2, 71.4],
    "алматы": [43.2, 76.9],
    "уст-каменогорск": [49.9, 86.0],
    "семей": [50.3, 80.2]
}

# Словарь маппинга старых названий (из CSV) на новые
old_to_new = {
    "акмолинская": "акмолинская область",
    "актюбинская": "актюбинская область",
    "алматинская": "алматинская область",
    "атырауская": "туркестанская область",
    "восточно-казахстанская": "восточно-казахстанская область",
    "жамбылская": "жамбылская область",
    "западно-казахстанская": "западно-казахстанская область",
    "карагандинская": "карагандинская область",
    "костанайская": "костанайская область",
    "кызылординская": "кызылординская область",
    "мангистауская": "мангистауская область",
    "павлодарская": "павлодарская область",
    "северо-казахстанская": "северо-казахстанская область",
    "южно-казахстанская": "туркестанская область"
}

# Функция загрузки данных с общей логикой
def load_data(file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df.columns = [col.lower() for col in df.columns]
        if 'region' not in df.columns:
            first = df.columns[0]
            df = df.rename(columns={first: 'region'})
        return df
    return None

def load_locust_data():
    return load_data(locust_data_file)

def load_bako_data():
    return load_data(bako_data_file)

def load_bread_data():
    return load_data(bread_data_file)

def load_colorado_data():
    return load_data(colorado_data_file)

def load_gesen_data():
    return load_data(gesen_data_file)

def load_gold_data():
    return load_data(gold_data_file)

def load_moroco_data():
    return load_data(moroco_data_file)

# Словарь сопоставления вредителей и функций загрузки
pest_loaders = {
    "итальянский прус": load_locust_data,
    "бактериальный ожог": load_bako_data,
    "хлебный жук": load_bread_data,
    "колорадский жук": load_colorado_data,
    "гесенская муха": load_gesen_data,
    "золотистая нематода": load_gold_data,
    "морокская саранча": load_moroco_data,
}

def convert_new_to_old(new_region):
    """
    Преобразует новое название (например, "Алматинская область") в старое,
    удаляя слово " область", если оно присутствует.
    """
    new_region = new_region.lower().strip()
    if new_region.endswith(" область"):
        return new_region.replace(" область", "")
    return new_region

def load_weather_data(region_new):
    """
    Загружает погодные данные для заданного региона (новое название).
    Если файл с таким названием отсутствует, выбирается ближайший по координатам.
    """
    available = [
        "акмолинская область",
        "актюбинская область",
        "алматинская область",
        "карагандинская область",
        "костанайская область",
        "кызылординская область",
        "северо-казахстанская область",
        "южно-казахстанская область"
    ]
    region_new_lc = region_new.lower().strip()
    if region_new_lc in available:
        file_name = region_new_lc + ".csv"
    else:
        best = None
        best_dist = float('inf')
        all_regions_lc = {k.lower(): v for k, v in all_regions.items()}
        if region_new_lc not in all_regions_lc:
            return None
        target_coord = all_regions_lc[region_new_lc]
        for reg in available:
            if reg in all_regions_lc:
                d = np.sqrt((all_regions_lc[reg][0] - target_coord[0]) ** 2 +
                            (all_regions_lc[reg][1] - target_coord[1]) ** 2)
                if d < best_dist:
                    best_dist = d
                    best = reg
        file_name = best + ".csv"
    weather_file = os.path.join(pogoda_dir, file_name)
    if os.path.exists(weather_file):
        weather_df = pd.read_csv(weather_file)
        print("Weather DF shape:", weather_df.shape)
        weather_df['year'] = weather_df['year'].astype(int)
        return weather_df
    return None

def distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)


def forecast_population(region_name, pest):
    loader = pest_loaders.get(pest.lower())
    if loader is None:
        return None
    df = loader()
    if df is None:
        return None
    df.columns = [col.lower() for col in df.columns]
    if 'region' not in df.columns:
        first = df.columns[0]
        df = df.rename(columns={first: 'region'})
    df_long = pd.melt(df, id_vars=['region'], var_name='year', value_name='population')
    df_long['year'] = pd.to_numeric(df_long['year'], errors='coerce')
    df_long['population'] = pd.to_numeric(df_long['population'], errors='coerce')
    df_long = df_long.dropna(subset=['year', 'population'])
    df_long['year'] = df_long['year'].astype(int)
    current_year = datetime.now().year
    df_long = df_long[(df_long['year'] >= 1935) & (df_long['year'] <= current_year)]
    df_long['region'] = df_long['region'].str.strip().str.lower()

    region_key = convert_new_to_old(region_name)
    if region_key in df_long['region'].unique():
        df_region = df_long[df_long['region'] == region_key]
    else:
        missing = region_name.lower().strip()
        best = None
        best_dist = float('inf')
        all_regions_lc = {k.lower(): v for k, v in all_regions.items()}
        if missing not in all_regions_lc:
            return None
        missing_coord = all_regions_lc[missing]
        for reg in df_long['region'].unique():
            if reg in all_regions_lc:
                d = distance(all_regions_lc[reg], missing_coord)
                if d < best_dist:
                    best_dist = d
                    best = reg
        if best is not None:
            df_region = df_long[df_long['region'] == best]
        else:
            return None

    df_region = df_region.sort_values('year')
    last_year = df_region['year'].max()

    weather_df = load_weather_data(region_name)
    if weather_df is None:
        return None

    merged = pd.merge(df_region, weather_df, on="year", how="inner")
    if len(merged) < 2:
        return None

    features = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "soil_temperature_0_to_7cm",
        "soil_temperature_7_to_28cm",
        "soil_moisture_0_to_7cm",
        "soil_moisture_7_to_28cm"
    ]
    for feature in features:
        if feature not in merged.columns:
            return None

    X_all = merged[features].values
    y_all = merged["population"].values

    corrections = []
    for leave in [5, 4, 3, 2, 1]:
        train_data = merged[merged["year"] <= last_year - leave]
        if len(train_data) < 5:
            continue
        X_train = train_data[features].values
        y_train = train_data["population"].values
        model_cv = LinearRegression().fit(X_train, y_train)
        test_data = merged[(merged["year"] > last_year - leave) & (merged["year"] <= last_year)]
        if test_data.empty:
            continue
        X_test = test_data[features].values
        y_test = test_data["population"].values
        y_pred_test = model_cv.predict(X_test)
        error = np.mean(y_test - y_pred_test)
        corrections.append(error)
    avg_correction = np.mean(corrections) if corrections else 0.0

    model_final = LinearRegression().fit(X_all, y_all)
    next_year = last_year + 1
    forecast_weather = weather_df[weather_df['year'] == next_year]
    if forecast_weather.empty:
        forecast_features = weather_df.sort_values('year').tail(3)[features].mean().values
    else:
        forecast_features = forecast_weather.iloc[0][features].values

    predicted_population = model_final.predict([forecast_features])[0] + avg_correction
    if predicted_population < 0:
        predicted_population = 0.0

    return {
        "year": int(next_year),
        "population": float(predicted_population),
        "avg_correction": float(avg_correction)
    }


@app.route("/")
def index():
    current_year = datetime.now().year
    years = list(range(1935, current_year + 1))
    # Обновлённый список вредителей
    pests = list(pest_loaders.keys())
    return render_template(
        "index.html",
        regions=list(all_regions.keys()),
        pests=pests,
        years=years,
    )

@app.route("/data", methods=["POST"])
def get_data():
    try:
        pest = request.json.get("pest", "").lower()
        region_filter = request.json.get("region")
        year_filter = request.json.get("year")

        loader = pest_loaders.get(pest)
        if loader is None:
            return jsonify({"error": "Неверный выбор вредителя"}), 400
        df = loader()
        if df is None:
            return jsonify({"error": "Данные не найдены"}), 404

        df.columns = [col.lower() for col in df.columns]
        if "region" not in df.columns:
            first = df.columns[0]
            df = df.rename(columns={first: "region"})
        df_long = pd.melt(df, id_vars=["region"], var_name="year", value_name="population")
        df_long["year"] = df_long["year"].astype(int)
        df_long["population"] = pd.to_numeric(df_long["population"], errors="coerce")
        df_long = df_long.dropna(subset=["population"])
        current_year = datetime.now().year
        df_long = df_long[(df_long["year"] >= 1935) & (df_long["year"] <= current_year)]
        df_long["region"] = df_long["region"].str.strip().str.lower()

        if region_filter:
            region_filter_lc = region_filter.lower()
            region_filter_old = convert_new_to_old(region_filter)
            if region_filter_old in df_long["region"].unique():
                df_region = df_long[df_long["region"] == region_filter_old]
            else:
                missing = region_filter.lower().strip()
                best = None
                best_dist = float("inf")
                all_regions_lc = {k.lower(): v for k, v in all_regions.items()}
                if missing not in all_regions_lc:
                    return jsonify({"error": "Нет координат для выбранной области"}), 400
                missing_coord = all_regions_lc[missing]
                for reg in df_long["region"].unique():
                    if reg in all_regions_lc:
                        d = distance(all_regions_lc[reg], missing_coord)
                        if d < best_dist:
                            best_dist = d
                            best = reg
                if best is not None:
                    df_region = df_long[df_long["region"] == best]
                else:
                    df_region = df_long[df_long["region"] == region_filter.lower().strip()]
            if year_filter:
                df_chart = df_region[df_region["year"] == int(year_filter)]
                df_map = df_chart.copy()
            else:
                df_chart = df_region.sort_values("year")
                df_map = df_region.groupby("region", as_index=False)["population"].sum()
            df_map["region"] = df_map["region"].apply(
                lambda r: old_to_new[r] if r in old_to_new else r
            ).str.lower()
        else:
            if year_filter:
                df_chart = (
                    df_long[df_long["year"] == int(year_filter)]
                    .groupby("year", as_index=False)["population"]
                    .sum()
                )
                df_map = (
                    df_long[df_long["year"] == int(year_filter)]
                    .groupby("region", as_index=False)["population"]
                    .sum()
                )
            else:
                df_chart = df_long.groupby("year", as_index=False)["population"].sum()
                df_map = df_long.groupby("region", as_index=False)["population"].sum()
            df_map["region"] = df_map["region"].apply(
                lambda r: old_to_new[r] if r in old_to_new else r
            ).str.lower()
        if year_filter:
            df_chart = df_chart[df_chart["year"] == int(year_filter)]
        chart_data = {
            "years": df_chart["year"].tolist() if "year" in df_chart.columns else [],
            "population": df_chart["population"].tolist(),
        }
        map_data = {
            "regions": df_map["region"].tolist(),
            "population": df_map["population"].tolist(),
        }
        return jsonify({"chart": chart_data, "map": map_data})
    except Exception as e:
        print("Ошибка в /data:", e)
        return jsonify({"error": "Ошибка на сервере: " + str(e)}), 500

@app.route("/forecast", methods=["POST"])
def predict():
    region_name = request.json.get("region")
    pest = request.json.get("pest", "").lower()
    forecast = forecast_population(region_name, pest)
    if not forecast:
        return jsonify({"error": "Недостаточно данных для прогноза"}), 400
    return jsonify(forecast)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask Car Tracker")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("FLASK_PORT", 5090)),
        help="Порт для запуска сервера",
    )
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port, debug=True)
