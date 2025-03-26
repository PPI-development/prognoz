import os
import pandas as pd
from config import data_dir, pogoda_dir

# 📌 Полный маппинг файлов вредителей
pest_file_map = {
    "итальянский прус": "locustitalian.csv",
    "хлебный жук": "breadbeetleZKO.csv",
    "колорадский жук": "coloradobeetleSKOZKO.csv",
    "южноамериканская томатная моль": "R_satomato_ALM.xlsx",
    "горчак ползучий": "repens_ALM.xlsx",
}

# 📌 Полный маппинг файлов погоды
weather_file_map = {
    "акмолинская область": "akmolinskaya.csv",
    "актюбинская область": "aktubinskaya.csv",
    "алматинская область": "almatinskaya.csv",
    "восточно-казахстанская область": "vko.csv",
    "жамбылская область": "zhambyl.csv",
    "западно-казахстанская область": "zko.csv",
    "карагандинская область": "karaganda.csv",
    "костанайская область": "kostanay.csv",
    "кызылординская область": "kyzylorda.csv",
    "мангистауская область": "mangistau.csv",
    "павлодарская область": "pavlodar.csv",
    "северо-казахстанская область": "sko.csv",
    "туркестанская область": "turkestan.csv"
}


### 📌 ФУНКЦИЯ ЗАГРУЗКИ ВРЕДИТЕЛЕЙ
def load_pest_data(pest_name):
    """
    Загружает данные о вредителях, учитывая разные структуры файлов.

    :param pest_name: Название вредителя.
    :return: DataFrame с колонками [region, year, population].
    """
    if pest_name not in pest_file_map:
        print(f"❌ Нет данных для вредителя '{pest_name}'")
        return None

    file_path = os.path.join(data_dir, pest_file_map[pest_name])
    if not os.path.exists(file_path):
        print(f"❌ Файл '{pest_file_map[pest_name]}' не найден")
        return None

    print(f"✅ Загружаем: {file_path}")

    # 📌 Логика обработки каждого файла отдельно
    if pest_name == "итальянский прус":
        df = pd.read_csv(file_path)
        df = df.rename(columns={"Регион": "region"})
        df = df.melt(id_vars=["region"], var_name="year", value_name="population")

    elif pest_name == "хлебный жук":
        df = pd.read_csv(file_path, delimiter=";", encoding="utf-8")
        df = df.rename(columns={"Регион": "region"})
        df = df.melt(id_vars=["region"], var_name="year", value_name="population")

    elif pest_name == "колорадский жук":
        df = pd.read_csv(file_path)
        df = df.rename(columns={"Регион": "region"})
        df = df.drop(columns=["Unnamed: 0"], errors="ignore")  # Убираем лишние колонки
        df = df.melt(id_vars=["region"], var_name="year", value_name="population")

    elif pest_name == "южноамериканская томатная моль":
        df = pd.read_excel(file_path, sheet_name="data")  # XLSX с листом "data"
        df = df.rename(columns={"ds": "year", "Total sum (тыс. га)": "population"})
        df["year"] = pd.to_datetime(df["year"]).dt.year
        df["region"] = "Алматинская область"

    elif pest_name == "горчак ползучий":
        df = pd.read_excel(file_path, sheet_name=0, skiprows=2)  # Пропускаем 2 строки
        df = df.rename(columns={"ds": "year", "Total sum (тыс. га)": "population"})
        df["year"] = pd.to_datetime(df["year"]).dt.year
        df["region"] = "Алматинская область"

    else:
        df = None

    # Проверяем, что нужные колонки есть
    if df is not None and {"region", "year", "population"}.issubset(df.columns):
        return df

    print(f"❌ Ошибка: данные в {file_path} имеют некорректную структуру")
    return None


### 📌 ФУНКЦИЯ ЗАГРУЗКИ ПОГОДЫ
def load_weather_data(region):
    """
    Загружает погодные данные для региона с учетом структуры каждого файла.

    :param region: Название региона (на русском).
    :return: DataFrame с погодными данными.
    """
    if region not in weather_file_map:
        print(f"❌ Ошибка: нет погодных данных для региона '{region}'")
        return None

    file_path = os.path.join(pogoda_dir, weather_file_map[region])

    if not os.path.exists(file_path):
        print(f"❌ Ошибка: файл '{weather_file_map[region]}' не найден в {pogoda_dir}")
        return None

    print(f"✅ Загружен погодный файл: {file_path}")

    # 📌 Логика обработки погодных данных
    if region == "акмолинская область":
        df = pd.read_csv(file_path, delimiter=",", encoding="utf-8")
    elif region == "жамбылская область":
        df = pd.read_csv(file_path, delimiter=";", encoding="windows-1251")
    elif region == "туркестанская область":
        df = pd.read_excel(file_path, sheet_name="Погода")
    else:
        # Универсальная загрузка
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        else:
            print(f"❌ Ошибка: неизвестный формат файла {file_path}")
            return None

    df.columns = [col.lower().strip() for col in df.columns]
    return df


### 📌 ФУНКЦИИ СПИСКОВ
def list_available_pests():
    """
    Возвращает список доступных вредителей.
    """
    return list(pest_file_map.keys())


def list_available_regions():
    """
    Возвращает список регионов, для которых есть погодные данные.
    """
    return list(weather_file_map.keys())
