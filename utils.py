import numpy as np
import pandas as pd


def convert_new_to_old(region_name):
    """
    Преобразует новое название (например, "Алматинская область") в старое,
    удаляя слово " область", если оно присутствует.

    :param region_name: Название региона (например, "Алматинская область").
    :return: Сокращенное название (например, "Алматинская").
    """
    region_name = region_name.lower().strip()
    if region_name.endswith(" область"):
        return region_name.replace(" область", "")
    return region_name


def distance(coord1, coord2):
    """
    Вычисляет евклидово расстояние между двумя точками (широта, долгота).

    :param coord1: Координаты первой точки [широта, долгота].
    :param coord2: Координаты второй точки [широта, долгота].
    :return: Евклидово расстояние.
    """
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)


def find_closest_region(target_region, all_regions):
    """
    Находит ближайший регион, если данных по целевому региону нет.

    :param target_region: Регион, для которого нет данных.
    :param all_regions: Словарь всех регионов и их координат.
    :return: Название ближайшего региона.
    """
    target_region = target_region.lower().strip()
    if target_region not in all_regions:
        return None

    target_coord = all_regions[target_region]
    best_match = None
    best_distance = float("inf")

    for region, coord in all_regions.items():
        if region != target_region:
            d = distance(target_coord, coord)
            if d < best_distance:
                best_distance = d
                best_match = region

    return best_match


def clean_dataframe(df):
    """
    Очищает DataFrame, убирая дубликаты, пропущенные значения и форматируя колонки.

    :param df: DataFrame с данными.
    :return: Очищенный DataFrame.
    """
    df = df.drop_duplicates().dropna()
    df.columns = [col.lower().strip() for col in df.columns]
    return df
