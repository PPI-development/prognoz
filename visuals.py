import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from config import pest_types, all_regions


def plot_population_trend(df, pest_name, region_name):
    """
    Создает интерактивный график численности вредителей по годам.

    :param df: DataFrame с данными (колонки: "year", "population").
    :param pest_name: Название вредителя.
    :param region_name: Название региона.
    :return: HTML-код с графиком.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["year"],
        y=df["population"],
        mode="lines+markers",
        marker=dict(color="blue"),
        name="Численность"
    ))

    fig.update_layout(
        title=f"Численность {pest_types.get(pest_name, pest_name)} в {region_name}",
        xaxis_title="Год",
        yaxis_title="Численность (тыс. га)",
        height=500,
        width=800
    )

    return fig.to_html(full_html=False)


def plot_population_map(df, year):
    """
    Создает интерактивную карту численности вредителей по регионам в заданном году.

    :param df: DataFrame с данными (колонки: "region", "population").
    :param year: Год для отображения данных.
    :return: HTML-код с картой.
    """
    map_fig = go.Figure(data=go.Choropleth(
        locations=df["region"],
        z=df["population"],
        locationmode="country names",
        colorscale="Reds",
        colorbar_title="Численность (тыс. га)"
    ))

    map_fig.update_layout(
        title=f"Численность вредителей в Казахстане ({year} г.)",
        geo_scope="asia",
        height=500,
        width=800
    )

    return map_fig.to_html(full_html=False)
