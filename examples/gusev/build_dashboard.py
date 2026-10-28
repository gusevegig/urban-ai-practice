#!/usr/bin/env python
r"""Собирает dashboard.html из готовых результатов исследования Гусева.

Скрипт НЕ скачивает данные и НЕ пересчитывает показатели. Он только читает
сохранённые GeoJSON/CSV из папки data и складывает их в одну HTML-страницу:
интерактивную карту, графики, таблицы и выводы.

Запуск (из корня репозитория):
    .\.venv\Scripts\python.exe .\examples\gusev\build_dashboard.py

По умолчанию результат: examples/gusev/dashboard.html
"""

from __future__ import annotations

import argparse
import html
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import mapping


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_OUTPUT = SCRIPT_DIR / "dashboard.html"
ASSETS_DIR = SCRIPT_DIR / "assets"
POPULATION_REFERENCE = 28_500

PALETTE_BLUE = ["#eff6ff", "#bfdbfe", "#60a5fa", "#2563eb", "#1e3a8a"]
PALETTE_ORANGE = ["#fff7ed", "#fed7aa", "#fb923c", "#ea580c", "#7c2d12"]
PALETTE_GREEN = ["#f0fdf4", "#bbf7d0", "#4ade80", "#16a34a", "#14532d"]
PALETTE_TEAL = ["#f0fdfa", "#99f6e4", "#2dd4bf", "#0d9488", "#134e4a"]
PALETTE_PURPLE = ["#faf5ff", "#e9d5ff", "#c084fc", "#9333ea", "#581c87"]
PALETTE_YELLOW = ["#fefce8", "#fef08a", "#facc15", "#ca8a04", "#713f12"]
PALETTE_RED_GREEN = ["#dc2626", "#f59e0b", "#facc15", "#84cc16", "#16a34a"]
PALETTE_GREEN_RED = ["#16a34a", "#84cc16", "#facc15", "#f97316", "#dc2626"]
PALETTE_SLATE = ["#f8fafc", "#cbd5e1", "#94a3b8", "#475569", "#0f172a"]
CATEGORICAL_PALETTE = [
    "#2563eb", "#f97316", "#10b981", "#8b5cf6", "#ef4444", "#06b6d4",
    "#eab308", "#ec4899", "#84cc16", "#14b8a6", "#64748b", "#a855f7",
]

LAND_USE_RU = {
    "residential": "Жилая",
    "business": "Деловая",
    "recreation": "Рекреационная",
    "industrial": "Промышленная",
    "transport": "Транспортная",
    "special": "Специальная",
    "agriculture": "Сельскохозяйственная",
}

MORPH_RU = {
    "low-rise low-density residential": "Малоэтажная низкоплотная жилая",
    "low-rise low-density non-residential": "Малоэтажная низкоплотная нежилая",
    "low-rise low-density mixed-use": "Малоэтажная низкоплотная смешанная",
    "mid-rise low-density residential": "Среднеэтажная низкоплотная жилая",
    "mid-rise residential": "Среднеэтажная жилая",
    "mid-rise low-density non-residential": "Среднеэтажная низкоплотная нежилая",
    "low-rise model": "Малоэтажная модель",
    "individual residential": "Индивидуальная жилая застройка",
    "low-rise non-residential": "Малоэтажная нежилая",
    "mid-rise": "Среднеэтажная",
    "mid-rise non-residential": "Среднеэтажная нежилая",
}

CATEGORY_COLUMNS = [
    "land_use",
    "spacematrix_cluster",
    "morphotype_spacematrix",
    "morphotype_strelka",
]

SUMMARY_COLUMNS = [
    ("population", "Население, чел.", 0),
    ("fsi", "FSI, м²/м²", 3),
    ("gsi", "GSI, доля", 3),
    ("mxi", "MXI, доля", 3),
    ("l", "L, этажи", 2),
    ("osr", "OSR, м²/м²", 2),
    ("share_living", "Жилая площадь на м² застройки", 3),
    ("share_non_living", "Нежилая площадь на м² застройки", 3),
    ("demand", "Спрос на школы, мест", 0),
    ("capacity", "Ёмкость школ, мест", 0),
    ("provision_strong", "Обеспеченность strong", 3),
    ("provision_weak", "Обеспеченность weak", 3),
    ("area_accessibility", "Доступность территории, мин", 2),
    ("mean_accessibility", "Средняя доступность, мин", 2),
]

TABLE_COLUMNS = [
    {"key": "block_id", "label": "ID квартала", "type": "number", "digits": 0},
    {"key": "population", "label": "Население, чел.", "type": "number", "digits": 0},
    {"key": "land_use", "label": "Land use", "type": "text"},
    {"key": "share", "label": "Доля land use", "type": "number", "digits": 3},
    {"key": "residential", "label": "Жилая, доля", "type": "number", "digits": 3},
    {"key": "business", "label": "Деловая, доля", "type": "number", "digits": 3},
    {"key": "recreation", "label": "Рекреация, доля", "type": "number", "digits": 3},
    {"key": "industrial", "label": "Промышленная, доля", "type": "number", "digits": 3},
    {"key": "transport", "label": "Транспорт, доля", "type": "number", "digits": 3},
    {"key": "special", "label": "Специальная, доля", "type": "number", "digits": 3},
    {"key": "agriculture", "label": "Сельхоз, доля", "type": "number", "digits": 3},
    {"key": "fsi", "label": "FSI, м²/м²", "type": "number", "digits": 3},
    {"key": "gsi", "label": "GSI, доля", "type": "number", "digits": 3},
    {"key": "mxi", "label": "MXI, доля", "type": "number", "digits": 3},
    {"key": "l", "label": "L, этажи", "type": "number", "digits": 2},
    {"key": "osr", "label": "OSR, м²/м²", "type": "number", "digits": 2},
    {"key": "share_living", "label": "Жилая площадь / застройка", "type": "number", "digits": 3},
    {"key": "share_non_living", "label": "Нежилая площадь / застройка", "type": "number", "digits": 3},
    {"key": "spacematrix_cluster", "label": "Кластер Spacematrix", "type": "text"},
    {"key": "morphotype_spacematrix", "label": "Морфотип Spacematrix", "type": "text"},
    {"key": "morphotype_strelka", "label": "Морфотип Strelka", "type": "text"},
    {"key": "demand", "label": "Спрос на школы, мест", "type": "number", "digits": 0},
    {"key": "capacity", "label": "Ёмкость школ, мест", "type": "number", "digits": 0},
    {"key": "provision_strong", "label": "Обеспеченность strong", "type": "number", "digits": 3},
    {"key": "provision_weak", "label": "Обеспеченность weak", "type": "number", "digits": 3},
    {"key": "area_accessibility", "label": "Доступность территории, мин", "type": "number", "digits": 2},
    {"key": "mean_accessibility", "label": "Средняя доступность, мин", "type": "number", "digits": 2},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Собирает dashboard.html из готовых GeoJSON/CSV исследования Гусева."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Папка с готовыми файлами data (по умолчанию: рядом со скриптом).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Путь к итоговому HTML (по умолчанию: examples/gusev/dashboard.html).",
    )
    return parser.parse_args()


def clean(value: Any) -> Any:
    """Преобразует значение в безопасный для JSON тип."""
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return str(value)


def pretty_category(column: str, value: Any) -> str:
    """Переводит категории OSM/BlocksNet в подписи для dashboard."""
    if value is None:
        return "Не определён"
    try:
        if pd.isna(value):
            return "Не определён"
    except (TypeError, ValueError):
        pass
    text = str(value)
    if column == "land_use":
        return LAND_USE_RU.get(text, text)
    if column in ("morphotype_spacematrix", "morphotype_strelka"):
        return MORPH_RU.get(text, text)
    if column == "spacematrix_cluster":
        try:
            return f"Кластер {int(round(float(value)))}"
        except (TypeError, ValueError):
            return text
    return text


def fmt_num(value: Any, digits: int = 0) -> str:
    """Русское форматирование числа: пробел между разрядами, запятая в дробях."""
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    text = f"{number:,.{digits}f}"
    return text.replace(",", " ").replace(".", ",")


def plural_ru(number: Any, one: str, few: str, many: str) -> str:
    """Русское согласование существительного с числом."""
    try:
        value = abs(int(number))
    except (TypeError, ValueError):
        return many
    last_two = value % 100
    if 11 <= last_two <= 14:
        return many
    last = value % 10
    if last == 1:
        return one
    if 2 <= last <= 4:
        return few
    return many


def round_coords(value: Any, digits: int = 6) -> Any:
    if isinstance(value, (list, tuple)):
        return [round_coords(item, digits) for item in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), digits)
    return value


def geojson_features(
    gdf: gpd.GeoDataFrame,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    """GeoJSON FeatureCollection из GeoDataFrame (геометрия уже в WGS84)."""
    features: list[dict[str, Any]] = []
    if gdf is None or gdf.empty:
        return {"type": "FeatureCollection", "features": features}

    property_columns = columns if columns is not None else [
        column for column in gdf.columns if column != gdf.geometry.name
    ]
    for _, row in gdf.iterrows():
        geometry = row.geometry
        if geometry is None or geometry.is_empty:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {column: clean(row[column]) for column in property_columns},
                "geometry": round_coords(mapping(geometry)),
            }
        )
    return {"type": "FeatureCollection", "features": features}


def geojson_from_rows(
    gdf: gpd.GeoDataFrame,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """GeoJSON, где свойства уже подготовлены (с русскими категориями)."""
    features: list[dict[str, Any]] = []
    for (_, row), properties in zip(gdf.iterrows(), rows):
        geometry = row.geometry
        if geometry is None or geometry.is_empty:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": properties,
                "geometry": round_coords(mapping(geometry)),
            }
        )
    return {"type": "FeatureCollection", "features": features}


def load_layer(
    data_dir: Path,
    filename: str,
    columns: list[str] | None = None,
    simplify: float | None = None,
    filter_fn: Callable[[gpd.GeoDataFrame], gpd.GeoDataFrame] | None = None,
) -> dict[str, Any] | None:
    path = data_dir / filename
    if not path.exists():
        return None
    gdf = gpd.read_file(path)
    if gdf.empty:
        return None
    if gdf.crs is not None:
        gdf = gdf.to_crs(4326)
    if filter_fn is not None:
        gdf = filter_fn(gdf)
    if simplify is not None:
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.simplify(simplify, preserve_topology=True)
    return geojson_features(gdf, columns=columns)


def prepare_rows(blocks: gpd.GeoDataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in blocks.iterrows():
        properties: dict[str, Any] = {}
        for column in blocks.columns:
            if column == blocks.geometry.name:
                continue
            value = row[column]
            properties[column] = (
                pretty_category(column, value) if column in CATEGORY_COLUMNS else clean(value)
            )
        rows.append(properties)
    return rows


def numeric_meta(
    blocks: gpd.GeoDataFrame,
    column: str,
    label: str,
    palette: list[str],
    digits: int,
    fixed: tuple[float, float] | None = None,
    log_scale: bool = False,
) -> dict[str, Any]:
    series = (
        pd.to_numeric(blocks[column], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if fixed is not None:
        minimum, maximum = fixed
    elif len(series) == 0:
        minimum, maximum = 0.0, 1.0
    else:
        minimum, maximum = float(series.min()), float(series.max())
    if maximum <= minimum:
        maximum = minimum + 1.0
    return {
        "label": label,
        "type": "numeric",
        "min": minimum,
        "max": maximum,
        "digits": digits,
        "palette": palette,
        "log": log_scale,
    }


def build_indicators(
    blocks: gpd.GeoDataFrame,
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    population_max = max(1.0, float(pd.to_numeric(blocks["population"], errors="coerce").max()))
    demand_max = max(1.0, float(pd.to_numeric(blocks["demand"], errors="coerce").max()))
    capacity_max = max(1.0, float(pd.to_numeric(blocks["capacity"], errors="coerce").max()))
    indicators: dict[str, Any] = {
        "population": numeric_meta(blocks, "population", "Население, чел.", PALETTE_ORANGE, 0, fixed=(0.0, population_max)),
        "fsi": numeric_meta(blocks, "fsi", "FSI, м²/м²", PALETTE_BLUE, 3, fixed=(0.0, 1.2)),
        "gsi": numeric_meta(blocks, "gsi", "GSI, доля", PALETTE_TEAL, 3, fixed=(0.0, 0.45)),
        "mxi": numeric_meta(blocks, "mxi", "MXI, доля", PALETTE_YELLOW, 3, fixed=(0.0, 0.9)),
        "l": numeric_meta(blocks, "l", "L, этажи", PALETTE_GREEN, 2, fixed=(1.0, 5.0)),
        "osr": numeric_meta(blocks, "osr", "OSR, м²/м²", PALETTE_BLUE, 2, log_scale=True),
        "share": numeric_meta(blocks, "share", "Доля land use", PALETTE_SLATE, 3, fixed=(0.0, 1.0)),
        "share_living": numeric_meta(blocks, "share_living", "Жилая площадь / застройка", PALETTE_PURPLE, 3),
        "share_non_living": numeric_meta(blocks, "share_non_living", "Нежилая площадь / застройка", PALETTE_SLATE, 3),
        "demand": numeric_meta(blocks, "demand", "Спрос на школы, мест", PALETTE_SLATE, 0, fixed=(0.0, demand_max)),
        "capacity": numeric_meta(blocks, "capacity", "Ёмкость школ, мест", PALETTE_TEAL, 0, fixed=(0.0, capacity_max)),
        "provision_strong": numeric_meta(blocks, "provision_strong", "Обеспеченность strong", PALETTE_RED_GREEN, 3, fixed=(0.0, 1.0)),
        "provision_weak": numeric_meta(blocks, "provision_weak", "Обеспеченность weak", PALETTE_RED_GREEN, 3, fixed=(0.0, 1.0)),
        "area_accessibility": numeric_meta(blocks, "area_accessibility", "Доступность территории, мин", PALETTE_GREEN_RED, 2),
        "mean_accessibility": numeric_meta(blocks, "mean_accessibility", "Средняя доступность, мин", PALETTE_GREEN_RED, 2),
    }

    category_counts: dict[str, list[dict[str, Any]]] = {}
    for column, label in [
        ("land_use", "Land use"),
        ("spacematrix_cluster", "Кластер Spacematrix"),
        ("morphotype_spacematrix", "Морфотип Spacematrix"),
        ("morphotype_strelka", "Морфотип Strelka"),
    ]:
        counts = Counter(row[column] for row in rows)
        items = sorted(
            counts.items(),
            key=lambda item: (item[0] == "Не определён", -item[1], item[0]),
        )
        category_counts[column] = [
            {"label": category, "count": int(count)} for category, count in items
        ]
        indicators[column] = {
            "label": label,
            "type": "categorical",
            "palette": CATEGORICAL_PALETTE,
        }

    return indicators, category_counts


def histogram(
    values: list[float | None],
    bins: int,
    minimum: float | None = None,
    maximum: float | None = None,
) -> dict[str, list[Any]]:
    numbers = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    if not numbers:
        return {"labels": [], "values": []}
    if minimum is None:
        minimum = float(min(numbers))
    if maximum is None:
        maximum = float(max(numbers))
    if maximum <= minimum:
        maximum = minimum + 1.0
    counts, edges = np.histogram(numbers, bins=bins, range=(minimum, maximum))
    labels = [
        f"{edges[index]:.2f}–{edges[index + 1]:.2f}"
        for index in range(len(counts))
    ]
    return {"labels": labels, "values": [int(value) for value in counts]}


def build_charts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    built = [row for row in rows if (row.get("fsi") or 0) > 0]
    populated = [row for row in rows if (row.get("population") or 0) > 0]

    def counts_chart(column: str) -> dict[str, list[Any]]:
        counts = Counter(row[column] for row in rows)
        items = sorted(
            counts.items(),
            key=lambda item: (item[0] == "Не определён", -item[1], item[0]),
        )
        return {
            "labels": [item[0] for item in items],
            "values": [int(item[1]) for item in items],
        }

    population_top = sorted(
        [row for row in rows if (row.get("population") or 0) > 0],
        key=lambda row: row.get("population") or 0,
        reverse=True,
    )[:12]

    demand_top = sorted(
        [row for row in rows if (row.get("demand") or 0) > 0],
        key=lambda row: row.get("demand") or 0,
        reverse=True,
    )[:10]

    max_population = max(((row.get("population") or 0) for row in built), default=0) or 1
    scatter = []
    for row in built:
        fsi = row.get("fsi")
        floors = row.get("l")
        if fsi is None or floors is None:
            continue
        population = row.get("population") or 0
        radius = 3.0 + 11.0 * math.sqrt(max(population, 0) / max_population)
        scatter.append(
            {
                "x": float(fsi),
                "y": float(floors),
                "r": round(radius, 2),
                "block_id": row.get("block_id"),
                "population": population,
            }
        )

    l_counts = Counter(
        int(round(float(row["l"])))
        for row in built
        if row.get("l") is not None
    )
    l_labels = [str(value) for value in range(1, max(l_counts, default=1) + 1)]

    provision_labels = ["0", "0–0,5", "0,5–0,75", "0,75–1"]
    provision_values = [0, 0, 0, 0]
    for row in populated:
        value = row.get("provision_strong")
        if value is None:
            continue
        if value <= 0:
            provision_values[0] += 1
        elif value < 0.5:
            provision_values[1] += 1
        elif value < 0.75:
            provision_values[2] += 1
        else:
            provision_values[3] += 1

    return {
        "landUse": counts_chart("land_use"),
        "populationTop": {
            "labels": [f"Квартал {int(row['block_id'])}" for row in population_top],
            "values": [int(row.get("population") or 0) for row in population_top],
        },
        "spacematrix": counts_chart("morphotype_spacematrix"),
        "strelka": counts_chart("morphotype_strelka"),
        "fsiHist": histogram([row.get("fsi") for row in built], bins=12, minimum=0.0, maximum=1.2),
        "gsiHist": histogram([row.get("gsi") for row in built], bins=9, minimum=0.0, maximum=0.45),
        "mxiHist": histogram([row.get("mxi") for row in built], bins=9, minimum=0.0, maximum=0.9),
        "lCounts": {
            "labels": l_labels,
            "values": [int(l_counts.get(int(label), 0)) for label in l_labels],
        },
        "fsiLScatter": scatter,
        "provision": {"labels": provision_labels, "values": provision_values},
        "accessHist": histogram(
            [row.get("area_accessibility") for row in rows],
            bins=10,
            minimum=30.0,
            maximum=75.0,
        ),
        "demandCapacity": {
            "labels": [f"Квартал {int(row['block_id'])}" for row in demand_top],
            "demand": [int(row.get("demand") or 0) for row in demand_top],
            "capacity": [int(row.get("capacity") or 0) for row in demand_top],
        },
    }


def build_summary(blocks: gpd.GeoDataFrame) -> list[dict[str, Any]]:
    summary = []
    for column, label, digits in SUMMARY_COLUMNS:
        series = (
            pd.to_numeric(blocks[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        values = series.dropna()
        summary.append(
            {
                "key": column,
                "label": label,
                "digits": digits,
                "count": int(values.size),
                "mean": clean(values.mean()) if len(values) else None,
                "median": clean(values.median()) if len(values) else None,
                "min": clean(values.min()) if len(values) else None,
                "max": clean(values.max()) if len(values) else None,
                "missing": int(series.isna().sum()),
            }
        )
    return summary


def build_stats(
    blocks: gpd.GeoDataFrame,
    schools_gdf: gpd.GeoDataFrame | None,
    data_path: Path,
) -> dict[str, Any]:
    population = float(pd.to_numeric(blocks["population"], errors="coerce").fillna(0).sum())
    fsi = pd.to_numeric(blocks["fsi"], errors="coerce").fillna(0)
    built = blocks[fsi > 0].copy()
    populated_mask = pd.to_numeric(blocks["population"], errors="coerce").fillna(0) > 0
    populated = blocks[populated_mask].copy()

    if schools_gdf is not None and "capacity" in schools_gdf.columns:
        school_capacity = float(pd.to_numeric(schools_gdf["capacity"], errors="coerce").fillna(0).sum())
        school_count = int(len(schools_gdf))
    else:
        school_capacity = float(pd.to_numeric(blocks["capacity"], errors="coerce").fillna(0).sum())
        school_count = 0

    demand_total = float(pd.to_numeric(blocks["demand"], errors="coerce").fillna(0).sum())
    capacity_total = float(pd.to_numeric(blocks["capacity"], errors="coerce").fillna(0).sum())
    provision = pd.to_numeric(blocks["provision_strong"], errors="coerce")
    provision_populated = pd.to_numeric(populated["provision_strong"], errors="coerce")
    area_access = pd.to_numeric(blocks["area_accessibility"], errors="coerce").dropna()
    mean_access = pd.to_numeric(blocks["mean_accessibility"], errors="coerce").dropna()

    land_counts = Counter(
        pretty_category("land_use", value) for value in blocks["land_use"]
    )
    morph_counts = Counter(
        pretty_category("morphotype_spacematrix", value)
        for value in blocks["morphotype_spacematrix"]
    )

    def number(series: pd.Series, method: str) -> float | None:
        values = pd.to_numeric(series, errors="coerce").dropna()
        if len(values) == 0:
            return None
        return clean(getattr(values, method)())

    top_land_use = [
        {"label": label, "count": int(count)}
        for label, count in land_counts.most_common()
    ]
    top_land_use_defined = [
        item for item in top_land_use if item["label"] != "Не определён"
    ]
    top_morph = [
        {"label": label, "count": int(count)}
        for label, count in morph_counts.most_common()
        if label != "Не определён"
    ]

    return {
        "population": population,
        "population_reference": POPULATION_REFERENCE,
        "population_diff_pct": (population - POPULATION_REFERENCE) / POPULATION_REFERENCE * 100,
        "area_km2": float(blocks.geometry.area.sum() / 1_000_000),
        "blocks": int(len(blocks)),
        "developed": int(len(built)),
        "no_buildings": int(len(blocks) - len(built)),
        "populated": int(populated_mask.sum()),
        "school_count": school_count,
        "school_capacity": school_capacity,
        "demand_total": demand_total,
        "capacity_total": capacity_total,
        "coverage_pct": (capacity_total / demand_total * 100) if demand_total else 0.0,
        "provision_mean": number(provision, "mean"),
        "provision_median": number(provision, "median"),
        "zero_provision": int((provision_populated.fillna(0) <= 0).sum()),
        "built_fsi_median": number(built["fsi"], "median"),
        "built_fsi_mean": number(built["fsi"], "mean"),
        "built_gsi_median": number(built["gsi"], "median"),
        "built_gsi_mean": number(built["gsi"], "mean"),
        "built_mxi_median": number(built["mxi"], "median"),
        "built_mxi_mean": number(built["mxi"], "mean"),
        "built_l_median": number(built["l"], "median"),
        "built_l_mean": number(built["l"], "mean"),
        "built_l_max": number(built["l"], "max"),
        "area_access_mean": number(area_access, "mean"),
        "area_access_median": number(area_access, "median"),
        "area_access_min": number(area_access, "min"),
        "area_access_max": number(area_access, "max"),
        "mean_access_mean": number(mean_access, "mean"),
        "top_land_use": top_land_use,
        "top_land_use_defined": top_land_use_defined,
        "top_morph": top_morph,
        "undefined_land_use": int(blocks["land_use"].isna().sum()),
        "undefined_morph": int(blocks["morphotype_spacematrix"].isna().sum()),
        "data_date": datetime.fromtimestamp(data_path.stat().st_mtime).strftime("%d.%m.%Y %H:%M"),
        "generated_at": datetime.now().strftime("%d.%m.%Y %H:%M"),
    }


def build_kpi(stats: dict[str, Any]) -> str:
    diff = stats["population_diff_pct"]
    diff_text = f"{diff:+.0f}%"
    cards = [
        (
            "Население (оценка)",
            fmt_num(stats["population"], 0),
            f"ориентир 28 500 · {diff_text}",
            "#f97316",
        ),
        (
            "Площадь территории",
            f"{fmt_num(stats['area_km2'], 1)} км²",
            f"{stats['blocks']} {plural_ru(stats['blocks'], 'квартал', 'квартала', 'кварталов')}",
            "#2563eb",
        ),
        (
            "Застроено кварталов",
            fmt_num(stats["developed"], 0),
            f"{stats['no_buildings']} без зданий",
            "#10b981",
        ),
        (
            "Ёмкость школ",
            fmt_num(stats["school_capacity"], 0),
            f"спрос {fmt_num(stats['demand_total'], 0)} · покрытие {fmt_num(stats['coverage_pct'], 0)}%",
            "#8b5cf6",
        ),
        (
            "Обеспеченность",
            fmt_num(stats["provision_mean"], 2),
            f"медиана {fmt_num(stats['provision_median'], 2)} · 0 в {stats['zero_provision']} из {stats['populated']}",
            "#ef4444",
        ),
        (
            "Доступность",
            f"{fmt_num(stats['area_access_mean'], 1)} мин",
            f"диапазон {fmt_num(stats['area_access_min'], 1)}–{fmt_num(stats['area_access_max'], 1)} мин",
            "#06b6d4",
        ),
    ]
    return "\n".join(
        f'<div class="kpi" style="--accent:{accent}">'
        f'<div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="kpi-value">{html.escape(value)}</div>'
        f'<div class="kpi-note">{html.escape(note)}</div>'
        f"</div>"
        for label, value, note, accent in cards
    )


def build_conclusions(stats: dict[str, Any]) -> tuple[str, str, str]:
    diff = stats["population_diff_pct"]
    diff_word = "выше" if diff >= 0 else "ниже"
    top_land = (
        stats["top_land_use_defined"][0]
        if stats["top_land_use_defined"]
        else {"label": "—", "count": 0}
    )
    top_morph = stats["top_morph"][0] if stats["top_morph"] else {"label": "—", "count": 0}
    land_summary = " · ".join(
        f"{item['label']} — {item['count']}" for item in stats["top_land_use_defined"][:6]
    )
    blocks_word = plural_ru(stats["blocks"], "квартал", "квартала", "кварталов")
    developed_word = plural_ru(stats["developed"], "квартал", "квартала", "кварталов")
    demand_word = plural_ru(stats["demand_total"], "место", "места", "мест")
    capacity_word = plural_ru(stats["capacity_total"], "место", "места", "мест")
    top_morph_word = plural_ru(top_morph["count"], "квартал", "квартала", "кварталов")
    top_land_word = plural_ru(top_land["count"], "квартал", "квартала", "кварталов")
    undefined_morph_word = plural_ru(
        stats["undefined_morph"], "квартала", "кварталов", "кварталов"
    )
    no_buildings_genitive_word = plural_ru(
        stats["no_buildings"], "квартала", "кварталов", "кварталов"
    )

    short = f"""
    <ul class="insight-list">
      <li><b>Население:</b> {fmt_num(stats['population'], 0)} чел. — на {fmt_num(abs(diff), 0)}% {diff_word} ориентира 28 500 чел.</li>
      <li><b>Застройка:</b> {stats['developed']} из {stats['blocks']} кварталов застроены; {stats['no_buildings']} — без зданий.</li>
      <li><b>Этажность:</b> медиана {fmt_num(stats['built_l_median'], 1)} эт., максимум {fmt_num(stats['built_l_max'], 0)} эт.</li>
      <li><b>Школы:</b> ёмкость {fmt_num(stats['capacity_total'], 0)} {capacity_word}; нормативный спрос — {fmt_num(stats['demand_total'], 0)} {demand_word}; средняя обеспеченность {fmt_num(stats['provision_mean'], 2)}.</li>
      <li><b>Доступность:</b> {fmt_num(stats['area_access_mean'], 1)} мин в среднем (взвешено по площади).</li>
    </ul>
    """

    full = f"""
    <div class="conclusion-grid">
      <div class="conclusion-card">
        <h3>Население и территория</h3>
        <ul>
          <li>Оценка населения — <b>{fmt_num(stats['population'], 0)} чел.</b> при ориентире 28 500; разница <b>{diff:+.0f}%</b>.</li>
          <li>Площадь территории — <b>{fmt_num(stats['area_km2'], 1)} км²</b>; <b>{stats['blocks']}</b> {blocks_word}.</li>
          <li>Застроено <b>{stats['developed']}</b> {developed_word}; <b>{stats['no_buildings']}</b> без зданий (лес, вода, пустыри, с/х).</li>
          <li>Население есть в <b>{stats['populated']}</b> кварталах; в остальных — 0.</li>
        </ul>
      </div>
      <div class="conclusion-card">
        <h3>Плотность и морфотипы</h3>
        <ul>
          <li>Среди застроенных кварталов медианный FSI — <b>{fmt_num(stats['built_fsi_median'], 3)}</b>, GSI — <b>{fmt_num(stats['built_gsi_median'], 3)}</b>.</li>
          <li>Медианная этажность — <b>{fmt_num(stats['built_l_median'], 1)}</b>; средняя — <b>{fmt_num(stats['built_l_mean'], 2)}</b>; максимум — <b>{fmt_num(stats['built_l_max'], 0)}</b>.</li>
          <li>MXI (доля жилья) — медиана <b>{fmt_num(stats['built_mxi_median'], 2)}</b>: застройка преимущественно жилая.</li>
          <li>Самый частый морфотип Spacematrix — <b>{html.escape(top_morph['label'])}</b> ({top_morph['count']} {top_morph_word}).</li>
        </ul>
      </div>
      <div class="conclusion-card">
        <h3>Школы и доступность</h3>
        <ul>
          <li>Нормативный спрос — <b>{fmt_num(stats['demand_total'], 0)}</b> {demand_word}; ёмкость школ — <b>{fmt_num(stats['capacity_total'], 0)}</b> {capacity_word} (покрытие <b>{fmt_num(stats['coverage_pct'], 0)}%</b>).</li>
          <li>Средняя обеспеченность strong — <b>{fmt_num(stats['provision_mean'], 2)}</b>, медиана — <b>{fmt_num(stats['provision_median'], 2)}</b>.</li>
          <li>В <b>{stats['zero_provision']}</b> из <b>{stats['populated']}</b> кварталов с населением обеспеченность = 0.</li>
          <li>Средняя доступность территории — <b>{fmt_num(stats['area_access_mean'], 1)} мин</b>; диапазон <b>{fmt_num(stats['area_access_min'], 1)}–{fmt_num(stats['area_access_max'], 1)}</b> мин.</li>
        </ul>
      </div>
      <div class="conclusion-card">
        <h3>Land use и данные</h3>
        <ul>
          <li>Преобладающий land use: <b>{html.escape(top_land['label'])}</b> ({top_land['count']} {top_land_word}).</li>
          <li>{html.escape(land_summary)}; не определён — {stats['undefined_land_use']}.</li>
          <li>Морфотип не определён у {stats['undefined_morph']} {undefined_morph_word} — это незастроенные территории.</li>
          <li>Ёмкость школ восстановлена по нормативам BlocksNet, а не взята из фактических данных.</li>
        </ul>
      </div>
    </div>
    """

    limitations = f"""
    <ul class="clean-list">
      <li>Данные OpenStreetMap могут быть неполными; пропуски этажности и ёмкости школ влияют на оценки.</li>
      <li>Население — расчётная оценка: жилая площадь / 30 м² на человека, а не официальная статистика.</li>
      <li>Ёмкость школ восстановлена по нормативам; фактическая заполняемость не учитывается.</li>
      <li>У {stats['no_buildings']} {no_buildings_genitive_word} без зданий плотностные и морфотипные показатели не определены.</li>
      <li>Обеспеченность рассчитана только для кварталов с населением; в остальных — «—».</li>
      <li>Dashboard не пересчитывает показатели, а только показывает сохранённые значения из <code>data/</code>.</li>
    </ul>
    """

    return short, full, limitations


def build_data(data_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    blocks_path = data_dir / "blocks_indicators.geojson"
    if not blocks_path.exists():
        raise FileNotFoundError(
            f"Не найден обязательный файл: {blocks_path}\n"
            "Сначала выполните ноутбук gusev.ipynb до раздела 11."
        )

    blocks = gpd.read_file(blocks_path)
    if blocks.empty:
        raise ValueError("Файл blocks_indicators.geojson пуст.")
    if blocks.crs is None:
        raise ValueError("У blocks_indicators.geojson не указана система координат (CRS).")
    if "block_id" not in blocks.columns:
        blocks = blocks.copy()
        blocks["block_id"] = blocks.index

    blocks = blocks.copy()
    blocks_wgs = blocks.to_crs(4326)
    rows = prepare_rows(blocks)

    schools_path = data_dir / "schools.geojson"
    schools_gdf = gpd.read_file(schools_path) if schools_path.exists() else None

    stats = build_stats(blocks, schools_gdf, blocks_path)
    indicators, category_counts = build_indicators(blocks, rows)
    charts = build_charts(rows)
    summary = build_summary(blocks)

    def roads_filter(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if "highway" not in gdf.columns:
            return gdf
        keep = {
            "motorway", "trunk", "primary", "secondary", "tertiary",
            "residential", "unclassified", "living_street", "pedestrian",
        }
        return gdf[gdf["highway"].isin(keep)]

    layers = {
        "boundary": load_layer(data_dir, "boundaries.geojson", columns=[]),
        "schools": load_layer(data_dir, "schools.geojson", columns=["capacity"]),
        "water": load_layer(
            data_dir, "water.geojson", columns=["name", "water", "natural"], simplify=0.00003
        ),
        "roads": load_layer(
            data_dir, "roads.geojson", columns=["highway", "name"], simplify=0.00003,
            filter_fn=roads_filter,
        ),
        "railways": load_layer(
            data_dir, "railways.geojson", columns=["name", "railway"], simplify=0.00003
        ),
    }

    data = {
        "meta": {
            "city": "Гусев",
            "generated_at": stats["generated_at"],
            "data_date": stats["data_date"],
            "population_reference": POPULATION_REFERENCE,
            "stats": stats,
        },
        "blocks": geojson_from_rows(blocks_wgs, rows),
        "layers": layers,
        "indicators": indicators,
        "categoryCounts": category_counts,
        "categoricalPalette": CATEGORICAL_PALETTE,
        "charts": charts,
        "summary": summary,
        "tableColumns": TABLE_COLUMNS,
    }
    return data, stats


def read_asset(filename: str) -> str | None:
    path = ASSETS_DIR / filename
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def inline_js(source: str) -> str:
    return source.replace("</script", "<\\/script")


def inline_css(source: str) -> str:
    return source.replace("</style", "<\\/style")


def build_assets() -> tuple[str, str]:
    """Возвращает (head_assets, body_assets).

    Если рядом со скриптом есть папка assets, Leaflet и Chart.js встраиваются
    прямо в HTML. Тогда dashboard не зависит от CDN (кроме подложки карты).
    Если папки нет, используется запасной вариант с CDN.
    """
    leaflet_css = read_asset("leaflet.css")
    leaflet_js = read_asset("leaflet.js")
    chart_js = read_asset("chart.umd.min.js")

    if leaflet_css and leaflet_js and chart_js:
        head_assets = f"<style>{inline_css(leaflet_css)}</style>"
        body_assets = (
            f"<script>{inline_js(leaflet_js)}</script>\n"
            f"<script>{inline_js(chart_js)}</script>"
        )
        return head_assets, body_assets

    head_assets = (
        '<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">'
    )
    body_assets = (
        '<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>\n'
        '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>'
    )
    return head_assets, body_assets


def render_html(data: dict[str, Any], stats: dict[str, Any]) -> str:
    data_json = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    data_json = data_json.replace("</", "<\\/")

    source_note = (
        f"Данные: <b>blocks_indicators.geojson</b> · обновлён {html.escape(stats['data_date'])} · "
        f"{stats['blocks']} {plural_ru(stats['blocks'], 'квартал', 'квартала', 'кварталов')} · "
        f"сгенерировано {html.escape(stats['generated_at'])}"
    )
    kpi_html = build_kpi(stats)
    short, full, limitations = build_conclusions(stats)
    head_assets, body_assets = build_assets()

    return (
        HTML_TEMPLATE.replace("__HEAD_ASSETS__", head_assets)
        .replace("__BODY_ASSETS__", body_assets)
        .replace("__SOURCE_NOTE__", source_note)
        .replace("__KPI_HTML__", kpi_html)
        .replace("__SHORT_CONCLUSIONS__", short)
        .replace("__FULL_CONCLUSIONS__", full)
        .replace("__LIMITATIONS__", limitations)
        .replace("__DATA_JSON__", data_json)
    )


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    data, stats = build_data(data_dir)
    html_text = render_html(data, stats)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"[готово] {output_path}")
    print(
        f"Размер: {size_mb:.1f} МБ | кварталов: {stats['blocks']} | "
        f"население: {stats['population']:,.0f} чел."
    )
    print("Откройте файл в браузере. Для подложки карты нужен интернет; Leaflet и Chart.js встроены в HTML.")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Гусев — городской dashboard</title>
__HEAD_ASSETS__
<style>
:root{
  --bg:#eef2f7;
  --card:#ffffff;
  --text:#0f172a;
  --muted:#64748b;
  --border:#e2e8f0;
  --accent:#2563eb;
  --radius:18px;
  --shadow:0 10px 28px rgba(15,23,42,.07);
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{
  margin:0;
  background:var(--bg);
  color:var(--text);
  font-family:"Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, Arial, sans-serif;
  line-height:1.45;
}
a{color:var(--accent)}
.hero{
  color:#fff;
  padding:46px 24px 40px;
  background:
    radial-gradient(circle at 85% 10%, rgba(56,189,248,.35), transparent 35%),
    radial-gradient(circle at 10% 90%, rgba(168,85,247,.25), transparent 38%),
    linear-gradient(135deg,#0f172a 0%,#172554 55%,#1e3a8a 100%);
}
.hero-inner{max-width:1440px;margin:0 auto}
.eyebrow{
  text-transform:uppercase;
  letter-spacing:.16em;
  font-size:12px;
  font-weight:700;
  color:#93c5fd;
  margin-bottom:12px;
}
.hero h1{
  margin:0 0 12px;
  font-size:clamp(30px,4vw,52px);
  line-height:1.06;
  letter-spacing:-.04em;
}
.subtitle{
  max-width:900px;
  margin:0;
  color:#dbeafe;
  font-size:17px;
}
.hero-meta{
  margin-top:22px;
  color:#bfdbfe;
  font-size:13px;
}
.container{max-width:1440px;margin:0 auto;padding:24px}
.kpi-grid{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
  gap:14px;
  margin-bottom:22px;
}
.kpi{
  background:linear-gradient(180deg,#fff,#f8fafc);
  border:1px solid var(--border);
  border-top:4px solid var(--accent);
  border-radius:16px;
  padding:16px 16px 14px;
  box-shadow:var(--shadow);
}
.kpi-label{
  color:var(--muted);
  font-size:12px;
  font-weight:700;
  text-transform:uppercase;
  letter-spacing:.06em;
}
.kpi-value{
  margin-top:8px;
  font-size:28px;
  line-height:1;
  font-weight:800;
  letter-spacing:-.03em;
}
.kpi-note{
  margin-top:8px;
  color:var(--muted);
  font-size:12px;
}
.card{
  background:var(--card);
  border:1px solid var(--border);
  border-radius:var(--radius);
  box-shadow:var(--shadow);
}
.map-layout{
  display:grid;
  grid-template-columns:minmax(0,2.1fr) minmax(310px,1fr);
  gap:18px;
  margin-bottom:22px;
}
.map-card{overflow:hidden}
.card-head{padding:18px 20px 14px}
.map-head{
  display:flex;
  align-items:flex-start;
  justify-content:space-between;
  gap:16px;
  flex-wrap:wrap;
}
.card h2,.card h3{margin:0}
.card h2{font-size:20px;letter-spacing:-.02em}
.card h3{font-size:15px;margin-top:2px}
.muted{color:var(--muted)}
.card-head p{margin:4px 0 0;font-size:13px}
.map-controls{
  display:flex;
  align-items:center;
  gap:8px;
  flex-wrap:wrap;
}
.map-controls label{font-size:12px;color:var(--muted);font-weight:700}
select,input,button{
  font:inherit;
  border:1px solid #cbd5e1;
  background:#fff;
  color:var(--text);
  border-radius:10px;
  padding:8px 10px;
  outline:none;
}
select:focus,input:focus,button:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(37,99,235,.12)}
button{cursor:pointer;font-weight:700}
button:hover{background:#f8fafc}
.map-wrap{position:relative;height:620px}
#map{height:100%;width:100%;background:#e2e8f0}
.legend{
  position:absolute;
  left:14px;
  bottom:14px;
  z-index:600;
  width:min(300px,calc(100% - 28px));
  max-height:240px;
  overflow:auto;
  background:rgba(255,255,255,.95);
  border:1px solid var(--border);
  border-radius:12px;
  padding:10px 12px;
  box-shadow:0 8px 20px rgba(15,23,42,.14);
  font-size:11px;
}
.legend-title{font-weight:800;margin-bottom:6px}
.legend-bar{height:10px;border-radius:999px;margin:5px 0}
.legend-ticks{display:flex;justify-content:space-between;color:var(--muted);gap:8px}
.legend-list{display:grid;gap:4px}
.legend-list div{display:flex;align-items:center;gap:6px}
.legend-list i{width:10px;height:10px;border-radius:3px;display:inline-block;flex:0 0 auto}
.legend-list span{margin-left:auto;color:var(--muted);white-space:nowrap}
.insight-card{padding:20px}
.insight-list,.clean-list{margin:10px 0 0;padding-left:18px;color:#334155;font-size:13px}
.insight-list li,.clean-list li{margin:7px 0}
.divider{height:1px;background:var(--border);margin:18px 0}
.block-info{
  min-height:120px;
  border:1px dashed #cbd5e1;
  border-radius:12px;
  padding:12px;
  background:#f8fafc;
  font-size:12px;
}
.info-head{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px}
.info-head b{font-size:14px}
.info-close{border:0;background:#e2e8f0;border-radius:8px;padding:2px 8px;font-size:16px;line-height:1.2}
.info-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px 10px}
.info-grid div{display:flex;flex-direction:column;gap:1px}
.info-grid span{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.04em}
.info-grid b{font-size:12px;font-weight:700}
.section{margin:26px 0}
.section-head{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-bottom:12px}
.section-head h2{margin:0;font-size:24px;letter-spacing:-.03em}
.section-head p{margin:2px 0 0;color:var(--muted);font-size:13px}
.chart-grid{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(330px,1fr));
  gap:16px;
}
.chart-card{padding:16px;min-height:350px;display:flex;flex-direction:column}
.chart-title{font-size:15px;font-weight:800;letter-spacing:-.01em}
.chart-note{color:var(--muted);font-size:11px;margin-top:3px}
.chart-box{position:relative;flex:1;min-height:250px;margin-top:10px}
.chart-box canvas{width:100%!important;height:100%!important}
.table-card{padding:14px}
.table-toolbar{
  display:flex;
  align-items:center;
  gap:10px;
  flex-wrap:wrap;
  margin:0 0 12px;
}
.table-toolbar input{min-width:240px;flex:1;max-width:420px}
.table-toolbar .counter{color:var(--muted);font-size:12px}
.table-wrap{
  max-height:640px;
  overflow:auto;
  border:1px solid var(--border);
  border-radius:12px;
}
table{border-collapse:separate;border-spacing:0;width:100%;font-size:12px}
thead th{
  position:sticky;
  top:0;
  z-index:2;
  background:#f8fafc;
  text-align:left;
  padding:10px 8px;
  border-bottom:1px solid var(--border);
  cursor:pointer;
  white-space:nowrap;
  user-select:none;
}
thead th:hover{background:#f1f5f9}
tbody td{padding:8px;border-bottom:1px solid #f1f5f9;white-space:nowrap}
tbody tr:hover{background:#f8fafc}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
th.sorted[data-dir="1"]::after{content:" ▲";color:var(--accent)}
th.sorted[data-dir="-1"]::after{content:" ▼";color:var(--accent)}
.summary-table td:first-child{white-space:normal;min-width:230px}
.conclusion-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
.conclusion-card{padding:18px;border:1px solid var(--border);border-radius:14px;background:#fcfdff}
.conclusion-card h3{font-size:15px;margin:0 0 8px}
.conclusion-card ul{margin:0;padding-left:18px;color:#334155;font-size:13px}
.conclusion-card li{margin:7px 0}
.note-box{padding:18px;border-left:4px solid #f59e0b;background:#fffbeb;border-radius:12px;color:#78350f;font-size:13px}
.footer{max-width:1440px;margin:0 auto;padding:10px 24px 40px;color:var(--muted);font-size:12px}
.error-box{padding:24px;color:#991b1b;background:#fef2f2;border:1px solid #fecaca;border-radius:12px;margin:14px}
.leaflet-tooltip.block-tooltip{
  background:rgba(15,23,42,.94);
  color:#fff;
  border:0;
  border-radius:10px;
  box-shadow:0 8px 24px rgba(15,23,42,.25);
  padding:8px 10px;
  font-size:11px;
  line-height:1.4;
}
.leaflet-tooltip.block-tooltip .tt-title{font-weight:800;font-size:12px;margin-bottom:3px}
.leaflet-tooltip.block-tooltip .tt-row{display:flex;justify-content:space-between;gap:14px}
.leaflet-tooltip.block-tooltip .tt-row span{color:#cbd5e1}
@media (max-width:980px){
  .map-layout{grid-template-columns:1fr}
  .map-wrap{height:520px}
}
@media (max-width:620px){
  .container{padding:14px}
  .hero{padding:32px 16px 28px}
  .map-wrap{height:440px}
  .kpi-grid{grid-template-columns:1fr 1fr}
  .chart-grid{grid-template-columns:1fr}
  .table-toolbar input{min-width:0}
}
</style>
</head>
<body>
<header class="hero">
  <div class="hero-inner">
    <div class="eyebrow">BlocksNet · учебное исследование</div>
    <h1>Гусев: городской dashboard</h1>
    <p class="subtitle">Кварталы, население, плотность, морфотипы, школы и пешеходная доступность. Все показатели уже рассчитаны и встроены в страницу.</p>
    <div class="hero-meta">__SOURCE_NOTE__</div>
  </div>
</header>

<main class="container">
  <section class="kpi-grid">__KPI_HTML__</section>

  <section class="map-layout">
    <div class="card map-card">
      <div class="card-head map-head">
        <div>
          <h2>Карта кварталов</h2>
          <p class="muted">Выберите показатель и нажмите на квартал, чтобы увидеть все значения.</p>
        </div>
        <div class="map-controls">
          <label for="indicatorSelect">Показатель</label>
          <select id="indicatorSelect"></select>
          <button id="resetView" type="button">Сбросить вид</button>
        </div>
      </div>
      <div class="map-wrap">
        <div id="map"></div>
        <div id="legend" class="legend"></div>
      </div>
    </div>

    <aside class="card insight-card">
      <h2>Быстрые выводы</h2>
      __SHORT_CONCLUSIONS__
      <div class="divider"></div>
      <h3>Выбранный квартал</h3>
      <div id="blockInfo" class="block-info muted">Нажмите на квартал на карте, чтобы увидеть все показатели.</div>
    </aside>
  </section>
  <section class="section">
    <div class="section-head">
      <div><h2>Структура территории</h2><p>Land use, население и морфотипы застройки.</p></div>
    </div>
    <div class="chart-grid">
      <div class="card chart-card">
        <div class="chart-title">Land use</div>
        <div class="chart-note">Преобладающий тип использования квартала; 125 кварталов без данных.</div>
        <div class="chart-box"><canvas id="chart-landuse"></canvas></div>
      </div>
      <div class="card chart-card">
        <div class="chart-title">Топ-12 кварталов по населению</div>
        <div class="chart-note">Оценка населения по жилой площади.</div>
        <div class="chart-box"><canvas id="chart-population"></canvas></div>
      </div>
      <div class="card chart-card">
        <div class="chart-title">Морфотипы Spacematrix</div>
        <div class="chart-note">«Не определён» — незастроенные кварталы.</div>
        <div class="chart-box"><canvas id="chart-spacematrix"></canvas></div>
      </div>
      <div class="card chart-card">
        <div class="chart-title">Морфотипы Strelka</div>
        <div class="chart-note">Альтернативная классификация застройки.</div>
        <div class="chart-box"><canvas id="chart-strelka"></canvas></div>
      </div>
    </div>
  </section>

  <section class="section">
    <div class="section-head">
      <div><h2>Плотность и этажность</h2><p>Только застроенные кварталы — их 102 из 233.</p></div>
    </div>
    <div class="chart-grid">
      <div class="card chart-card">
        <div class="chart-title">Распределение FSI</div>
        <div class="chart-note">FSI = площадь застройки / площадь квартала.</div>
        <div class="chart-box"><canvas id="chart-fsi"></canvas></div>
      </div>
      <div class="card chart-card">
        <div class="chart-title">Этажность L</div>
        <div class="chart-note">Средняя этажность застроенных кварталов.</div>
        <div class="chart-box"><canvas id="chart-l"></canvas></div>
      </div>
      <div class="card chart-card">
        <div class="chart-title">FSI и этажность</div>
        <div class="chart-note">Размер круга — население квартала.</div>
        <div class="chart-box"><canvas id="chart-fsi-l"></canvas></div>
      </div>
      <div class="card chart-card">
        <div class="chart-title">Распределение MXI</div>
        <div class="chart-note">MXI — доля жилой функции в застройке.</div>
        <div class="chart-box"><canvas id="chart-mxi"></canvas></div>
      </div>
    </div>
  </section>

  <section class="section">
    <div class="section-head">
      <div><h2>Школы и доступность</h2><p>Нормативный спрос, ёмкость и пешеходная доступность.</p></div>
    </div>
    <div class="chart-grid">
      <div class="card chart-card">
        <div class="chart-title">Обеспеченность школами</div>
        <div class="chart-note">Только 99 кварталов с населением; 80 из них имеют нулевую обеспеченность.</div>
        <div class="chart-box"><canvas id="chart-provision"></canvas></div>
      </div>
      <div class="card chart-card">
        <div class="chart-title">Доступность территории</div>
        <div class="chart-note">Среднее время пешей доступности, минуты.</div>
        <div class="chart-box"><canvas id="chart-access"></canvas></div>
      </div>
      <div class="card chart-card">
        <div class="chart-title">Спрос и ёмкость школ</div>
        <div class="chart-note">Топ-10 кварталов по нормативному спросу.</div>
        <div class="chart-box"><canvas id="chart-demand"></canvas></div>
      </div>
    </div>
  </section>
  <section class="section">
    <div class="section-head">
      <div><h2>Сводная статистика</h2><p>Средние, медианы и диапазоны по всем кварталам.</p></div>
    </div>
    <div class="card table-card">
      <div class="table-wrap">
        <table id="summaryTable" class="summary-table">
          <thead>
            <tr>
              <th>Показатель</th>
              <th class="num">N</th>
              <th class="num">Среднее</th>
              <th class="num">Медиана</th>
              <th class="num">Мин.</th>
              <th class="num">Макс.</th>
              <th class="num">Пропуски</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
    </div>
  </section>

  <section class="section">
    <div class="section-head">
      <div><h2>Все показатели по кварталам</h2><p>Нажмите на заголовок столбца для сортировки; используйте поиск для фильтра.</p></div>
    </div>
    <div class="card table-card">
      <div class="table-toolbar">
        <input id="tableSearch" type="search" placeholder="Поиск по таблице…" autocomplete="off">
        <span id="tableCount" class="counter"></span>
        <button id="downloadCsv" type="button">Скачать CSV</button>
      </div>
      <div class="table-wrap">
        <table id="blocksTable">
          <thead></thead>
          <tbody></tbody>
        </table>
      </div>
    </div>
  </section>

  <section class="section">
    <div class="section-head">
      <div><h2>Выводы</h2><p>Что показывают расчёты и где есть ограничения.</p></div>
    </div>
    __FULL_CONCLUSIONS__
  </section>

  <section class="section">
    <div class="section-head">
      <div><h2>Методика и ограничения</h2><p>Коротко о том, как читать dashboard.</p></div>
    </div>
    <div class="card" style="padding:18px">
      <div class="note-box">
        <b>Важно:</b> это учебная оценка по данным OpenStreetMap и модели BlocksNet, а не официальная статистика.
      </div>
      <div style="height:14px"></div>
      __LIMITATIONS__
      <p class="muted" style="font-size:12px;margin:12px 0 0">
        Показатели FSI, GSI, MXI, L, OSR, обеспеченность и доступность рассчитаны в ноутбуке
        <code>gusev.ipynb</code> и сохранены в <code>data/blocks_indicators.geojson</code>.
      </p>
    </div>
  </section>
</main>

<footer class="footer">
  <div>Источники: OpenStreetMap contributors, BlocksNet 1.0.0a9. Dashboard собирается скриптом <code>build_dashboard.py</code>.</div>
  <div style="margin-top:6px">Для подложки карты (тайлов) нужен доступ в интернет. Leaflet, Chart.js и сами данные встроены в HTML.</div>
</footer>
__BODY_ASSETS__
<script>
const DATA = __DATA_JSON__;
const BLOCKS = DATA.blocks.features.map(feature => feature.properties);
const CATEGORY_COLORS = DATA.categoricalPalette;
const MISSING_COLOR = "#cbd5e1";

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[character]));
}

function fmt(value, digits = 0) {
  if (value === null || value === undefined || value === "" || Number.isNaN(Number(value))) {
    return "—";
  }
  return Number(value).toLocaleString("ru-RU", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits
  });
}

function fmtValue(value, digits = 2) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "number") return fmt(value, digits);
  return escapeHtml(value);
}

function hexToRgb(hex) {
  const normalized = hex.replace("#", "");
  return {
    r: parseInt(normalized.slice(0, 2), 16),
    g: parseInt(normalized.slice(2, 4), 16),
    b: parseInt(normalized.slice(4, 6), 16)
  };
}

function mixColor(first, second, ratio) {
  const a = hexToRgb(first);
  const b = hexToRgb(second);
  const r = Math.round(a.r + (b.r - a.r) * ratio);
  const g = Math.round(a.g + (b.g - a.g) * ratio);
  const blue = Math.round(a.b + (b.b - a.b) * ratio);
  return `rgb(${r},${g},${blue})`;
}

function interpolateColor(palette, ratio) {
  const position = Math.max(0, Math.min(1, ratio)) * (palette.length - 1);
  const index = Math.min(Math.floor(position), palette.length - 2);
  return mixColor(palette[index], palette[index + 1], position - index);
}

function numericColor(value, meta) {
  if (value === null || value === undefined || value === "" || Number.isNaN(Number(value))) {
    return MISSING_COLOR;
  }
  const number = Number(value);
  let ratio;
  if (meta.log) {
    const low = Math.log1p(Math.max(0, meta.min));
    const high = Math.log1p(Math.max(0, meta.max));
    ratio = (Math.log1p(Math.max(0, number)) - low) / (high - low || 1);
  } else {
    ratio = (number - meta.min) / (meta.max - meta.min || 1);
  }
  return interpolateColor(meta.palette, ratio);
}

function categoryColor(indicator, label) {
  if (label === "Не определён") return MISSING_COLOR;
  const items = DATA.categoryCounts[indicator] || [];
  const index = items.findIndex(item => item.label === label);
  return CATEGORY_COLORS[(index >= 0 ? index : 0) % CATEGORY_COLORS.length];
}

function setSelectOptions() {
  const select = document.getElementById("indicatorSelect");
  Object.entries(DATA.indicators).forEach(([key, meta]) => {
    const option = document.createElement("option");
    option.value = key;
    option.textContent = meta.label;
    select.appendChild(option);
  });
}
let currentIndicator = "population";
let selectedLayer = null;
let blocksLayer = null;
let map = null;
let initialBounds = null;
let selectedProperties = null;

const DETAIL_FIELDS = [
  ["population", "Население", 0],
  ["land_use", "Land use", null],
  ["fsi", "FSI", 3],
  ["gsi", "GSI", 3],
  ["mxi", "MXI", 3],
  ["l", "L, этажи", 2],
  ["osr", "OSR", 2],
  ["share_living", "Жилая / застройка", 3],
  ["share_non_living", "Нежилая / застройка", 3],
  ["spacematrix_cluster", "Кластер Spacematrix", null],
  ["morphotype_spacematrix", "Морфотип Spacematrix", null],
  ["morphotype_strelka", "Морфотип Strelka", null],
  ["demand", "Спрос на школы", 0],
  ["capacity", "Ёмкость школ", 0],
  ["provision_strong", "Обеспеченность strong", 3],
  ["provision_weak", "Обеспеченность weak", 3],
  ["area_accessibility", "Доступность территории, мин", 2],
  ["mean_accessibility", "Средняя доступность, мин", 2]
];

function styleFeature(feature) {
  const meta = DATA.indicators[currentIndicator];
  const value = feature.properties[currentIndicator];
  let fill = MISSING_COLOR;
  if (meta.type === "categorical") {
    fill = categoryColor(currentIndicator, value);
  } else {
    fill = numericColor(value, meta);
  }
  return {
    color: "#ffffff",
    weight: 0.8,
    opacity: 0.9,
    fillColor: fill,
    fillOpacity: 0.78
  };
}

function tooltipHtml(properties) {
  return `
    <div class="tt-title">Квартал ${escapeHtml(properties.block_id)}</div>
    <div class="tt-row"><span>Население</span><b>${fmt(properties.population, 0)}</b></div>
    <div class="tt-row"><span>Land use</span><b>${escapeHtml(properties.land_use || "—")}</b></div>
    <div class="tt-row"><span>FSI / GSI</span><b>${fmt(properties.fsi, 3)} / ${fmt(properties.gsi, 3)}</b></div>
    <div class="tt-row"><span>L / MXI</span><b>${fmt(properties.l, 2)} / ${fmt(properties.mxi, 3)}</b></div>
    <div class="tt-row"><span>Обеспеченность</span><b>${fmt(properties.provision_strong, 3)}</b></div>
  `;
}

function showBlockInfo(properties) {
  selectedProperties = properties;
  const container = document.getElementById("blockInfo");
  if (!container) return;
  const fields = DETAIL_FIELDS.map(([key, label, digits]) => {
    const value = properties[key];
    let rendered;
    if (value === null || value === undefined || value === "") {
      rendered = "—";
    } else if (typeof value === "number") {
      rendered = fmt(value, digits === null ? 2 : digits);
    } else {
      rendered = escapeHtml(value);
    }
    return `<div><span>${escapeHtml(label)}</span><b>${rendered}</b></div>`;
  }).join("");
  container.classList.remove("muted");
  container.innerHTML = `
    <div class="info-head">
      <b>Квартал ${escapeHtml(properties.block_id)}</b>
      <button class="info-close" type="button" title="Закрыть">×</button>
    </div>
    <div class="info-grid">${fields}</div>
  `;
  const closeButton = container.querySelector(".info-close");
  if (closeButton) {
    closeButton.addEventListener("click", () => {
      selectedProperties = null;
      container.classList.add("muted");
      container.textContent = "Нажмите на квартал на карте, чтобы увидеть все показатели.";
      if (selectedLayer && blocksLayer) {
        blocksLayer.resetStyle(selectedLayer);
        selectedLayer = null;
      }
    });
  }
}

function updateLegend() {
  const legend = document.getElementById("legend");
  if (!legend) return;
  const meta = DATA.indicators[currentIndicator];
  if (!meta) {
    legend.innerHTML = "";
    return;
  }
  if (meta.type === "numeric") {
    const gradient = meta.palette.join(", ");
    const logNote = meta.log ? '<div class="legend-note">логарифмическая шкала</div>' : "";
    legend.innerHTML = `
      <div class="legend-title">${escapeHtml(meta.label)}</div>
      <div class="legend-bar" style="background:linear-gradient(90deg,${gradient})"></div>
      <div class="legend-ticks"><span>${fmt(meta.min, meta.digits)}</span><span>${fmt(meta.max, meta.digits)}</span></div>
      ${logNote}
    `;
  } else {
    const items = (DATA.categoryCounts[currentIndicator] || []).slice(0, 12);
    legend.innerHTML = `
      <div class="legend-title">${escapeHtml(meta.label)}</div>
      <div class="legend-list">
        ${items.map(item => `
          <div><i style="background:${categoryColor(currentIndicator, item.label)}"></i>
          <span style="margin-left:0">${escapeHtml(item.label)}</span>
          <span>${item.count}</span></div>
        `).join("")}
      </div>
    `;
  }
}

function applyIndicator() {
  if (!blocksLayer) return;
  blocksLayer.setStyle(styleFeature);
  if (selectedLayer) {
    blocksLayer.resetStyle(selectedLayer);
    selectedLayer = null;
  }
  updateLegend();
}

function initMap() {
  map = L.map("map", { preferCanvas: true, zoomControl: true }).setView([54.5928, 22.1995], 12);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxZoom: 19
  }).addTo(map);

  const overlays = {};
  if (DATA.layers.boundary) {
    const boundary = L.geoJSON(DATA.layers.boundary, {
      style: { color: "#0f172a", weight: 2, fill: false, dashArray: "5 4" }
    }).addTo(map);
    overlays["Граница города"] = boundary;
  }
  if (DATA.layers.water) {
    const water = L.geoJSON(DATA.layers.water, {
      style: { color: "#60a5fa", weight: 1, fillColor: "#bfdbfe", fillOpacity: 0.45 }
    }).addTo(map);
    overlays["Вода"] = water;
  }
  if (DATA.layers.roads) {
    const roads = L.geoJSON(DATA.layers.roads, {
      style: { color: "#94a3b8", weight: 1, opacity: 0.55 }
    });
    overlays["Дороги"] = roads;
  }
  if (DATA.layers.railways) {
    const railways = L.geoJSON(DATA.layers.railways, {
      style: { color: "#475569", weight: 1.2, opacity: 0.7, dashArray: "4 3" }
    });
    overlays["Железные дороги"] = railways;
  }
  if (DATA.layers.schools) {
    const schools = L.geoJSON(DATA.layers.schools, {
      pointToLayer: (feature, latlng) => L.circleMarker(latlng, {
        radius: 6, color: "#ffffff", weight: 2, fillColor: "#7c3aed", fillOpacity: 1
      }),
      onEachFeature: (feature, layer) => {
        layer.bindTooltip(`Школа · ёмкость ${fmt(feature.properties.capacity, 0)} мест`, { sticky: true });
      }
    }).addTo(map);
    overlays["Школы"] = schools;
  }

  blocksLayer = L.geoJSON(DATA.blocks, {
    style: styleFeature,
    onEachFeature: (feature, layer) => {
      layer.bindTooltip(tooltipHtml(feature.properties), {
        sticky: true,
        className: "block-tooltip"
      });
      layer.on("mouseover", () => {
        if (layer !== selectedLayer) {
          layer.setStyle({ weight: 2.2, color: "#0f172a" });
        }
      });
      layer.on("mouseout", () => {
        if (layer !== selectedLayer) {
          blocksLayer.resetStyle(layer);
        }
      });
      layer.on("click", () => {
        if (selectedLayer && selectedLayer !== layer) {
          blocksLayer.resetStyle(selectedLayer);
        }
        selectedLayer = layer;
        layer.setStyle({ weight: 3, color: "#0f172a" });
        layer.bringToFront();
        showBlockInfo(feature.properties);
      });
    }
  }).addTo(map);
  overlays["Кварталы"] = blocksLayer;

  L.control.layers({}, overlays, { collapsed: false, position: "topright" }).addTo(map);

  initialBounds = blocksLayer.getBounds();
  if (initialBounds.isValid()) {
    map.fitBounds(initialBounds, { padding: [24, 24] });
  }
  setTimeout(() => map.invalidateSize(), 150);
  updateLegend();
}
const chartInstances = [];

function initChart(id, config) {
  const element = document.getElementById(id);
  if (!element || !window.Chart) return;
  chartInstances.push(new Chart(element, config));
}

const SHORT_LABELS = {
  "Малоэтажная низкоплотная жилая": "Мал. низкоплотная жилая",
  "Малоэтажная низкоплотная нежилая": "Мал. низкоплотная нежилая",
  "Малоэтажная низкоплотная смешанная": "Мал. низкоплотная смеш.",
  "Среднеэтажная низкоплотная жилая": "Сред. низкоплотная жилая",
  "Среднеэтажная низкоплотная нежилая": "Сред. низкоплотная нежилая",
  "Индивидуальная жилая застройка": "Индивидуальная жилая",
  "Малоэтажная модель": "Малоэтажная модель",
  "Малоэтажная нежилая": "Малоэтажная нежилая",
  "Среднеэтажная жилая": "Среднеэтажная жилая",
  "Среднеэтажная нежилая": "Среднеэтажная нежилая",
  "Не определён": "Не определён"
};

function shortLabel(value) {
  const text = String(value);
  return SHORT_LABELS[text] || (text.length > 30 ? text.slice(0, 28) + "…" : text);
}

function initCharts() {
  if (!window.Chart) return;
  Chart.defaults.font.family = '"Segoe UI", system-ui, sans-serif';
  Chart.defaults.color = "#475569";

  const landUse = DATA.charts.landUse;
  initChart("chart-landuse", {
    type: "doughnut",
    data: {
      labels: landUse.labels,
      datasets: [{
        data: landUse.values,
        backgroundColor: landUse.labels.map(label => categoryColor("land_use", label)),
        borderColor: "#ffffff",
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "58%",
      plugins: {
        legend: { position: "bottom", labels: { boxWidth: 10, font: { size: 10 } } }
      }
    }
  });

  const population = DATA.charts.populationTop;
  initChart("chart-population", {
    type: "bar",
    data: {
      labels: population.labels,
      datasets: [{ data: population.values, backgroundColor: "#f97316", borderRadius: 4 }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { beginAtZero: true, ticks: { precision: 0, font: { size: 10 } } },
        y: { ticks: { font: { size: 10 } } }
      }
    }
  });

  const spacematrix = DATA.charts.spacematrix;
  initChart("chart-spacematrix", {
    type: "bar",
    data: {
      labels: spacematrix.labels,
      datasets: [{
        data: spacematrix.values,
        backgroundColor: spacematrix.labels.map(label => categoryColor("morphotype_spacematrix", label)),
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { beginAtZero: true, ticks: { precision: 0, font: { size: 10 } } },
        y: {
          ticks: {
            font: { size: 10 },
            callback: function(value) { return shortLabel(this.getLabelForValue(value)); }
          }
        }
      }
    }
  });

  const strelka = DATA.charts.strelka;
  initChart("chart-strelka", {
    type: "bar",
    data: {
      labels: strelka.labels,
      datasets: [{
        data: strelka.values,
        backgroundColor: strelka.labels.map(label => categoryColor("morphotype_strelka", label)),
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { beginAtZero: true, ticks: { precision: 0, font: { size: 10 } } },
        y: {
          ticks: {
            font: { size: 10 },
            callback: function(value) { return shortLabel(this.getLabelForValue(value)); }
          }
        }
      }
    }
  });

  const fsi = DATA.charts.fsiHist;
  initChart("chart-fsi", {
    type: "bar",
    data: {
      labels: fsi.labels,
      datasets: [{ data: fsi.values, backgroundColor: "#3b82f6", borderRadius: 3 }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { maxRotation: 45, minRotation: 0, font: { size: 9 } } },
        y: { beginAtZero: true, ticks: { precision: 0, font: { size: 10 } } }
      }
    }
  });

  const floors = DATA.charts.lCounts;
  initChart("chart-l", {
    type: "bar",
    data: {
      labels: floors.labels,
      datasets: [{ data: floors.values, backgroundColor: "#10b981", borderRadius: 4 }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: "Этажность L", font: { size: 10 } }, ticks: { font: { size: 10 } } },
        y: { beginAtZero: true, ticks: { precision: 0, font: { size: 10 } } }
      }
    }
  });

  initChart("chart-fsi-l", {
    type: "bubble",
    data: {
      datasets: [{
        label: "Кварталы",
        data: DATA.charts.fsiLScatter,
        backgroundColor: "rgba(37,99,235,.42)",
        borderColor: "#1d4ed8",
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: context => {
              const point = context.raw;
              return `Квартал ${point.block_id}: FSI ${fmt(point.x, 3)}, L ${fmt(point.y, 2)}, население ${fmt(point.population, 0)}`;
            }
          }
        }
      },
      scales: {
        x: { title: { display: true, text: "FSI, м²/м²", font: { size: 10 } }, beginAtZero: true },
        y: { title: { display: true, text: "L, этажи", font: { size: 10 } }, beginAtZero: true }
      }
    }
  });

  const mxi = DATA.charts.mxiHist;
  initChart("chart-mxi", {
    type: "bar",
    data: {
      labels: mxi.labels,
      datasets: [{ data: mxi.values, backgroundColor: "#ca8a04", borderRadius: 3 }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { maxRotation: 45, minRotation: 0, font: { size: 9 } } },
        y: { beginAtZero: true, ticks: { precision: 0, font: { size: 10 } } }
      }
    }
  });

  const provision = DATA.charts.provision;
  initChart("chart-provision", {
    type: "bar",
    data: {
      labels: provision.labels,
      datasets: [{
        data: provision.values,
        backgroundColor: ["#dc2626", "#f59e0b", "#f59e0b", "#16a34a"],
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: "provision_strong", font: { size: 10 } }, ticks: { font: { size: 10 } } },
        y: { beginAtZero: true, ticks: { precision: 0, font: { size: 10 } } }
      }
    }
  });

  const access = DATA.charts.accessHist;
  initChart("chart-access", {
    type: "bar",
    data: {
      labels: access.labels,
      datasets: [{ data: access.values, backgroundColor: "#8b5cf6", borderRadius: 3 }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { maxRotation: 45, minRotation: 0, font: { size: 9 } } },
        y: { beginAtZero: true, ticks: { precision: 0, font: { size: 10 } } }
      }
    }
  });

  const demand = DATA.charts.demandCapacity;
  initChart("chart-demand", {
    type: "bar",
    data: {
      labels: demand.labels,
      datasets: [
        { label: "Спрос", data: demand.demand, backgroundColor: "#ef4444", borderRadius: 4 },
        { label: "Ёмкость", data: demand.capacity, backgroundColor: "#10b981", borderRadius: 4 }
      ]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: "bottom", labels: { boxWidth: 10, font: { size: 10 } } } },
      scales: {
        x: { beginAtZero: true, ticks: { precision: 0, font: { size: 10 } } },
        y: { ticks: { font: { size: 10 } } }
      }
    }
  });
}
let sortKey = "block_id";
let sortDirection = 1;

function renderSummary() {
  const tbody = document.querySelector("#summaryTable tbody");
  if (!tbody) return;
  tbody.innerHTML = DATA.summary.map(row => `
    <tr>
      <td>${escapeHtml(row.label)}</td>
      <td class="num">${fmt(row.count, 0)}</td>
      <td class="num">${fmt(row.mean, row.digits)}</td>
      <td class="num">${fmt(row.median, row.digits)}</td>
      <td class="num">${fmt(row.min, row.digits)}</td>
      <td class="num">${fmt(row.max, row.digits)}</td>
      <td class="num">${fmt(row.missing, 0)}</td>
    </tr>
  `).join("");
}

function initTable() {
  const table = document.getElementById("blocksTable");
  if (!table) return;
  const thead = table.querySelector("thead");
  const tbody = table.querySelector("tbody");
  const search = document.getElementById("tableSearch");
  const counter = document.getElementById("tableCount");

  thead.innerHTML = `<tr>${DATA.tableColumns.map(column =>
    `<th data-key="${escapeHtml(column.key)}" data-type="${column.type}" class="${column.type === "number" ? "num" : ""}">${escapeHtml(column.label)}</th>`
  ).join("")}</tr>`;

  thead.querySelectorAll("th").forEach(header => {
    header.addEventListener("click", () => {
      const key = header.dataset.key;
      if (sortKey === key) {
        sortDirection *= -1;
      } else {
        sortKey = key;
        sortDirection = 1;
      }
      renderTable();
    });
  });

  function renderTable() {
    const query = (search?.value || "").trim().toLowerCase();
    let rows = BLOCKS;
    if (query) {
      rows = BLOCKS.filter(row =>
        DATA.tableColumns.some(column =>
          String(row[column.key] ?? "").toLowerCase().includes(query)
        )
      );
    }
    rows = rows.slice().sort((a, b) => {
      const first = a[sortKey];
      const second = b[sortKey];
      if (first === null || first === undefined) return 1;
      if (second === null || second === undefined) return -1;
      if (typeof first === "number" && typeof second === "number") {
        return (first - second) * sortDirection;
      }
      return String(first).localeCompare(String(second), "ru") * sortDirection;
    });

    tbody.innerHTML = rows.map(row => {
      const cells = DATA.tableColumns.map(column => {
        const value = row[column.key];
        if (column.type === "number") {
          return `<td class="num">${fmt(value, column.digits)}</td>`;
        }
        return `<td>${value === null || value === undefined || value === "" ? "—" : escapeHtml(value)}</td>`;
      }).join("");
      return `<tr>${cells}</tr>`;
    }).join("");

    if (counter) {
      counter.textContent = `Показано ${fmt(rows.length, 0)} из ${fmt(BLOCKS.length, 0)}`;
    }
    thead.querySelectorAll("th").forEach(header => {
      const active = header.dataset.key === sortKey;
      header.classList.toggle("sorted", active);
      if (active) header.dataset.dir = String(sortDirection);
      else delete header.dataset.dir;
    });
  }

  if (search) search.addEventListener("input", renderTable);
  renderTable();
}

function downloadCsv() {
  const columns = DATA.tableColumns;
  const csvCell = value => {
    if (value === null || value === undefined) return '""';
    return '"' + String(value).replace(/"/g, '""') + '"';
  };
  const lines = [
    columns.map(column => csvCell(column.label)).join(";"),
    ...BLOCKS.map(row => columns.map(column => csvCell(row[column.key])).join(";"))
  ];
  const blob = new Blob(["\ufeff" + lines.join("\r\n")], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "gusev_blocks_indicators.csv";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

function initDashboard() {
  setSelectOptions();
  const select = document.getElementById("indicatorSelect");
  if (select) {
    select.value = currentIndicator;
    select.addEventListener("change", () => {
      currentIndicator = select.value;
      applyIndicator();
    });
  }
  const reset = document.getElementById("resetView");
  if (reset) {
    reset.addEventListener("click", () => {
      if (map && initialBounds && initialBounds.isValid()) {
        map.fitBounds(initialBounds, { padding: [24, 24] });
      }
    });
  }

  if (window.L) {
    initMap();
  } else {
    const mapElement = document.getElementById("map");
    if (mapElement) {
      mapElement.innerHTML = '<div class="error-box">Карта не загрузилась: нет доступа к Leaflet CDN. Проверьте интернет и обновите страницу.</div>';
    }
  }

  if (window.Chart) {
    initCharts();
  } else {
    document.querySelectorAll(".chart-box").forEach(element => {
      element.innerHTML = '<div class="error-box">График не загрузился: нет доступа к Chart.js CDN.</div>';
    });
  }

  renderSummary();
  initTable();
  const download = document.getElementById("downloadCsv");
  if (download) download.addEventListener("click", downloadCsv);
}

document.addEventListener("DOMContentLoaded", initDashboard);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
