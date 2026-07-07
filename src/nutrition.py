import json
import os
import warnings
from pathlib import Path

import requests

from src.config import Config

API_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
CACHE_PATH = Config.DATA_DIR / "nutrition_cache.json"
DENSITY_PATH = Config.DATA_DIR / "food_densities.json"
DEFAULT_DENSITY = 0.70

NUTRIENT_IDS = {
    "kcal_per_100g": 1008,
    "protein_per_100g": 1003,
    "carbs_per_100g": 1005,
    "fat_per_100g": 1004,
}


def _load_json(path: Path, default):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return default


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def get_density(dish_name: str) -> float:
    densities = _load_json(DENSITY_PATH, {})
    if dish_name not in densities:
        warnings.warn(
            f"No density entry for '{dish_name}'; "
            f"using default {DEFAULT_DENSITY} g/cm3."
        )
    return float(densities.get(dish_name, DEFAULT_DENSITY))


def grams_from_volume(dish_name: str, volume_cm3: float) -> float:
    return volume_cm3 * get_density(dish_name)


def _parse_nutrients(food: dict) -> dict:
    by_id = {}
    for nut in food.get("foodNutrients", []):
        nid = nut.get("nutrientId")
        if nid is not None and "value" in nut:
            by_id[nid] = nut["value"]

    result = {}
    for key, nid in NUTRIENT_IDS.items():
        if nid not in by_id:
            return None
        result[key] = float(by_id[nid])
    return result


def get_nutrition(dish_name: str):
    cache = _load_json(CACHE_PATH, {})
    if dish_name in cache:
        return cache[dish_name]

    api_key = os.environ.get("USDA_API_KEY")
    if not api_key:
        warnings.warn(
            "USDA_API_KEY environment variable not set; "
            "cannot query FoodData Central."
        )
        return None

    query = dish_name.replace("_", " ")
    params = {
        "api_key": api_key,
        "query": query,
        "pageSize": 10,
        "dataType": ["SR Legacy", "Foundation", "Survey (FNDDS)"],
    }

    try:
        response = requests.get(API_URL, params=params, timeout=15)
        response.raise_for_status()
        foods = response.json().get("foods", [])
    except requests.RequestException as e:
        warnings.warn(f"USDA API request failed for '{dish_name}': {e}")
        return None

    for food in foods:
        nutrition = _parse_nutrients(food)
        if nutrition is not None:
            nutrition["source"] = food.get("description", query)
            nutrition["fdc_id"] = food.get("fdcId")
            cache[dish_name] = nutrition
            _save_cache(cache)
            return nutrition

    warnings.warn(f"No usable USDA result for '{dish_name}'.")
    return None


def compute_meal_nutrition(dish_name: str, estimated_grams: float):
    per_100g = get_nutrition(dish_name)
    if per_100g is None:
        return None

    factor = estimated_grams / 100.0
    return {
        "dish": dish_name,
        "grams": round(estimated_grams, 1),
        "kcal": round(per_100g["kcal_per_100g"] * factor, 1),
        "protein_g": round(per_100g["protein_per_100g"] * factor, 1),
        "carbs_g": round(per_100g["carbs_per_100g"] * factor, 1),
        "fat_g": round(per_100g["fat_per_100g"] * factor, 1),
        "source": per_100g.get("source", ""),
    }
