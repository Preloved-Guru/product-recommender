from typing import List, Dict, Any, Optional
import pandas as pd
import re

# Constants for scoring
USAGE_OPTIONS = [
    "Road running (paved surfaces)",
    "Trail running (uneven terrain)",
    "Track Running (running tracks)",
    "Treadmill running",
    "Cross training",
    "Daily wear"
]

TOEBOX_OPTIONS = ["Wide", "Narrow", "Regular"]
FOOTBED_OPTIONS = ["Firm", "Soft", "Regular"]

WIDTH_COMPATIBILITY = {
    'narrow': {'exact': ['narrow'], 'compatible': ['medium']},
    'medium': {'exact': ['medium', 'regular'], 'compatible': []},
    'wide': {'exact': ['wide'], 'compatible': ['extra wide']},
    'extra wide': {'exact': ['extra wide'], 'compatible': []}
}

def score_usage(usage_str: str, user_preferences: List[str]) -> float:
    """
    Score product based on usage scenario matching

    Args:
        usage_str: Product's usage scenarios string
        user_preferences: List of user's preferred usage scenarios

    Returns:
        Usage matching score (float)
    """
    if pd.isna(usage_str) or not user_preferences:
        return 0.0

    try:
        # Extract top 3 usage scenarios with percentages
        scenes = []
        for entry in str(usage_str).split(','):
            if '(' in entry and ')' in entry:
                scene = entry.split('(')[0].strip().lower()
                scenes.append(scene)
                if len(scenes) >= 3:
                    break

        # Calculate score based on match position
        for rank, scene in enumerate(scenes, 1):
            for user_pref in user_preferences:
                if user_pref.lower() == scene:
                    if rank == 1:
                        return 15.0
                    elif rank == 2:
                        return 10.0
                    elif rank == 3:
                        return 5.0
    except Exception as e:
        print(f"Error scoring usage: {str(e)}")

    return 0.0

def score_brand(row: pd.Series, brand_preferences: Dict[str, Any]) -> float:
    """
    Score product based on brand/model preferences

    Args:
        row: Product data row
        brand_preferences: Dict of brand preferences
                          (format: {"Brand": {"models": ["model1", "model2"]}})

    Returns:
        Brand matching score (float)
    """
    if not brand_preferences:
        return 0.0

    try:
        vendor = str(row.get('vendor', '')).strip().lower()
        custom_model = str(row.get('custom.model', '')).strip().lower()
        full_name = str(row.get('full_product_name', '')).strip().lower()

        for brand, prefs in brand_preferences.items():
            brand_lower = brand.strip().lower()
            if brand_lower != vendor:
                continue

            if 'models' in prefs and prefs['models']:
                # Check for exact model matches
                for req_model in prefs['models']:
                    req_lower = req_model.strip().lower()
                    norm_custom = ' '.join(custom_model.replace('-', ' ').split())
                    norm_req = ' '.join(req_lower.replace('-', ' ').split())

                    if norm_req == norm_custom:
                        return 45.0  # Exact model match

                # Check for partial matches in full name
                for req_model in prefs['models']:
                    req_lower = req_model.strip().lower()
                    norm_full = ' '.join(full_name.replace('-', ' ').split())
                    if req_lower in norm_full:
                        return 35.0  # Partial model match

                return 20.0  # Brand match only

            return 20.0  # Brand match only
    except Exception as e:
        print(f"Error scoring brand: {str(e)}")

    return 0.0

def score_color(row: pd.Series, color_preferences: List[str]) -> float:
    """
    Score product based on color preferences

    Args:
        row: Product data row
        color_preferences: List of preferred colors

    Returns:
        Color matching score (float)
    """
    if not color_preferences or 'custom.color' not in row or pd.isna(row['custom.color']):
        return 0.0

    try:
        product_colors = [c.strip().lower() for c in str(row['custom.color']).split('/')]
        total_score = 0.0
        user_prefs = [p.lower() for p in color_preferences]

        for pref in user_prefs:
            for pos, color in enumerate(product_colors):
                # Exact match
                if color == pref:
                    base_score = max(7.5 - pos * 2.5, 1)
                    total_score += base_score
                    break
                # Partial match
                elif pref in color:
                    base_score = max(7.5 - pos * 2.5, 1) * 0.9
                    total_score += base_score
                    break

        return round(total_score, 2)
    except Exception as e:
        print(f"Error scoring color: {str(e)}")
        return 0.0

def score_size(row: pd.Series, target_size: float, toebox_pref: Optional[str]) -> float:
    """
    Score product based on size matching

    Args:
        row: Product data row
        target_size: User's target size (float)
        toebox_pref: User's toebox preference

    Returns:
        Size matching score (float)
    """
    if pd.isna(row.get('size_min')) or pd.isna(row.get('size_max')):
        return 0.0

    try:
        # Exact size match
        if not row['is_range'] and abs(row['size_min'] - target_size) < 0.01:
            return 35.0

        # Adjust scoring based on toebox preference
        if toebox_pref:
            toebox_pref = toebox_pref.lower()

            # Wide fit preference - prioritize larger sizes
            if toebox_pref == "wide":
                if not row['is_range'] and abs(row['size_min'] - (target_size + 0.5)) < 0.01:
                    return 25.0
                elif row['is_range'] and (target_size <= row['size_max'] <= target_size + 0.5):
                    return 20.0

            # Narrow fit preference - prioritize smaller sizes
            elif toebox_pref == "narrow":
                if not row['is_range'] and abs(row['size_min'] - (target_size - 0.5)) < 0.01:
                    return 25.0
                elif row['is_range'] and (target_size - 0.5 <= row['size_min'] <= target_size):
                    return 20.0

        # Default size matching (within ±0.5)
        elif not row['is_range'] and abs(row['size_min'] - target_size) == 0.5:
            return 20.0

    except Exception as e:
        print(f"Error scoring size: {str(e)}")

    return 0.0

def score_width(row: pd.Series, target_width: str, target_gender: str) -> float:
    """
    Score product based on width matching

    Args:
        row: Product data row
        target_width: User's target width
        target_gender: User's gender for width mapping

    Returns:
        Width matching score (float)
    """
    if not target_width or pd.isna(row.get('normalized_width')):
        return 0.0

    try:
        target_width_lower = map_width_to_category(target_width, target_gender).lower()
        product_width = str(row['normalized_width']).strip().lower()

        if target_width_lower in WIDTH_COMPATIBILITY:
            if product_width in WIDTH_COMPATIBILITY[target_width_lower]['exact']:
                return 10.0  # Exact width match
            elif product_width in WIDTH_COMPATIBILITY[target_width_lower]['compatible']:
                return 5.0  # Compatible width match
    except Exception as e:
        print(f"Error scoring width: {str(e)}")

    return 0.0

def score_toebox(row: pd.Series, toebox_preference: Optional[str]) -> float:
    """
    Score product based on toebox preference

    Args:
        row: Product data row
        toebox_preference: User's toebox preference

    Returns:
        Toebox matching score (float)
    """
    if not toebox_preference or 'toebox' not in row or pd.isna(row['toebox']):
        return 0.0

    try:
        return 10.0 if str(row['toebox']).lower() == toebox_preference.lower() else 0.0
    except Exception as e:
        print(f"Error scoring toebox: {str(e)}")
        return 0.0

def score_footbed(row: pd.Series, footbed_preference: Optional[str]) -> float:
    """
    Score product based on footbed preference

    Args:
        row: Product data row
        footbed_preference: User's footbed preference

    Returns:
        Footbed matching score (float)
    """
    if not footbed_preference or 'footbed' not in row or pd.isna(row['footbed']):
        return 0.0

    try:
        return 10.0 if str(row['footbed']).lower() == footbed_preference.lower() else 0.0
    except Exception as e:
        print(f"Error scoring footbed: {str(e)}")
        return 0.0

def map_width_to_category(width_label: str, gender_label: str) -> str:
    """
    Map width codes to standardized categories

    Args:
        width_label: Raw width label (e.g. "D", "2E")
        gender_label: Gender for width mapping context

    Returns:
        Standardized width category
    """
    men_map = {
        '2A': 'x-narrow', 'B': 'narrow', 'D': 'medium',
        '2E': 'wide', '4E': 'extra wide', '6E': 'xx-wide',
        'medium (regular)': 'medium', 'regular': 'medium',
        'wide': 'wide', 'extra wide': 'extra wide'
    }
    women_map = {
        '4A': 'x-narrow', '2A': 'narrow', 'B': 'medium',
        'D': 'wide', '2E': 'extra wide', '4E': 'xx-wide',
        'medium (regular)': 'medium', 'regular': 'medium',
        'wide': 'wide', 'extra wide': 'extra wide'
    }

    width_label = str(width_label).strip()
    gender_label = str(gender_label).lower()

    if 'women' in gender_label:
        return women_map.get(width_label.upper(), women_map.get(width_label.lower(), ''))
    elif 'men' in gender_label:
        return men_map.get(width_label.upper(), men_map.get(width_label.lower(), ''))
    return ''