import pandas as pd
from typing import Dict, Any, List, Optional
import re

def generate_recommendation_reason(row: pd.Series) -> str:
    """
    Generate human-readable explanation for why a product was recommended

    Args:
        row: A single product row with all scoring information

    Returns:
        String explaining the recommendation reason
    """
    reasons = []

    # Usage scenario reasons
    if not pd.isna(row.get('usage')) and row.get('usage_score', 0) > 0:
        try:
            top_usages = [
                u.split('(')[0].strip()
                for u in str(row['usage']).split(',')[:2]
                if '(' in u
            ]
            if top_usages:
                reasons.append(f"Usage: {', '.join(top_usages)}")
        except Exception:
            pass

    # Width reasons
    if not pd.isna(row.get('normalized_width')) and row.get('width_score', 0) > 0:
        reasons.append(f"Width: {row['normalized_width']}")

    # Size reasons
    if not pd.isna(row.get('size_score')) and row['size_score'] > 0:
        if row.get('is_range'):
            reasons.append(f"Size range: {row['size_min']}-{row['size_max']}")
        else:
            reasons.append(f"Exact size: {row['size_min']}")

    # Brand/model reasons
    if not pd.isna(row.get('brand_score')) and row['brand_score'] > 0:
        brand_info = []
        if row.get('vendor'):
            brand_info.append(row['vendor'])
        if row.get('custom.model'):
            brand_info.append(row['custom.model'])
        if brand_info:
            reasons.append("Brand: " + " ".join(brand_info))

    # Color reasons
    if not pd.isna(row.get('custom.color')) and row.get('color_score', 0) > 0:
        colors = str(row['custom.color']).split('/')
        if colors:
            reasons.append(f"Color: {colors[0].strip()}")

    # Toebox reasons
    if not pd.isna(row.get('toebox')) and row.get('toebox_score', 0) > 0:
        reasons.append(f"Toebox: {row['toebox']}")

    # Footbed reasons
    if not pd.isna(row.get('footbed')) and row.get('footbed_score', 0) > 0:
        reasons.append(f"Footbed: {row['footbed']}")

    return " | ".join(reasons) if reasons else "General recommendation"

def extract_width_from_query(query: str, gender: str) -> str:
    """
    Extract standardized width category from search query

    Args:
        query: User search query string
        gender: Gender context for width interpretation

    Returns:
        Standardized width category or empty string if not found
    """
    possible_widths = ['4A', '2A', 'B', 'D', '2E', '4E', '6E']
    pattern = r'\b(?:' + '|'.join(possible_widths) + r')\b'
    match = re.search(pattern, query.upper())

    if match:
        width_code = match.group(0)
        return map_width_to_category(width_code, gender)
    return ''

def map_width_to_category(width_label: str, gender_label: str) -> str:
    """
    Map width codes to standardized categories based on gender

    Args:
        width_label: Raw width code (e.g. "D", "2E")
        gender_label: Gender context ("Men's" or "Women's")

    Returns:
        Standardized width category
    """
    men_map = {
        '2A': 'x-narrow', 'B': 'narrow', 'D': 'medium',
        '2E': 'wide', '4E': 'extra wide', '6E': 'xx-wide'
    }
    women_map = {
        '4A': 'x-narrow', '2A': 'narrow', 'B': 'medium',
        'D': 'wide', '2E': 'extra wide', '4E': 'xx-wide'
    }

    width_label = str(width_label).strip().upper()
    gender_label = str(gender_label).lower()

    if 'women' in gender_label:
        return women_map.get(width_label, '')
    elif 'men' in gender_label:
        return men_map.get(width_label, '')
    return ''

def normalize_gender_input(gender_input: str) -> str:
    """
    Normalize various gender inputs to standard format

    Args:
        gender_input: Raw gender input from user

    Returns:
        Standardized gender string ("Men's" or "Women's")
    """
    gender_input = str(gender_input).lower()
    if any(x in gender_input for x in ['men', 'male', 'man']):
        return "Men's"
    elif any(x in gender_input for x in ['women', 'female', 'woman']):
        return "Women's"
    return "Unisex"

def validate_size_input(size_input: str) -> bool:
    """
    Validate that a size input is in acceptable format

    Args:
        size_input: User-provided size string

    Returns:
        True if valid, False otherwise
    """
    try:
        if size_input.endswith('.'):
            size_input = size_input[:-1]
        float(size_input)
        return True
    except ValueError:
        return False

def parse_user_preferences(
    gender: str,
    size: str,
    width: Optional[str] = None,
    usage: Optional[List[str]] = None,
    toebox: Optional[str] = None,
    footbed: Optional[str] = None,
    brands: Optional[Dict] = None,
    colors: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Parse and validate all user preferences into standardized format

    Args:
        All raw user preference inputs

    Returns:
        Dictionary of cleaned and validated preferences
    """
    preferences = {
        'gender': normalize_gender_input(gender),
        'size': size,
        'is_valid': True,
        'validation_errors': []
    }

    # Validate size
    if not validate_size_input(size):
        preferences['is_valid'] = False
        preferences['validation_errors'].append("Invalid size format")

    # Process width
    if width:
        preferences['width'] = width.upper()

    # Process usage scenarios
    if usage:
        preferences['usage'] = [s.strip() for s in usage if s.strip()]

    # Process toebox preference
    if toebox and toebox.lower() in ['narrow', 'wide', 'regular']:
        preferences['toebox'] = toebox.lower().capitalize()

    # Process footbed preference
    if footbed and footbed.lower() in ['firm', 'soft', 'regular']:
        preferences['footbed'] = footbed.lower().capitalize()

    # Process brand preferences
    if brands:
        preferences['brands'] = {}
        for brand, models in brands.items():
            if brand.strip():
                clean_models = [m.strip() for m in models if m.strip()]
                preferences['brands'][brand.strip()] = {'models': clean_models}

    # Process color preferences
    if colors:
        preferences['colors'] = [c.strip() for c in colors if c.strip()]

    return preferences