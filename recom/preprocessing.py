import pandas as pd
import json
import re
from typing import Tuple, List, Dict, Any

def extract_color_from_name(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract color information from product names

    Args:
        df: DataFrame containing product data with 'product_name' column

    Returns:
        DataFrame with added 'color_from_name' column containing extracted colors
    """
    def split_colors(name: str) -> List[str]:
        """Helper function to split colors from product name"""
        try:
            if not isinstance(name, str):
                return []

            parts = name.split(',')
            if len(parts) >= 3:
                color_str = parts[1].strip()
                return [c.strip() for c in color_str.split('/')]
            return []
        except Exception:
            return []

    df = df.copy()
    df['color_from_name'] = df['product_name'].apply(split_colors)
    return df

def expand_options_columns(df: pd.DataFrame, options_col: str = 'options') -> Tuple[pd.DataFrame, List[str]]:
    """
    Expand JSON options column into separate columns

    Args:
        df: Input DataFrame
        options_col: Name of column containing options JSON

    Returns:
        Tuple of (expanded DataFrame, list of new column names)
    """
    def parse_options(option_str: Any) -> Dict[str, Any]:
        """Parse options JSON string into dictionary"""
        result = {}
        try:
            if pd.isna(option_str):
                return result

            if isinstance(option_str, str):
                if option_str.strip() == "":
                    return result
                option_dict = json.loads(option_str)
            elif isinstance(option_str, dict):
                option_dict = option_str
            else:
                return result

            for name, values in option_dict.items():
                if isinstance(values, list) and values:
                    result[name] = values[0]
        except Exception as e:
            print(f"JSON parse error: {option_str} -> {str(e)}")
        return result

    df = df.copy()
    parsed = df[options_col].apply(parse_options)
    options_df = pd.DataFrame(parsed.tolist(), index=df.index)

    # Drop overlapping columns if needed
    for col in options_df.columns:
        if col in df.columns:
            df = df.drop(columns=[col])

    df_expanded = pd.concat([df, options_df], axis=1)
    return df_expanded, list(options_df.columns)

def extract_department(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract gender/department information from product names

    Args:
        df: DataFrame containing product data with 'product_name' column

    Returns:
        DataFrame with added 'Department' column
    """
    def get_department(name: str) -> str:
        """Extract gender/department from product name"""
        if not isinstance(name, str):
            return 'Unknown'

        match = re.search(r"\b(Women's|Men's|Unisex|Kids')\b", name)
        return match.group(1) if match else 'Unknown'

    df = df.copy()
    df['Department'] = df['product_name'].apply(get_department)
    return df

def expand_metadata_columns(df: pd.DataFrame, metadata_col: str = 'metadata') -> Tuple[pd.DataFrame, List[str]]:
    """
    Expand metadata JSON column into separate columns

    Args:
        df: Input DataFrame
        metadata_col: Name of column containing metadata JSON

    Returns:
        Tuple of (expanded DataFrame, list of new column names)
    """
    selected_keys = [
        "custom.color",
        "custom.model",
        "google.gender",
        "my_fields.size",
        "my_fields.width",
        "custom.use_case"
    ]

    def extract_keys(meta_str: Any) -> Dict[str, Any]:
        """Extract selected keys from metadata"""
        try:
            if pd.isna(meta_str):
                return {k: None for k in selected_keys}

            if isinstance(meta_str, str):
                meta_dict = json.loads(meta_str)
            elif isinstance(meta_str, dict):
                meta_dict = meta_str
            else:
                return {k: None for k in selected_keys}

            return {k: meta_dict.get(k) for k in selected_keys}
        except Exception as e:
            print(f"Metadata parse error: {meta_str} -> {str(e)}")
            return {k: None for k in selected_keys}

    parsed = df[metadata_col].apply(extract_keys)
    meta_df = pd.DataFrame(parsed.tolist(), index=df.index)

    df_combined = pd.concat([df, meta_df], axis=1)
    return df_combined, list(meta_df.columns)

def build_product_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct full product names from vendor and model information

    Args:
        df: Input DataFrame containing vendor and model info

    Returns:
        DataFrame with added 'full_product_name' column
    """
    df = df.copy()
    df['model_clean'] = df['custom.model'].fillna('').str.strip()
    df['vendor_clean'] = df['vendor'].fillna('').str.strip()
    df['full_product_name'] = df['vendor_clean'] + ' ' + df['model_clean']
    df = df[df['full_product_name'].str.strip() != '']
    return df

def preprocess_product_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for product data

    Args:
        df: Raw product DataFrame

    Returns:
        Fully preprocessed DataFrame
    """
    # Step 1: Extract color from names
    df = extract_color_from_name(df)

    # Step 2: Expand options JSON
    df, _ = expand_options_columns(df)

    # Step 3: Extract department/gender
    df = extract_department(df)

    # Step 4: Expand metadata
    df, _ = expand_metadata_columns(df)

    # Step 5: Build full product names
    df = build_product_names(df)

    # Step 6: Clean column names
    df = df.rename(columns={
        'Size': 'size_from_options',
        'Color': 'color_from_options',
        'Width': 'width_from_options',
        'Model': 'model_from_options',
        'first_word': 'first_word_from_name',
        'Department': 'gender_from_name'
    })

    return df