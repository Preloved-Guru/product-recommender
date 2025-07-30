import psycopg2
from psycopg2 import sql
import pandas as pd
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
import os

# Database Connection
load_dotenv()

def connect_to_db():
    """Establish database connection"""
    connection = psycopg2.connect(
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        dbname=os.getenv('DB_NAME'),
        sslmode='require'
    )
    
    cursor = connection.cursor()
    print("Database connection successful!")

    
    # Set schema
    schema_name = "wishlist_data"
    cursor.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema_name)))
    
    # Current schema
    cursor.execute("SELECT current_schema()")
    current_schema = cursor.fetchone()[0]
    print(f"Current schema: {current_schema}")
    
    return connection, cursor

# Data Loading Function
def load_existing_products(cursor) -> pd.DataFrame:
    """Load all existing product data from products_results_combined table"""
    query = sql.SQL("SELECT * FROM {}").format(sql.Identifier("products_results_combined"))
    cursor.execute(query)
    
    columns = [desc[0] for desc in cursor.description]
    data = cursor.fetchall()
    
    if not data:
        print("Warning: No data found in products_results_combined table")
        return pd.DataFrame(columns=columns)
    
    return pd.DataFrame(data, columns=columns)

# Core Recommendation System
def enhanced_recommend_existing_products(
    cursor,
    target_gender: str,
    target_size: str,
    target_width: Optional[str] = None,
    usage_preferences: Optional[List[str]] = None,
    toebox_preference: Optional[str] = None,
    footbed_preference: Optional[str] = None,
    brand_preferences: Optional[Dict] = None,
    color_preferences: Optional[List[str]] = None,
    top_k: int = 10
) -> pd.DataFrame:
    """
"Recommendation system based on existing products in database"

"Parameters:"

"cursor: Database cursor"

"target_gender: Target gender ('Men's'/'Women's')"

"target_size: User size (e.g. '9.5')"

"target_width: Shoe width preference (e.g. 'medium')"

"usage_preferences: List of usage scenario preferences"

"toebox_preference: Toebox type preference ('narrow'/'wide')"

"footbed_preference: Footbed preference"

"brand_preferences: Brand preferences {brand: {'models': [model_names]}}"

"color_preferences: Color preference list (e.g. ['Black', 'Blue'])"

"top_k: Number of recommendations to return"

"Returns: DataFrame containing recommendation results"
    """
    # Load existing products
    merged_df = load_existing_products(cursor)
    
    if merged_df.empty:
        print("Error: No products found in the database")
        return pd.DataFrame()
    
    usage_preferences = usage_preferences or []
    brand_preferences = brand_preferences or {}
    color_preferences = color_preferences or []
    
    print("\n=== Information ===")
    print(f"columns: {merged_df.columns.tolist()}")
    print(f"columns count: {len(merged_df)}")
    
    # 1. Gender Filter
    if 'gender_from_name' not in merged_df.columns:
        print("Error: Missing gender_from_name column")
        return pd.DataFrame()
    
    merged_df = merged_df[merged_df['gender_from_name'].str.lower() == target_gender.lower()]
    print(f"Row count after gender filter:  {len(merged_df)}")
    
    if merged_df.empty:
        print("Warning: No data remaining after gender filter")
        return pd.DataFrame()
    
    # 2. Size Processing System
    def parse_size(size_str):
        """Parse size range, return (min, max, is_range)"""
        if pd.isna(size_str):
            return (None, None, False)
        
        size_str = str(size_str).strip()

        try:
            # Process size range (like "9.5-10.5")
            if '-' in size_str:
                low, high = map(str.strip, size_str.split('-'))
                # low value
                if low.endswith('.'):  
                    low_val = float(low[:-1]) + 0.5
                else:
                    low_val = float(low)
                # high value
                if high.endswith('.'):  
                    high_val = float(high[:-1]) + 0.5
                else:
                    high_val = float(high)
                return (low_val, high_val, True)
            
            # process half size (like "10.5" 或 "10.")
            else:
                if size_str.endswith('.'):
                    val = float(size_str[:-1]) + 0.5
                elif size_str.isdigit():
                    val = float(size_str)
                else:
                    val = float(size_str)
                return (val, val, False)
                
        except Exception as e:
            print(f"Size processing error: {size_str}, Error: {str(e)}")
            return (None, None, False)
    

    size_data = merged_df['my_fields.size'].apply(parse_size)
    merged_df[['size_min', 'size_max', 'is_range']] = pd.DataFrame(
        size_data.tolist(), 
        index=merged_df.index,
        columns=['size_min', 'size_max', 'is_range']
    )
    
    
    try:
        
        if '.' in target_size:
            target_size_num = float(target_size[:-1]) + 0.5 if target_size.endswith('.') else float(target_size)
        else:
            target_size_num = float(target_size)
        
        
        if toebox_preference and toebox_preference.lower() == "wide":
            size_mask = (
                ((merged_df['size_min'] <= target_size_num + 0.5) & 
                (merged_df['size_max'] >= target_size_num)) |
                (abs(merged_df['size_min'] - target_size_num) < 0.01))
        elif toebox_preference and toebox_preference.lower() == "narrow":
            size_mask = (
                ((merged_df['size_min'] <= target_size_num) & 
                (merged_df['size_max'] >= target_size_num - 0.5)) |
                (abs(merged_df['size_min'] - target_size_num) < 0.01))
        else:
            size_mask = (
                (merged_df['size_min'] <= target_size_num + 0.5) & 
                (merged_df['size_max'] >= target_size_num - 0.5))
        
        merged_df = merged_df[size_mask].copy()
        
    except Exception as e:
        print(f"Size processing error: {str(e)}")
        return pd.DataFrame()
    
    # 3. Width Processing System
    def map_width_to_category(width_label, gender_label):
        """Map width code to category"""
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
    
    if 'my_fields.width' in merged_df.columns:
        merged_df['normalized_width'] = merged_df.apply(
            lambda row: map_width_to_category(
                row['my_fields.width'],
                row['gender_from_name']
            ),
            axis=1
        )
    else:
        merged_df['normalized_width'] = ''
        print("Warning: No my_fields.width column in data, created empty column")
    
    # Width filter
    if target_width:
        width_compatibility = {
            'narrow': {'exact': ['narrow'], 'compatible': ['medium']},
            'medium': {'exact': ['medium', 'regular'], 'compatible': []},
            'wide': {'exact': ['wide'], 'compatible': ['extra wide']},
            'extra wide': {'exact': ['extra wide'], 'compatible': []}
        }
        
        target_width_lower = map_width_to_category(target_width, target_gender).lower()
        
        def is_width_compatible(product_width):
            pw = str(product_width).lower()
            exact_match = pw in width_compatibility.get(target_width_lower, {}).get('exact', [])
            compatible_match = pw in width_compatibility.get(target_width_lower, {}).get('compatible', [])
            return exact_match or compatible_match
        
        merged_df = merged_df[merged_df['normalized_width'].apply(is_width_compatible)]
        print(f"Row count after width filter: {len(merged_df)}")
    
    # 4. Usage Scenario Scoring
    if 'usage' in merged_df.columns:
        def score_usage(usage_str: str, user_preferences: List[str]) -> float:
            if pd.isna(usage_str) or not user_preferences:
                return 0
            
            scenes = []
            for entry in str(usage_str).split(','):
                if '(' in entry and ')' in entry:
                    scene = entry.split('(')[0].strip().lower()
                    scenes.append(scene)
                    if len(scenes) >= 3:
                        break

            for rank, scene in enumerate(scenes, 1):
                for user_pref in user_preferences:
                    if user_pref.lower() == scene:
                        if rank == 1:
                            return 15
                        elif rank == 2:
                            return 10
                        elif rank == 3:
                            return 5
            return 0
        
        merged_df['usage_score'] = merged_df['usage'].apply(lambda x: score_usage(x, usage_preferences))
        print(f"Usage scenario scoring complete, highest score: {merged_df['usage_score'].max()}")
    else:
        merged_df['usage_score'] = 0
        print("Warning: No usage column in data, skipped usage scenario scoring")
    
    # 5. Brand Preference Scoring
    def score_brand(row) -> float:
        if not brand_preferences:
            return 0
        
        vendor = str(row.get('vender_clean', '')).strip().lower()
        model_clean = str(row.get('model_clean', '')).strip().lower()
        full_name = str(row.get('full_product_name', '')).strip().lower()
        
        for brand, prefs in brand_preferences.items():
            brand_lower = brand.strip().lower()
            if brand_lower != vendor:
                continue  
            
            if 'models' in prefs and prefs['models']:
                for req_model in prefs['models']:
                    req_lower = req_model.strip().lower()
                    norm_model = ' '.join(model_clean.replace('-', ' ').split())
                    norm_req = ' '.join(req_lower.replace('-', ' ').split())
                    
                    if norm_req == norm_model:
                        return 45
                
                    norm_full = ' '.join(full_name.replace('-', ' ').split())
                    if norm_req in norm_full:
                        return 35
            
                return 20
            
            return 20
        
        return 0
    
    merged_df['brand_score'] = merged_df.apply(score_brand, axis=1)
    print(f"Brand score range: {merged_df['brand_score'].min()} - {merged_df['brand_score'].max()}")
    
    # 6. Color Preference Scoring
    def score_color(row) -> float:
        if not color_preferences or 'custom.color' not in row or pd.isna(row['custom.color']):
            return 0
        
        product_colors = [c.strip().lower() for c in str(row['custom.color']).split('/')]
        total_score = 0.0
        user_prefs = [p.lower() for p in color_preferences]

        for pref in user_prefs:
            for pos, color in enumerate(product_colors):
                if color == pref:
                    base_score = max(7.5 - pos * 2.5, 1)
                    total_score += base_score
                    break
                elif pref in color:
                    base_score = max(7.5 - pos * 2.5, 1) * 0.9
                    total_score += base_score
                    break
        
        return round(total_score, 2)
    
    if 'custom.color' not in merged_df.columns:
        merged_df['custom.color'] = ''
        merged_df['color_score'] = 0
    else:
        merged_df['color_score'] = merged_df.apply(score_color, axis=1)
    
    # 7. Size scoring
    def score_size(row, target_size: float, toebox_pref: str) -> float:
        if pd.isna(row['size_min']) or pd.isna(row['size_max']):
            return 0
        
        if not row['is_range'] and abs(row['size_min'] - target_size) < 0.01:
            return 35
        
        if toebox_pref and toebox_pref.lower() == "wide":
            if not row['is_range'] and abs(row['size_min'] - (target_size + 0.5)) < 0.01:
                return 25
            elif row['is_range'] and (target_size <= row['size_max'] <= target_size + 0.5):
                return 20
        
        elif toebox_pref and toebox_pref.lower() == "narrow":
            if not row['is_range'] and abs(row['size_min'] - (target_size - 0.5)) < 0.01:
                return 25
            elif row['is_range'] and (target_size - 0.5 <= row['size_min'] <= target_size):
                return 20
        
        elif not row['is_range'] and abs(row['size_min'] - target_size) == 0.5:
            return 20
        
        return 0

    if not merged_df.empty:
        merged_df['size_score'] = merged_df.apply(
            lambda row: score_size(row, target_size_num, toebox_preference),
            axis=1
        )
    else:
        return pd.DataFrame()
    
    print(f"Size score range: {merged_df['size_score'].min()} - {merged_df['size_score'].max()}")
    
    # 8. Width Scoring
    def score_width(row) -> float:
        if not target_width or pd.isna(row.get('normalized_width')):
            return 0
            
        target_width_lower = map_width_to_category(target_width, target_gender).lower()
        product_width = str(row['normalized_width']).strip().lower()
        
        if target_width_lower in width_compatibility:
            if product_width in width_compatibility[target_width_lower]['exact']:
                return 10
            elif product_width in width_compatibility[target_width_lower]['compatible']:
                return 5
        
        return 0
    
    merged_df['width_score'] = merged_df.apply(score_width, axis=1)
    print(f"Width score range: {merged_df['width_score'].min()} - {merged_df['width_score'].max()}")
    
    # 9. toebox Scoring
    def score_toebox(row) -> float:
        if not toebox_preference or 'toebox' not in row or pd.isna(row['toebox']):
            return 0
        return 10 if str(row['toebox']).lower() == toebox_preference.lower() else 0
    
    # 10. footbed Scoring
    def score_footbed(row) -> float:
        if not footbed_preference or 'footbed' not in row or pd.isna(row['footbed']):
            return 0
        return 10 if str(row['footbed']).lower() == footbed_preference.lower() else 0
    
    # 11. Comprehensive Scoring
    merged_df['total_score'] = (
        merged_df['usage_score'] +
        merged_df['brand_score'] +
        merged_df['color_score'] +
        merged_df['size_score'] +
        merged_df['width_score'] +
        merged_df.apply(score_toebox, axis=1) +
        merged_df.apply(score_footbed, axis=1)
    )
    print(f"Total score range: {merged_df['total_score'].min()} - {merged_df['total_score'].max()}")

    # 12. Recommendation Results
    if len(merged_df) == 0:
        print("Warning: No data remaining after all filters")
        return pd.DataFrame()
    
    recommended = merged_df.sort_values(
        by=['brand_score', 'total_score'], 
        ascending=[False, False]
    ).head(top_k)
    
    
    # Return results
    result_columns = [
        'product_id', 'full_product_name', 'my_fields.size', 
        'my_fields.width', 'normalized_width', 'toebox', 'footbed',
        'usage', 'custom.color', 'vender_clean', 'total_score'
    ]
    return recommended[[col for col in result_columns if col in recommended.columns]]

# Test function
def test_existing_products_recommendation():
    """Test recommendation for existing products in database"""
    
    connection, cursor = connect_to_db()
    
    try:
       
        recommendations = enhanced_recommend_existing_products(
            cursor=cursor,
            target_gender="Women's",
            target_size="7",
            target_width="wide",
            usage_preferences=["Trail Running", "Daily Wear"],
            toebox_preference="Narrow",
            footbed_preference="Soft",
            brand_preferences={"Hoka": {"models": []}},
            color_preferences=["pink", "white", "orange"],
            top_k=10
        )
        
       
        print("\nFinal recommendation results:")
        if not recommendations.empty:
            print(recommendations.to_markdown(index=False))
        else:
            print("No matching products found")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    finally:
        cursor.close()
        connection.close()
        print("Database connection closed")

# Run test
if __name__ == "__main__":
    test_existing_products_recommendation()