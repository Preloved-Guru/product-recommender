import pandas as pd
from typing import List, Dict, Optional, Any
from .scoring import *
from .preprocessing import *
from .utils import generate_recommendation_reason

class ShoeRecommender:
    """
    A recommendation system for shoes that analyzes product features and user preferences
    to provide personalized recommendations.
    """

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the recommender system

        Args:
            data_path: Optional path to preprocessed CSV data file
        """
        self.df = None
        if data_path:
            self.load_from_csv(data_path)

    def load_data(self, df: pd.DataFrame) -> None:
        """
        Load data directly from a DataFrame

        Args:
            df: Pandas DataFrame containing product data
        """
        self.df = preprocess_product_data(df)
        print(f"Loaded data with {len(self.df)} products")

    def load_from_csv(self, filepath: str) -> None:
        """
        Load preprocessed data from CSV file

        Args:
            filepath: Path to CSV file
        """
        try:
            self.df = pd.read_csv(filepath)
            print(f"Loaded data from {filepath} with {len(self.df)} products")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def recommend(
        self,
        target_gender: str,
        target_size: str,
        target_width: Optional[str] = None,
        usage_preferences: Optional[List[str]] = None,
        toebox_preference: Optional[str] = None,
        footbed_preference: Optional[str] = None,
        brand_preferences: Optional[Dict[str, Any]] = None,
        color_preferences: Optional[List[str]] = None,
        top_k: int = 10
    ) -> pd.DataFrame:
        """
        Generate shoe recommendations based on user preferences

        Args:
            target_gender: Target gender ("Men's" or "Women's")
            target_size: Target shoe size (e.g. "10" or "9.5")
            target_width: Optional width preference (e.g. "D", "2E")
            usage_preferences: List of preferred usage scenarios
            toebox_preference: Preferred toebox type ("Narrow", "Wide", etc.)
            footbed_preference: Preferred footbed type ("Firm", "Soft", etc.)
            brand_preferences: Dictionary of brand preferences
            color_preferences: List of preferred colors
            top_k: Number of recommendations to return

        Returns:
            DataFrame containing top recommendations with scores
        """
        if self.df is None or self.df.empty:
            print("Error: No data loaded")
            return pd.DataFrame()

        try:
            # Convert target size to numeric value
            target_size_num = self._parse_size(target_size)

            # Create a working copy of the data
            working_df = self.df.copy()

            # Apply gender filter
            working_df = working_df[
                working_df['gender_from_name'].str.lower() == target_gender.lower()
            ]
            if working_df.empty:
                print("No products match the gender filter")
                return pd.DataFrame()

            # Parse and add size columns
            size_data = working_df['size'].apply(self._parse_size_string)
            working_df[['size_min', 'size_max', 'is_range']] = pd.DataFrame(
                size_data.tolist(),
                index=working_df.index,
                columns=['size_min', 'size_max', 'is_range']
            )

            # Apply size filter based on toebox preference
            working_df = self._filter_by_size(
                working_df,
                target_size_num,
                toebox_preference
            )
            if working_df.empty:
                print("No products match the size criteria")
                return pd.DataFrame()

            # Apply width filter if specified
            if target_width:
                working_df = self._filter_by_width(
                    working_df,
                    target_width,
                    target_gender
                )
                if working_df.empty:
                    print("No products match the width criteria")
                    return pd.DataFrame()

            # Calculate all scores
            working_df = self._calculate_scores(
                working_df,
                target_gender,
                target_size_num,
                target_width,
                usage_preferences,
                toebox_preference,
                footbed_preference,
                brand_preferences,
                color_preferences
            )

            # Sort and select top recommendations
            recommendations = self._generate_recommendations(working_df, top_k)

            return recommendations

        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            return pd.DataFrame()

    def _parse_size(self, size_str: str) -> float:
        """
        Parse size string into numeric value

        Args:
            size_str: Size string (e.g. "10" or "9.5")

        Returns:
            Numeric size value
        """
        try:
            if size_str.endswith('.'):  # Handle "10." format
                return float(size_str[:-1]) + 0.5
            return float(size_str)
        except ValueError:
            print(f"Invalid size format: {size_str}")
            raise

    def _parse_size_string(self, size_str: Any) -> tuple:
        """
        Parse product size string into (min, max, is_range)

        Args:
            size_str: Size string from product data

        Returns:
            Tuple of (min_size, max_size, is_range)
        """
        if pd.isna(size_str):
            return (None, None, False)

        size_str = str(size_str).strip()

        try:
            # Handle size ranges (e.g. "9.5-10.5")
            if '-' in size_str:
                low, high = map(str.strip, size_str.split('-'))
                low_val = self._parse_size(low)
                high_val = self._parse_size(high)
                return (low_val, high_val, True)

            # Handle single sizes
            val = self._parse_size(size_str)
            return (val, val, False)

        except Exception:
            return (None, None, False)

    def _filter_by_size(
        self,
        df: pd.DataFrame,
        target_size: float,
        toebox_preference: Optional[str]
    ) -> pd.DataFrame:
        """
        Filter products by size with toebox preference consideration

        Args:
            df: Input DataFrame
            target_size: Target size as float
            toebox_preference: User's toebox preference

        Returns:
            Filtered DataFrame
        """
        if toebox_preference and toebox_preference.lower() == "wide":
            # Wide fit - include target size and +0.5
            size_mask = (
                ((df['size_min'] <= target_size + 0.5) &
                (df['size_max'] >= target_size)) |
                (abs(df['size_min'] - target_size) < 0.01)
            )
        elif toebox_preference and toebox_preference.lower() == "narrow":
            # Narrow fit - include target size and -0.5
            size_mask = (
                ((df['size_min'] <= target_size) &
                (df['size_max'] >= target_size - 0.5)) |
                (abs(df['size_min'] - target_size) < 0.01)
            )
        else:
            # Default - include ±0.5 sizes
            size_mask = (
                (df['size_min'] <= target_size + 0.5) &
                (df['size_max'] >= target_size - 0.5)
            )

        return df[size_mask].copy()

    def _filter_by_width(
        self,
        df: pd.DataFrame,
        target_width: str,
        target_gender: str
    ) -> pd.DataFrame:
        """
        Filter products by width compatibility

        Args:
            df: Input DataFrame
            target_width: Target width code (e.g. "D")
            target_gender: Gender for width mapping

        Returns:
            Filtered DataFrame
        """
        target_width_lower = map_width_to_category(target_width, target_gender).lower()

        def is_width_compatible(product_width: str) -> bool:
            pw = str(product_width).lower()
            exact_match = pw in WIDTH_COMPATIBILITY.get(target_width_lower, {}).get('exact', [])
            compatible_match = pw in WIDTH_COMPATIBILITY.get(target_width_lower, {}).get('compatible', [])
            return exact_match or compatible_match

        return df[df['normalized_width'].apply(is_width_compatible)]

    def _calculate_scores(
        self,
        df: pd.DataFrame,
        target_gender: str,
        target_size: float,
        target_width: Optional[str],
        usage_preferences: Optional[List[str]],
        toebox_preference: Optional[str],
        footbed_preference: Optional[str],
        brand_preferences: Optional[Dict[str, Any]],
        color_preferences: Optional[List[str]]
    ) -> pd.DataFrame:
        """
        Calculate all scoring components

        Args:
            df: Input DataFrame
            All user preference parameters

        Returns:
            DataFrame with all score columns added
        """
        scored_df = df.copy()

        # Calculate individual scores
        scored_df['usage_score'] = scored_df['usage'].apply(
            lambda x: score_usage(x, usage_preferences or [])
        )
        scored_df['brand_score'] = scored_df.apply(
            lambda row: score_brand(row, brand_preferences or {}),
            axis=1
        )
        scored_df['color_score'] = scored_df.apply(
            lambda row: score_color(row, color_preferences or []),
            axis=1
        )
        scored_df['size_score'] = scored_df.apply(
            lambda row: score_size(row, target_size, toebox_preference),
            axis=1
        )
        scored_df['width_score'] = scored_df.apply(
            lambda row: score_width(row, target_width or '', target_gender),
            axis=1
        )
        scored_df['toebox_score'] = scored_df.apply(
            lambda row: score_toebox(row, toebox_preference),
            axis=1
        )
        scored_df['footbed_score'] = scored_df.apply(
            lambda row: score_footbed(row, footbed_preference),
            axis=1
        )

        # Calculate total score
        scored_df['total_score'] = (
            scored_df['usage_score'] +
            scored_df['brand_score'] +
            scored_df['color_score'] +
            scored_df['size_score'] +
            scored_df['width_score'] +
            scored_df['toebox_score'] +
            scored_df['footbed_score']
        )

        return scored_df

    def _generate_recommendations(
        self, 
        scored_df: pd.DataFrame, 
        top_k: int
    ) -> pd.DataFrame:
        """
        Generate final recommendations from scored products
        
        Args:
            scored_df: DataFrame with all scores calculated
            top_k: Number of recommendations to return
            
        Returns:
            DataFrame with top recommendations
        """
        if scored_df.empty:
            return pd.DataFrame()
        
        # Sort by brand_score first, then total_score
        sorted_df = scored_df.sort_values(
            by=['brand_score', 'total_score'], 
            ascending=[False, False]
        )
        
        # Select top recommendations
        recommendations = sorted_df.head(top_k).copy()
        
        # Add recommendation reasons
        recommendations['recommendation_reason'] = recommendations.apply(
            generate_recommendation_reason,
            axis=1
        )
        
        # Select output columns
        result_columns = [
            'product_id', 'product_name', 'size', 
            'my_fields.width', 'normalized_width', 'toebox', 'footbed',
            'usage', 'custom.color', 'vendor', 'total_score', 
            'size_score', 'size_min', 'size_max'
        ]
        
        return recommendations[[col for col in result_columns if col in recommendations.columns]]