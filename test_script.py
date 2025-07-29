import unittest
import pandas as pd
from recom.database import get_db_connection, fetch_products
from recom.recom import ShoeRecommender
from recom.preprocessing import preprocess_product_data
import os

class TestShoeRecommenderWithDB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize test environment with live database data"""
        # Establish database connection
        cls.connection = get_db_connection()

        # Fetch test data from database
        raw_data, _ = fetch_products(cls.connection)
        columns = [
            'product_id', 'product_name', 'partner_id', 'category', 'size',
            'color', 'quantity', 'options', 'vendor', 'metadata',
            'clickthrough_link', 'image_link'
        ]

        # Create DataFrame and preprocess
        cls.test_df = pd.DataFrame(raw_data, columns=columns)
        cls.test_df = preprocess_product_data(cls.test_df)

        # Initialize recommender
        cls.recommender = ShoeRecommender()
        cls.recommender.load_data(cls.test_df)

        print(f"Loaded {len(cls.test_df)} preprocessed products from database")

    @classmethod
    def tearDownClass(cls):
        """Clean up resources"""
        if hasattr(cls, 'connection') and cls.connection:
            cls.connection.close()

    def test_database_connection(self):
        """Test database connection is working"""
        self.assertIsNotNone(self.connection)
        self.assertFalse(self.connection.closed)

    def test_data_loading(self):
        """Test data was properly loaded and preprocessed"""
        self.assertIsInstance(self.test_df, pd.DataFrame)
        self.assertGreater(len(self.test_df), 0)
        self.assertIn('full_product_name', self.test_df.columns)

    def test_basic_recommendation(self):
        """Test basic recommendation scenario"""
        result = self.recommender.recommend(
            target_gender="Men's",
            target_size="10",
            target_width="D",
            top_k=5
        )

        self.assertIsInstance(result, pd.DataFrame)
        if not result.empty:
            print("\nSample recommendation:")
            print(result[['product_name', 'size', 'total_score']].head())

    def test_brand_filter(self):
        """Test recommendation with brand filter"""
        result = self.recommender.recommend(
            target_gender="Women's",
            target_size="8",
            target_width="B",
            brand_preferences={"Nike": {"models": []}},
            top_k=3
        )

        if not result.empty:
            self.assertTrue(any('Nike' in str(v) for v in result['vendor'].values))
            print("\nBrand filtered recommendations:")
            print(result[['product_name', 'vendor']])

    def test_color_preference(self):
        """Test color preference filtering"""
        result = self.recommender.recommend(
            target_gender="Men's",
            target_size="10",
            color_preferences=["Black", "Blue"],
            top_k=3
        )

        if not result.empty:
            print("\nColor filtered recommendations:")
            print(result[['product_name', 'custom.color']])

    def test_usage_scenario(self):
        """Test usage scenario scoring"""
        result = self.recommender.recommend(
            target_gender="Women's",
            target_size="7",
            usage_preferences=["Running", "Trail"],
            top_k=3
        )

        if not result.empty:
            print("\nUsage scenario recommendations:")
            print(result[['product_name', 'usage']])

def execute_test_cases_from_db():
    """Execute all test cases using live database data"""
    # Initialize with DB connection
    conn = get_db_connection()
    raw_data, _ = fetch_products(conn)
    columns = [
        'product_id', 'product_name', 'partner_id', 'category', 'size',
        'color', 'quantity', 'options', 'vendor', 'metadata',
        'clickthrough_link', 'image_link'
    ]

    test_df = pd.DataFrame(raw_data, columns=columns)
    test_df = preprocess_product_data(test_df)
    recommender = ShoeRecommender()
    recommender.load_data(test_df)

    # Define test cases (similar to what you provided)
    test_cases = [
        {
            "name": "Case 1 - Men's Asics Blue/White",
            "params": {
                "target_gender": "Men's",
                "target_size": "10",
                "target_width": "D",
                "brand_preferences": {"Asics": {"models": ["Gel-Kayano"]}},
                "color_preferences": ["Blue", "White"],
                "top_k": 5
            }
        },
        # Add other test cases...
    ]

    # Execute and collect results
    all_results = []
    for case in test_cases:
        print(f"\nExecuting: {case['name']}")
        result = recommender.recommend(**case['params'])

        if not result.empty:
            result['test_case'] = case['name']
            all_results.append(result)
            print(f"Found {len(result)} recommendations")
        else:
            print("No recommendations found")

    # Save consolidated results
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        output_path = "database_test_results.csv"
        final_df.to_csv(output_path, index=False)
        print(f"\nAll results saved to: {output_path}")
    else:
        print("\nNo recommendations generated from any test case")

    conn.close()

if __name__ == '__main__':
    # Run unit tests
    print("Running unit tests...")
    unittest.main(exit=False)

    # Execute full test cases with database
    print("\nExecuting full test cases with database...")
    execute_test_cases_from_db()