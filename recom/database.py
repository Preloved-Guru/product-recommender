import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
import os

load_dotenv()

def get_db_connection():
    """Establish database connection"""
    connection = psycopg2.connect(
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        sslmode='require'
    )
    return connection

def fetch_products(connection, partner_id=306, category='Apparel & Accessories > Shoes', min_quantity=1):
    """Fetch product data from database"""
    cursor = connection.cursor()

    # Set schema
    schema_name = "wishlist_data"
    cursor.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema_name)))

    # Fetch data
    query = sql.SQL("""
        SELECT product_id, product_name, partner_id, category, size, color, quantity,
               options, vendor, metadata, clickthrough_link, image_link
        FROM products
        WHERE partner_id = %s
          AND category = %s
          AND quantity >= %s
    """)

    cursor.execute(query, (partner_id, category, min_quantity))
    columns = [desc[0] for desc in cursor.description]
    data = cursor.fetchall()

    return data, columns