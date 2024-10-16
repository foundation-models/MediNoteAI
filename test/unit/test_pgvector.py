import os
from dotenv import load_dotenv
# pip install psycopg2-binary pgvector
import psycopg2
from pgvector.psycopg2 import register_vector

# Load environment variables from .env file
load_dotenv()

# Database connection details from environment variables
db_config = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
}

# Connect to the database
conn = psycopg2.connect(**db_config)
register_vector(conn)

# Create a cursor object
cur = conn.cursor()

try:
    cur.execute("""
        DROP TABLE items
    """)
except:
    pass

# Create a table with a vector column
cur.execute("""
    CREATE TABLE IF NOT EXISTS items (
        id SERIAL PRIMARY KEY,
        embedding VECTOR(3)
    )
""")
conn.commit()

# Insert vectors into the table
vectors = [
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9]
]

for vector in vectors:
    cur.execute("INSERT INTO items (embedding) VALUES (%s)", (vector,))
conn.commit()

# Query for the most similar vectors
query_vector = [0.11, 0.22, 0.33]
cur.execute(f"""
    SELECT id, embedding, embedding <-> '{query_vector}' AS distance
    FROM items
    ORDER BY distance
    LIMIT 5
""")
results = cur.fetchall()

# Display results
for row in results:
    print(f"ID: {row[0]}, Embedding: {row[1]}, Distance: {row[2]}")

# Close the cursor and connection
cur.close()
conn.close()
