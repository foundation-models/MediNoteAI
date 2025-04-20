## Setup Instructions

### 1. Create Supabase Project

1. Go to [Supabase](https://supabase.com) and create a new project
2. Note down your project URL and anon/public key

### 2. Create Database Table

Run the following SQL in your Supabase SQL editor:

```sql
-- Create the web_search table
CREATE TABLE web_search (
    id BIGSERIAL PRIMARY KEY,
    metadata JSONB,
    embedding vector(1536)
);

-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
```

### 3. Create User and Set Up RLS

```sql
-- Create a new user (replace with your desired email and password)
CREATE USER your_email@example.com WITH PASSWORD 'your_secure_password';

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON TABLE web_search TO your_email@example.com;

-- Enable Row Level Security
ALTER TABLE web_search ENABLE ROW LEVEL SECURITY;

-- Create RLS policy for your specific user
CREATE POLICY "Allow access for specific user" ON web_search
FOR ALL
TO authenticated
```

### 4. Create Vector Similarity Search Function

```sql
-- Create the function for vector similarity search
CREATE OR REPLACE FUNCTION match_web_search(
  query_embedding vector(1536),
  match_threshold float,
  match_count int
)
RETURNS TABLE (
  id bigint,
  metadata jsonb,
  similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    web_search.id,
    web_search.metadata,
    1 - (web_search.embedding <=> query_embedding) as similarity
  FROM web_search
  WHERE 1 - (web_search.embedding <=> query_embedding) > match_threshold
  ORDER BY web_search.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;
```

### 5. Environment Setup

1. Create a `.env` file based on `.env.example`:
```bash
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Code Files

### 1. supabase_demo.py
```python
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import numpy as np

# Load environment variables
load_dotenv()

# Initialize Supabase client
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Sample data
sample_data = {
    "metadata": {
        "title": "Sample Web Search Result",
        "url": "https://example.com",
        "description": "This is a sample web search result",
        "timestamp": "2024-03-20T12:00:00Z"
    },
    "embedding": np.random.rand(1536).tolist()  # Sample 1536-dimensional vector
}

# Insert data into Supabase
try:
    data = supabase.table('web_search').insert({
        "metadata": sample_data["metadata"],
        "embedding": sample_data["embedding"]
    }).execute()
    print(f"Successfully inserted data: {data}")
except Exception as e:
    print(f"Error inserting data: {e}")
```

### 2. get_data.py
```python
import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Initialize Supabase client
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

try:
    # Fetch all data from the web_search table
    response = supabase.table('web_search').select("*").execute()
    
    # Print the data
    print("Retrieved data:")
    for item in response.data:
        print("\nRecord:")
        print(f"ID: {item.get('id')}")
        print("Metadata:")
        for key, value in item['metadata'].items():
            print(f"  {key}: {value}")
        print(f"Embedding vector length: {len(item['embedding'])}")
        
except Exception as e:
    print(f"Error fetching data: {e}")
```

### 3. similarity_search.py
```python
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import numpy as np

# Load environment variables
load_dotenv()

# Initialize Supabase client
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Example query embedding
query_embedding = [0.92006755, 0.13812089, 0.07962512, ...]  # Your 1536-dimensional vector

try:
    # Perform vector similarity search
    response = supabase.rpc(
        'match_web_search',
        {
            'query_embedding': query_embedding,
            'match_threshold': 0.7,  # Similarity threshold (0 to 1)
            'match_count': 5  # Number of matches to return
        }
    ).execute()
    
    # Print the results
    print("\nSimilarity Search Results:")
    for item in response.data:
        print("\nMatch:")
        print(f"Similarity Score: {item.get('similarity')}")
        print("Metadata:")
        for key, value in item['metadata'].items():
            print(f"  {key}: {value}")
        
except Exception as e:
    print(f"Error performing similarity search: {e}")
```
