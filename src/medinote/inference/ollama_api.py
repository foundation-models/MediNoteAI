import requests
import numpy as np
import pandas as pd

def embed_batch(texts, config):
    # Define the API URL
    url = config.get('inference_url')

    # Prepare the payload with a list of prompts
    payload = {
        "model": "nomic-embed-text",
        "input": texts  # Pass all texts in a batch
    }

    # Send the request
    response = requests.post(url, json=payload)
    # Check if the request was successful
    if response.status_code == 200:
        result = response.json()
        embeddings = np.array(result['embeddings'])  # Assuming the response contains a list of embeddings
        return embeddings  # Return the embeddings as a NumPy array
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None  # If there was an error, return None

def row_infer(df, config):
    df = df.copy()

    # Collect all texts into a list
    all_texts = df['text'].tolist()

    # Get embeddings for the entire batch
    embeddings = embed_batch(all_texts, config)

    if embeddings is not None:
        # Assign the embeddings to the 'embedding' column in the DataFrame
        df['embedding'] = embeddings.tolist()  # Convert to list for compatibility with DataFrame
    else:
        # Handle error, fill with None or other suitable value
        df['embedding'] = [None] * len(df)

    return df  # Return the updated DataFrame with the 'embedding' column
