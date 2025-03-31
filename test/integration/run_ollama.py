import requests
import json

# Ollama's API URL
url = f"{os.environ['OPENAI_ENDPOINT']}/v1/completions"

# The model you want to use
model = "qwen2.5:latest"

# Set up the request headers (use your API key if necessary)
headers = {
    "Content-Type": "application/json",
    # Replace 'your_ollama_api_key_here' with your actual Ollama API key
    "Authorization": "Bearer your_ollama_api_key_here"
}

# Define the prompt you want to send
data = {
    "model": model,
    "prompt": "Hello, how can I access Ollama API?"
}

# Make a POST request to Ollama's API
response = requests.post(url, headers=headers, json=data)

# Check if the response is successful and print the result
if response.status_code == 200:
    print(response.json())
else:
    print(f"Error: {response.status_code}, {response.text}")
