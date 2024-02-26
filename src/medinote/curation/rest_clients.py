import requests
import json
import os
import logging
# Set up logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

inference_url = os.environ.get(
    'BASE_CURATION_URL', 'http://llm:8888/worker_generate')  # Default if not set


def generate_via_rest_client(text: str,
                             method: str = "post",
                             payload: dict = None,
                             headers: dict = None,
                             token: str = None):
    try:
        headers = headers or {
            "Content-Type": "application/json",
        }
        print(f"Fetching ....{inference_url}....{token}... {method}")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        payload = payload or {
            "echo": False,
            "stop": [
                "<|im_start|>"
            ],
            "prompt": f"<|im_start|>system\nYou are \"Hermes 2\", a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia.<|im_end|>\n<|im_start|>user\nCreate a very short, couple of words note for the following text: \n\n{text}<|im_end|>\n<|im_start|>assistant\n"
        }

        if method.lower() == "post":
            response = requests.post(
                url=inference_url, headers=headers, json=payload)
        else:
            url = f"{inference_url}/{text}"
            response = requests.get(url=url, headers=headers)
        print(response.json())
        return response.json()['text'] if 'text' in response.json() else json.dumps(response.json())
    except requests.RequestException as e:
        print(f"Error fetching URL {inference_url}: {e}")
        logger.error(f"Error fetching URL {inference_url}: {e}")
        return json.dumps({"error": f"Error fetching URL {inference_url}: {e}"})
    except Exception as e:
        print(f"Error on the response:\n \n from {inference_url}:\n {e}")
        logger.error(
            f"Error on the response:\n \n from {inference_url}:\n {e}")
        return json.dumps({"error": f"Error on the response:\n \n from {inference_url}:\n {e}"})
