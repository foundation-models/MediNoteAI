from openai import AzureOpenAI
import requests
import json
import os
import logging

from medinote import setup_logging

logger = setup_logging()


def generate_via_rest_client(
    text: str = None,
    inference_url: str = None,
    method: str = "post",
    payload: dict = None,
    headers: dict = None,
    token: str = None,
):
    try:
        inference_url = (
            inference_url
            or os.getenv("inference_url", inference_url)
            or "http://llm:8888/worker_generate"
        )
        headers = headers or {
            "Content-Type": "application/json",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        payload = payload or {
            "echo": False,
            "stop": ["<|im_start|>"],
            "prompt": f'<|im_start|>system\nYou are "Hermes 2", a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia.<|im_end|>\n<|im_start|>user\nCreate a very short, couple of words note for the following text: \n\n{text}<|im_end|>\n<|im_start|>assistant\n',
        }
        payload = (
            json.loads(payload, strict=False) if isinstance(payload, str) else payload
        )
        if method.lower() == "post":
            response = requests.post(url=inference_url, headers=headers, json=payload)
        else:
            url = f"{inference_url}/{text}"
            response = requests.get(url=url, headers=headers)
        result = response.json()
        if "embedding" in result:
            embeddings = []
            for value in result["embedding"]:
                embeddings.append(value)
            return embeddings
        else:
            return result["text"] if "text" in result else json.dumps(result)
    except requests.RequestException as e:
        logger.error(f"Error fetching URL {inference_url}: {e}")
        return json.dumps({"error": f"Error fetching URL {inference_url}: {e}"})
    except Exception as e:
        logger.error(f"Error on the response:\n \n from {inference_url}:\n {e}")
        return json.dumps(
            {"error": f"Error on the response:\n \n from {inference_url}:\n {e}"}
        )


def call_openai(prompt: str, instruction: str = None):
    # model = "gpt-3.5-turbo"
    # template_name = "chatgpt"
    # conv = get_conv_template(template_name)
    # conv.set_system_message(instruction)
    # conv.append_message(conv.roles[0], prompt)
    # conv.append_message(conv.roles[1], None)

    # response = chat_completion_openai_azure(model, conv, temperature=0, max_tokens=256)
    # return response.json()

    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_OPENAI_VERSION"),
    )
    instruction = (
        instruction or "You are an AI assistant that helps people find information"
    )

    message_text = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model="gpt-4",  # model = "deployment_name"
        messages=message_text,
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
    )
    output = response.choices[0].message.content

    return output
