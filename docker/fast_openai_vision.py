from http import HTTPStatus
import os

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, JSONResponse

from pydantic import BaseModel

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


app = FastAPI()


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_path = os.getenv("MODEL_PATH", "/mnt/models/Florence-2-large")

pretrained_model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch_dtype, trust_remote_code=True
).to(device)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)


class ImageRequest(BaseModel):
    model: str
    messages: list
    max_tokens: int


def extract_prompt_and_url(payload):
    try:
        prompt = None
        url = None
        for message in payload["messages"]:
            for content in message["content"]:
                if content["type"] == "text":
                    prompt = content["text"]
                elif content["type"] == "image_url":
                    url = content["image_url"]["url"]
        return prompt, url
    except KeyError as e:
        raise ValueError(f"Missing expected key in payload: {e}")


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ImageRequest):
    payload = request.dict()
    try:
        prompt, url = extract_prompt_and_url(payload)
        if prompt is None or url is None:
            raise ValueError("Missing prompt or URL in payload.")

        image = Image.open(requests.get(url, stream=True).raw)

        # Assuming 'processor' and 'device' are defined elsewhere in your code
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(
            device, torch_dtype
        )

        generated_ids = pretrained_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )

        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed_answer = processor.post_process_generation(
            generated_text, task=prompt, image_size=(image.width, image.height)
        )

        return JSONResponse(
            content={"parsed_answer": parsed_answer}, status_code=HTTPStatus.OK
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
