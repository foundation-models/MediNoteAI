import argparse

import torch
from fastchat.constants import SERVER_ERROR_MSG, ErrorCode
from fastapi import Request, FastAPI, File, UploadFile, Form
import requests

import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
import io
from fastapi.responses import JSONResponse

import torch
from transformers import AutoProcessor, VipLlavaForConditionalGeneration
from fastapi.responses import HTMLResponse
from pathlib import Path

model_path = "/mnt/models/vip-llava-7b-hf"
model = VipLlavaForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
    ).to(0)


app = FastAPI(
    title="Generative AI",
    description="Generative AI API"
)

# make this only for dev environment but for production list the allowed origins
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def default():
    """
    Returns a 200 to signal that the server is up.
    """
    return {"status": "ok"}


@app.get("/liveness")
async def liveness():
    """
    Returns a 200 to signal that the server is up.
    """
    return {"status": "live"}


@app.get("/readiness")
async def readiness():
    """
    Returns a 200 if the server is ready for prediction.
    """

    # Currently simply take the first tenant.
    # to decrease chances of loading a not needed model.
    return {"status": "ready"}

@app.get("/image-qa", response_class=HTMLResponse)
async def get_upload_form(request: Request):
    domain = request.url_for('get_upload_form')
    form_html = Path("/home/agent/workspace/MediNoteAI/src/medinote/inference/upload_form.html").read_text()
    # Replace the placeholder with the actual domain
    form_html = form_html.replace("http://localhost:8888", f'http://{str(domain).split("://")[1].split("/")[0]}')
    return HTMLResponse(content=form_html)

@app.post("/image-qa/")
async def upload_image(file: UploadFile = File(...), question: str = Form(...)):
    # Read the uploaded file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Convert PIL Image to the required format for model.generate here if necessary

    # Call your model's generate function
    result = await model_generate(image, question)

    return JSONResponse(content=result)


async def model_generate(image, question):
    try:
        prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: <image>\n{question}###Assistant:"
        ret = {"prompt": prompt}

        processor = AutoProcessor.from_pretrained(model_path)
        inputs = processor(prompt, image, return_tensors='pt').to(0, torch.float16)
        image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(image_file, stream=True).raw)

        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        response =processor.decode(output[0][2:], skip_special_tokens=True)
        ret["answer"] = response.split("###Assistant:")[1]
    except torch.cuda.OutOfMemoryError as e:
        ret = {
            "text": f"{SERVER_ERROR_MSG}\n\n({e})",
            "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
        }
    except (ValueError, RuntimeError) as e:
        ret = {
            "text": f"{SERVER_ERROR_MSG}\n\n({e})",
            "error_code": ErrorCode.INTERNAL_ERROR,
        }
    return ret            

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args()
    app.title = "Generate API"
    app.description = "API for Captioning"
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
