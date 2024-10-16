import argparse

import torch
from fastchat.constants import SERVER_ERROR_MSG, ErrorCode
from fastapi import Request, FastAPI, File, UploadFile, Form


import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
import io
from fastapi.responses import JSONResponse

import torch
from fastapi.responses import HTMLResponse
from pathlib import Path
from ultralytics import YOLO

model_path = "/home/agent/workspace/models/YOLOv8/yolov8n.pt"
model = YOLO("/home/agent/workspace/models/YOLOv8/yolov8n.pt").to(0)


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

@app.get("/image-detect", response_class=HTMLResponse)
async def get_upload_form(request: Request):
    domain = request.url_for('get_upload_form')
    form_html = Path("/home/agent/workspace/MediNoteAI/src/medinote/inference/upload_form.html").read_text()
    # Replace the placeholder with the actual domain
    form_html = form_html.replace("http://localhost:8888", f'http://{str(domain).split("://")[1].split("/")[0]}')
    return HTMLResponse(content=form_html)

@app.post("/image-detect/")
async def upload_image(file: UploadFile = File(...), question: str = Form(...)):
    # Read the uploaded file
    image_file = await file.read()
    result = await model_generate(image_file)

    return JSONResponse(content=result)


async def model_generate(image_file: str):
    try:
        prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: <image>\n{question}###Assistant:"
        ret = {"prompt": prompt}



        output = model(image_file)
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
