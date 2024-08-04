import asyncio
from uvicorn import Server, Config
from yolofastapi.utils.utility import process_image
from fastapi import (
    UploadFile,
    File,
)
from starlette.middleware.cors import CORSMiddleware
import os

from yolofastapi.routers import yolo

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@router.post("/process_images/")
async def process_images(
    files: List[UploadFile] = File(...),
    device: int = 0,
    conf: float = 0.6,
    imgsz: int = 1024,
):
    tasks = [process_image(file, device, conf, imgsz) for file in files]
    results = await asyncio.gather(*tasks)
    return {file.filename: result for file, result in zip(files, results)}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/liveness")
async def liveness_check():
    # try to load the model
    try:
        return {"status": "live"}
    except:
        return {"status": "not live"}, 500
    
app.include_router(yolo.router)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 80))
    server = Server(Config(app, host="0.0.0.0", port=port, lifespan="on"))
    server.run()