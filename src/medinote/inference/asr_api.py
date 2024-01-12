import argparse
import asyncio
import json
from typing import List
import logging
import os

from fastapi import Body, FastAPI, HTTPException, Path, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse

from fastapi.middleware.cors import CORSMiddleware
import torch
import uvicorn

from fastchat.serve.base_model_worker import BaseModelWorker

from fastchat.serve.model_worker import (
    logger,
    worker_id,
)
from fastchat.utils import get_context_length, is_partial_stop
import nemo.collections.asr as nemo_asr
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
    model_name="nvidia/parakeet-rnnt-1.1b" # "/mnt/models/parakeet-rnnt-1.1b"
)

app = FastAPI()


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

@app.get("/whisper/{audio_file_naae}")
async def whisper_get(
    audio_file_naae: str = Path(title="audio_file_naae"),
    ):
    try:
        # dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
        # sample = dataset[0]["audio"]
        # result = pipe(sample)
        transcript = pipe(f"/mnt/media/audio/{audio_file_naae}")
        return {"transcription": transcript["text"]}
    except Exception as e:
        logger.error(e)
        raise HTTPException(
            status_code=500,
            detail=f"Train was not successful due to error: -- {e.args[0]} --",
        )



@app.get("/transcribe/{audio_file_naae}")
async def transcribe_get(
    audio_file_naae: str = Path(title="audio_file_naae"),
    ):
    try:
        transcrition = asr_model.transcribe(paths2audio_files=[f"/mnt/media/audio/{audio_file_naae}"])
        return {"transcription": transcrition[0]}
    except Exception as e:
        logger.error(e)
        raise HTTPException(
            status_code=500,
            detail=f"Train was not successful due to error: -- {e.args[0]} --",
        )


@app.post("/transcribe")
async def gpt_perform_json(body: dict = Body(..., media_type="application/json")):
    try:
        paths2audio = body.get("paths2audio")
        if paths2audio is None:
            raise ValueError("audio_src is required")
        transcrition = asr_model.transcribe(paths2audio_files=[paths2audio])
        return {"transcription": transcrition[0]}
    except Exception as e:
        logger.error(e)
        raise HTTPException(
            status_code=500,
            detail=f"Train was not successful due to error: -- {e.args[0]} --",
        )

# @app.post("/uploadfile/")
# async def create_upload_file(file: UploadFile):
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
