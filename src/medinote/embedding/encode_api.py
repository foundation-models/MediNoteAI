import argparse

import torch
from fastchat.constants import SERVER_ERROR_MSG, ErrorCode
from fastapi import Request

import uvicorn
from opencopilot.oss_llm.entities import TokenizeResponse, TokenizeRequest

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from angle_emb import AnglE, Prompts
model_path = '/mnt/ai-llm/models/UAE-Large-V1'
model = AnglE.from_pretrained(model_path, pooling_strategy='cls').cuda()
model.set_prompt(prompt=Prompts.C)


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



@app.post("/worker_get_embeddings")
async def encode(request: Request):
    try:
        ret = {"embedding": [], "token_num": 0}
        params = await request.json()
        prompts = params.pop("input")
        new_prompt = []
        token_counts = []
        for prompt in prompts:
            new_prompt.append({'text': prompt})
            input_ids = model.tokenizer(prompt).input_ids
            token_counts.append(len(input_ids))
        vec = model.encode(new_prompt, to_numpy=True)
        ret["embedding"] = vec.tolist()
        ret["token_num"] = token_counts        
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

@app.post("/tokenize", response_model=TokenizeResponse)
async def tokenize(request: TokenizeRequest):
    return TokenizeResponse(tokens=model.tokenizer(request.text)['input_ids'])

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7777)
    args = parser.parse_args()
    app.title = "Embedding Encode API"
    app.description = "API for FineTuning"
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
