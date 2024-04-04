import argparse

import torch
from fastchat.constants import SERVER_ERROR_MSG, ErrorCode
from fastapi import Request
from sentence_transformers import SentenceTransformer

import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# model_path = '/mnt/ai-llm/models/mxbai-embed-large-v1'
# model_path = '/mnt/ai-llm/models/SFR-Embedding-Mistral'
model_path = '/mnt/ai-llm/models/UAE-Large-V1'


# model = SentenceTransformer(model_path)
from angle_emb import AnglE, Prompts
model = AnglE.from_pretrained(model_path, pooling_strategy='cls').cuda()
model.set_prompt(prompt=Prompts.C)

# import voyageai

# vo = voyageai.Client()

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
        embeddings = model.encode(new_prompt, to_numpy=True)
        # embeddings = model.encode(prompts, to_numpy=True)
        ret["embedding"] = embeddings.tolist()
        ret["token_num"] = token_counts        

        # result = vo.embed(prompts, model="voyage-2", input_type="document")
        # ret["embedding"] = result.embeddings
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
    parser.add_argument("--port", type=int, default=7777)
    args = parser.parse_args()
    app.title = "Embedding Encode API"
    app.description = "API for FineTuning"
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
