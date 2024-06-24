import asyncio
from typing import Any, Set

import fastapi
import uvicorn
from fastapi import HTTPException, Request


from fastapi import Request
from transformers import AutoTokenizer, AutoModel


TIMEOUT_KEEP_ALIVE = 5  # seconds


app = fastapi.FastAPI()

# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-7B-instruct", trust_remote_code=True)
# # In case you want to reduce the maximum length:
# model.max_seq_length = 8192

# model = AutoModel.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True)
model = AutoModel.from_pretrained('/mnt/ai-nfs/models/NV-Embed-v1', trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True)

# get the embeddings
max_length = 4096

@app.post("/v1/embeddings")
async def create_embedding(request: Any, raw_request: Request):
    try:
        ret = {"embedding": [], "token_num": 0}
        prompt = request.input
        token_counts = []
        input_ids = model.tokenizer(prompt).input_ids
        token_counts.append(len(input_ids))
        embeddings = model.encode(prompt, instruction="", max_length=max_length) # model.encode([prompt])[0]
        ret["embedding"] = embeddings.tolist()
        ret["token_num"] = token_counts
        return ret
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    