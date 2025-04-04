import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import json
from typing import List

app = FastAPI()

model_path = os.getenv("MODEL_PATH", "Alibaba-NLP/gte-large-en-v1.5")
model = SentenceTransformer(model_path, trust_remote_code=True)

class SentencesRequest(BaseModel):
    sentences: List[str]

@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)

@app.post("/encode")
def encode_sentences(request: SentencesRequest):
    try:
        ret = {"embedding": [], "token_num": 0}
        prompts = request.sentences
        token_counts = []
        for prompt in prompts:
            input_ids = model.tokenizer(prompt).input_ids
            token_counts.append(len(input_ids))
        embeddings = model.encode(prompts)
        ret["embedding"] = embeddings.tolist()
        ret["token_num"] = token_counts
        return ret
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)