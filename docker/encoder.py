from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List

app = FastAPI()

model = SentenceTransformer('/mnt/models/gte-large-en-v1.5', trust_remote_code=True)

class SentencesRequest(BaseModel):
    sentences: List[str]

@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)

@app.post("/encode")
def encode_sentences(request: SentencesRequest):
    try:
        embeddings = model.encode(request.sentences)      
        return {"embeddings": embeddings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
