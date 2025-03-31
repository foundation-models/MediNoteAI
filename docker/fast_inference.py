import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from unsloth import FastLanguageModel
# from transformers import TextStreamer

app = FastAPI()

model_path = os.getenv("MODEL_PATH", "unsloth/Phi-3-mini-4k-instruct-bnb-4bit")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = 4096,
    dtype = None,
    load_in_4bit = True
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

FastLanguageModel.for_inference(model)


prompt_template = """
<|user|>
{prompt}<|end|>
<|assistant|>"""

class PromptsRequest(BaseModel):
    prompt: str

@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)

@app.post("/v1/completions")
def encode_sentences(request: PromptsRequest):
    try:
        ret = {}
        prompt = request.prompt
        inputs =  tokenizer(
            [
                prompt_template.format(prompt=prompt)
            ] * 1,
            return_tensors="pt",
        ).to("cuda")


        # x = model.generate(**inputs, streamer=text_streamer, max_new_tokens=1028)
        x = model.generate(**inputs, max_new_tokens=2048, use_cache = True)
        # Decode the generated tokens back to text
        generated_text = tokenizer.decode(x[0], skip_special_tokens=True)

        # Re-tokenize the generated text to get the token count
        tokenized_output = tokenizer(generated_text, return_tensors="pt")

        # Token count
        token_count = tokenized_output.input_ids.size(1)        
        response = generated_text.replace(prompt, "")
        response = response.strip()
        ret["response"] = response
        ret["token_count"] = token_count
        return ret
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)