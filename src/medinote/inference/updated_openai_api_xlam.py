
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from medinote import initialize

# Initialize config and logger
main_config, logger = initialize(
    logger_name=os.path.splitext(os.path.basename(__file__))[0],
    root_path=os.environ.get("ROOT_PATH") or os.path.join(os.path.dirname(__file__), "..")
)

config = main_config.get(os.path.basename(__file__)[:-3])

# Define device to use (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FastAPI app initialization
app = FastAPI()

# Model and Tokenizer loading function
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return model, tokenizer

# Example API request structure
class RequestBody(BaseModel):
    input_text: str

# Inference route in FastAPI
@app.post("/inference/")
async def run_inference(request_body: RequestBody):
    try:
        # Load the model and tokenizer
        model_name = "gpt2"  # Example model, update if needed
        model, tokenizer = load_model_and_tokenizer(model_name)

        # Tokenize input text
        inputs = tokenizer(request_body.input_text, return_tensors='pt', padding=True, truncation=True)
        
        # Move inputs to the correct device (GPU/CPU)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        # Generate text with attention mask
        outputs = model.generate(**inputs, attention_mask=inputs['attention_mask'])
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {"generated_text": generated_text}
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

