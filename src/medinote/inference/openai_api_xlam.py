from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Load model and tokenizer
model_name = "/mnt/models/xLAM-1b-fc-r"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define input schema for the request
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 1.0  # Default temperature to 1.0 if not provided

@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    if request.model != "xLAM-1b-fc-r":
        raise HTTPException(status_code=400, detail="Unsupported model. This API only supports 'xLAM-1b-fc-r'.")
    
    # Validate the user message
    user_message = next((msg for msg in request.messages if msg.role == "user"), None)
    if user_message is None:
        raise HTTPException(status_code=400, detail="No message with role 'user' found.")

    input_text = user_message.content

    # Prepare input tokens
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    # Generate output from the model using temperature
    with torch.no_grad():
        output = model.generate(**inputs, max_length=512, temperature=request.temperature)
    
    # Decode output tokens
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Prepare the response in OpenAI API style
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": response_text
                }
            }
        ]
    }
    
    return response

# To run the FastAPI app, use the command:
# uvicorn app:app --host 0.0.0.0 --port 8000
