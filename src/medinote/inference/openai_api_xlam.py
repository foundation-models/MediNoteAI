import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from medinote import initialize


main_config, logger = initialize(
    logger_name=os.path.splitext(os.path.basename(__file__))[0],
    root_path=os.environ.get("ROOT_PATH") or os.path.join(os.path.dirname(__file__), "..")
)

config = main_config.get(os.path.basename(__file__)[:-3])
# Assume task_instruction, format_instruction, and xlam_format_tools are predefined
task_instruction = config.get("task_instruction")
format_instruction = config.get("format_instruction")
get_weather_api = config.get("get_weather_api")
search_api = config.get("search_api")
openai_format_tools = [get_weather_api, search_api]

app = FastAPI()

# Load model and tokenizer
torch.random.manual_seed(0) 
model_name = config.get("model_name")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Model and Tokenizer loading function
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return model, tokenizer
# Helper function to convert openai format tools to our more concise xLAM format
def convert_to_xlam_tool(tools):
    ''''''
    if isinstance(tools, dict):
        return {
            "name": tools["name"],
            "description": tools["description"],
            "parameters": {k: v for k, v in tools["parameters"].get("properties", {}).items()}
        }
    elif isinstance(tools, list):
        return [convert_to_xlam_tool(tool) for tool in tools]
    else:
        return tools
    
xlam_format_tools = convert_to_xlam_tool(openai_format_tools)

# Helper function to build the input prompt for our model
def build_prompt(task_instruction: str, format_instruction: str, tools: list, query: str):
    prompt = f"[BEGIN OF TASK INSTRUCTION]\n{task_instruction}\n[END OF TASK INSTRUCTION]\n\n"
    prompt += f"[BEGIN OF AVAILABLE TOOLS]\n{json.dumps(tools)}\n[END OF AVAILABLE TOOLS]\n\n"
    prompt += f"[BEGIN OF FORMAT INSTRUCTION]\n{format_instruction}\n[END OF FORMAT INSTRUCTION]\n\n"
    prompt += f"[BEGIN OF QUERY]\n{query}\n[END OF QUERY]\n\n"
    return prompt

# Define input schema for the request
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 1.0  # Default temperature to 1.0 if not provided

model, tokenizer = load_model_and_tokenizer(model_name)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    if request.model != "xLAM-1b-fc-r":
        raise HTTPException(status_code=400, detail="Unsupported model. This API only supports 'xLAM-1b-fc-r'.")

    # Extract user query from the messages
    user_message = next((msg for msg in request.messages if msg.role == "user"), None)
    if user_message is None:
        raise HTTPException(status_code=400, detail="No message with role 'user' found.")

    query = user_message.content


    # Build the input prompt using the helper function
    content = build_prompt(task_instruction, format_instruction, xlam_format_tools, query)

    # Prepare the input for the model
    messages = [{'role': 'user', 'content': content}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

    # Perform inference with the model
    outputs = model.generate(
        inputs, 
        max_new_tokens=512, 
        do_sample=False, 
        num_return_sequences=1, 
        eos_token_id=tokenizer.eos_token_id
    )

    # Decode the model's output
    response_text = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)