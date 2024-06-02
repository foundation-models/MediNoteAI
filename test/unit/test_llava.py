from transformers import pipeline
from PIL import Image    
import requests

model_id = "/mnt/models/vip-llava-7b-hf"
pipe = pipeline("image-to-text", model=model_id)
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"

image = Image.open(requests.get(url, stream=True).raw)
question = "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud"
prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: <image>\n{question}###Assistant:"

outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
print(outputs)
