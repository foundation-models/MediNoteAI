from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("/home/agent/workspace/models/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "/home/agent/workspace/models/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
)  # doctest: +IGNORE_RESULT

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# prompt = "Question: how many cats are there? Answer:"

image = Image.open(open("/home/agent/workspace/surgery_room.jpg", 'rb')).convert("RGB")
prompt = "Question: how many men are in the room? Answer:"
# prompt = "Question: describe it in detail? Answer:"

inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

generated_ids = model.generate(**inputs)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(f"No prompt: {generated_text}")


inputs = processor(images=image, text=prompt, return_tensors="pt", truncation=False).to(device="cuda", dtype=torch.float16)

generated_ids = model.generate(**inputs)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(f"With prompt: {prompt}\n{generated_text}")

