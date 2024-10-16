import requests
from PIL import Image

import torch
from transformers import AutoProcessor, VipLlavaForConditionalGeneration

model_id = "/mnt/models/vip-llava-7b-hf"

question = "What are these?"
question = "how many women are in the room?"
question = "would you describe this image?"

prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: <image>\n{question}###Assistant:"

image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
image_file = "https://dttdrlk9qx747.cloudfront.net/cms/thumbnails/00/445x300/sub/49822/images/bigstock-Group-Of-Seniors-Playing-Game-258166369.0000000000000.jpg"


model = VipLlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to(0)

processor = AutoProcessor.from_pretrained(model_id)


image = Image.open(requests.get(image_file, stream=True).raw)
# image = Image.open(open("/home/agent/workspace/surgery_room.jpg", 'rb')).convert("RGB")

inputs = processor(prompt, image, return_tensors='pt').to(0, torch.float16)

output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))