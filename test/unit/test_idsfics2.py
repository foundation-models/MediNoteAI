import requests
import torch
from PIL import Image
from io import BytesIO

from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

# DEVICE = "cuda:0"
DEVICE = "cpu"
# DEVICE = "auto"

# Note that passing the image urls (instead of the actual pil images) to the processor is also possible
# image1 = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")
# image2 = load_image("https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg")
# image3 = load_image("https://cdn.britannica.com/68/170868-050-8DDE8263/Golden-Gate-Bridge-San-Francisco.jpg")
image4 = Image.open(open("/mnt/ai-nfs/workspace/handwritten1.png", 'rb')).convert("RGB")

processor = AutoProcessor.from_pretrained("/mnt/ai-llm/models/idefics2-8b")
model = AutoModelForVision2Seq.from_pretrained(
    "/mnt/ai-llm/models/idefics2-8b",
).to(DEVICE)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Can you read the dentist handwritten text?"},
        ]
    },      
]
# Create inputs
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image"},
#             {"type": "text", "text": "What do we see in this image?"},
#         ]
#     },
#     {
#         "role": "assistant",
#         "content": [
#             {"type": "text", "text": "In this image, we can see the city of New York, and more specifically the Statue of Liberty."},
#         ]
#     },
#     {
#         "role": "user",
#         "content": [
#             {"type": "image"},
#             {"type": "text", "text": "And how about this image?"},
#         ]
#     },       
# ]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image4], return_tensors="pt")
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}


# Generate
generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

print(generated_texts)
