from transformers import pipeline
from PIL import Image    
# import requests

model_id = "/mnt/models-nfs/models/llava-1.5-7b-hf"
pipe = pipeline("image-to-text", model=model_id)

# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

iamge_src = '/home/agent/workspace/shahram_in_surgery_room.jpg'
image = Image.open(iamge_src)

# prompt = "USER: <image>\nWhat does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud\nASSISTANT:"
prompt = "USER: <image>\nDescribe this image... \nASSISTANT:"

outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
print(outputs)
