import torch
from diffusers import AutoPipelineForText2Image


device = "cuda"
dtype = torch.float16

pipeline =  AutoPipelineForText2Image.from_pretrained(
    "/mnt/ai-llm/models/wuerstchen", torch_dtype=dtype
).to(device)

prompt = "Anthropomorphic cat dressed as a fire fighter"
dirpath = "/home/agent/workspace"

while True:
    prompt = input("Enter a prompt: ")
    output = pipeline(
        prompt=prompt,
        height=1024,
        width=1024,
        prior_guidance_scale=4.0,
        decoder_guidance_scale=0.0,
    )
    for idx, image in enumerate(output.images):
        image_name = f'{idx}.png'
        image_path = f"{dirpath}/{image_name}"
        image.save(image_path)

