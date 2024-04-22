import torch
from diffusers import AutoPipelineForText2Image


device = "cuda"
dtype = torch.float16


pipe = AutoPipelineForText2Image.from_pretrained("/mnt/ai-llm/models/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
dirpath = "/home/agent/workspace"

while True:
    prompt = input("Enter a prompt: ")
    output = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0)
    for idx, image in enumerate(output.images):
        image_name = f'{idx}.png'
        image_path = f"{dirpath}/{image_name}"
        image.save(image_path)

