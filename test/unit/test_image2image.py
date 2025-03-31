from diffusers import AutoPipelineForImage2Image
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInstructPix2PixPipeline
from diffusers.utils import load_image

resolution = 768
image = load_image(
    "/home/agent/workspace/screenshot.png"
).resize((resolution, resolution))

# pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
#     "/mnt/ai-llm/models/sdxl-instructpix2pix-768", torch_dtype=torch.float16
# ).to("cuda")

# pipe = AutoPipelineForImage2Image.from_pretrained("/mnt/ai-llm/models/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to("cuda")

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "/mnt/ai-llm/models/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

while True:
    prompt = input("Enter a prompt: ")
    edited_image = pipe(
        prompt=prompt,
        image=image,
        # num_inference_steps=2, 
        # strength=0.5, 
        # guidance_scale=0.0
        ).images[0]
    edited_image.save("/home/agent/workspace/edited_image.png")
