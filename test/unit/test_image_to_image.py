import torch
from diffusers.utils import load_image
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image


pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "/mnt/models/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)

pipe = pipe.to("cuda")
init_image = Image.open(open("/home/agent/workspace/img2img-init.png", 'rb')).convert("RGB")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt, image=init_image).images[0]
print("Image generated")
image.save('/home/agent/workspace/test5.jpg')

print("Done")