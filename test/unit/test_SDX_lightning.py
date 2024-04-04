import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
# from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

base = "/mnt/models/stable-diffusion-xl-base-1.0"
# repo = "/mnt/models/SDXL-Lightning"
ckpt = "/mnt/models/SDXL-Lightning/sdxl_lightning_8step_unet.safetensors" # Use the correct ckpt for your step setting!

# Load model.
print("1111")
unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
print("222")

# unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
unet.load_state_dict(load_file(ckpt, device="cuda"))
pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to("cuda")
print("333")
# Ensure sampler uses "trailing" timesteps.
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
print("Now")

# Ensure using the same inference steps as the loaded model and CFG set to 0.
pipe("A girl smiling", num_inference_steps=4, guidance_scale=0).images[0].save("output.png")
print("Done")