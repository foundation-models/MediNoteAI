import csv
import os
from PIL import Image
from dvc import api

import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from safetensors.torch import load_file




#========= model ==========
base = "/mnt/ai-nfs/models/stable-diffusion-xl-base-1.0"
# repo = "/mnt/models/SDXL-Lightning"
ckpt = "/mnt/ai-nfs/models/SDXL-Lightning/sdxl_lightning_8step_unet.safetensors" # Use the correct ckpt for your step setting!

# Load model.
unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)

# unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
unet.load_state_dict(load_file(ckpt, device="cuda"))
pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to("cuda")

# Ensure sampler uses "trailing" timesteps.
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

# Ensure using the same inference steps as the loaded model and CFG set to 0.
pipe("A girl smiling", num_inference_steps=4, guidance_scale=0).images[0].save("output.png")

#=+=+=+=+= model =+=+=+=+=+
# Load the CSV file with the prompts
with open('prompts.csv', 'r') as f:
    reader = csv.reader(f)
    prompts = [row[0] for row in reader]

# Initialize the DVC API
api = api.DVC()

# Loop through each prompt and generate an image
for prompt in prompts:
    # Generate the image using your model
    #image = generate_image(prompt)
    image = await model_generate(prompt)
    # Save the image to a file
    filename = f'{prompt}.png'
    with open(filename, 'wb') as f:
        Image.fromarray(image).save(f)

    # Add the image to DVC
    api.add(filename)
    
    
async def model_generate(prompt):
    try:
        ret = {"prompt": prompt}
        
        # Ensure using the same inference steps as the loaded model and CFG set to 0.
        response = pipe(prompt, num_inference_steps=8, guidance_scale=0).images[0].save("output.png")
        # with open("output.png", "rb") as image_file:
        #     encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        # ret["answer"] = encoded_string
    except:
       ret = {
             "error_code": "Error"
         } 
    return response 

# def generate_image(prompt):
#     # Generate the image using your model
#     response = await model_generate(prompt)
    
#     # Convert the output to an image object
#     image = Image.fromarray(response).convert('RGB')
    
#     return image