import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
# from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

#========= Image Encode/Decode ==========
import base64
from PIL import Image
import io
#=+=+=+=+= Image Encode/Decode =+=+=+=+=+



#========= model ==========
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
#=+=+=+=+= model =+=+=+=+=+



    # Call your model's generate function
#result = await model_generate(prompt)

    


async def model_generate(prompt):
    try:
        #text = f"A chat between a curious human and an artificial intelligence assistant.###Human: <image>\n{prompt}
        ret = {"prompt": prompt}
        
        
        # Load model.
        print("1111")
        
        
        print("333")
        # Ensure sampler uses "trailing" timesteps.
        
        print("Now")
        # Ensure using the same inference steps as the loaded model and CFG set to 0.
        response = pipe(prompt, num_inference_steps=8, guidance_scale=0).images[0].save("output.png")
        with open("output.png", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        print("Done")
        #output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        #response =processor.decode(output[0][2:], skip_special_tokens=True)
        ret["answer"] = encoded_string
    
    except:
       ret = {
             "error_code": "Error"
         } 
    return ret         

