from medinote import initialize
import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
# from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import time

config, logger = initialize(caller_module_name="image_gen_augmentation")

base = config.get("image_gen")['base_model']
ckpt = config.get("image_gen")['checkpoint']
unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
unet.load_state_dict(load_file(ckpt, device="cuda"))
pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to("cuda")


def generate_image_from_text(file_text_map: dict = None, 
                             num_inference_steps: int = None,
                             guidance_scale: int = None,
                             output_path: str = None
                             ):
    """_summary_
    
    Generate an image from a text.
    """
    file_text_map = file_text_map or config.get("image_gen")['file_text_map']
    num_inference_steps = num_inference_steps or config.get("image_gen")['num_inference_steps']
    guidance_scale = guidance_scale or config.get("image_gen")['guidance_scale']
    output_path = output_path or config.get("image_gen")['output_path']
    
    
    
    for file_name, text in file_text_map.items():
        output_file = f"{output_path}/{file_name}.png" if output_path else file_name + ".png"
        
        start_time = time.time()  # Mark the start time

        pipe(text, 
             num_inference_steps=num_inference_steps, 
             guidance_scale=guidance_scale
             ).images[0].save(output_file)
    
        end_time = time.time()  # Mark the end time
        generation_time = end_time - start_time  # Calculate the time taken

        logger.info(f"Generated image '{output_file}' in {generation_time:.2f} seconds")  # Log the time
        
if __name__ == "__main__":
    generate_image_from_text()
    