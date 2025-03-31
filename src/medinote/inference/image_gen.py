import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
# from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

#========= Image Encode/Decode ==========
import base64
from PIL import Image
import io
#=+=+=+=+= Image Encode/Decode =+=+=+=+=+

#========= fastAPI ==========
import argparse
import uvicorn
from fastapi import Request, FastAPI, Form, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from pathlib import Path
#=+=+=+=+= fastAPI =+=+=+=+=+

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

app = FastAPI(
    title="Generative AI",
    description="Generative AI API"
)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
async def default():
    """
    Returns a 200 to signal that the server is up.
    """
    return {"status": "ok"}


@app.get("/liveness")
async def liveness():
    """
    Returns a 200 to signal that the server is up.
    """
    return {"status": "live"}


@app.get("/readiness")
async def readiness():
    """
    Returns a 200 if the server is ready for prediction.
    """

    # Currently simply take the first tenant.
    # to decrease chances of loading a not needed model.
    return {"status": "ready"}

@app.get("/image-gen", response_class=HTMLResponse)
async def get_text_form(request: Request):
    domain = request.url_for('get_text_form')
    form_html = Path("/home/agent/workspace/MediNoteAI/test/unit/text_form.html").read_text()
    # Replace the placeholder with the actual domain
    form_html = form_html.replace("http://localhost:8888", f'http://{str(domain).split("://")[1].split("/")[0]}')
   
    
    return HTMLResponse(content=form_html)

@app.post("/image-gen/")
async def send_gat_data(prompt: str = Form(...)):
    # Read the uploaded file
    #contents = await file.read()
    #image = Image.open(io.BytesIO(contents))

    # Convert PIL Image to the required format for model.generate here if necessary

    # Call your model's generate function
    result = await model_generate(prompt)

    return JSONResponse(content=result)
    


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
    # except error as e:
    #     ret = {
    #         "text": f"{SERVER_ERROR_MSG}\n\n({e})",
    #         "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
    #     }
    except:
       ret = {
             "error_code": "Error"
         } 
    return ret         

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args()
    app.title = "Generate API"
    app.description = "API for SDX"
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
