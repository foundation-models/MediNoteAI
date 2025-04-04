from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests

# model = InstructBlipForConditionalGeneration.from_pretrained("/mnt/models/instructblip-vicuna-7b")
# processor = InstructBlipProcessor.from_pretrained("/mnt/models/instructblip-vicuna-7b")
# model = InstructBlipForConditionalGeneration.from_pretrained("/mnt/models/instructblip-vicuna-7b-gptq-4bit", device_map="auto", torch_dtype=torch.bfloat16, use_safetensors=True, load_in_4bit=True)
model = InstructBlipForConditionalGeneration.from_pretrained("/mnt/models-nfs/models/instructblip-vicuna-7b_8bit", device_map="auto", torch_dtype=torch.bfloat16, use_safetensors=True, load_in_8bit=True)
processor = InstructBlipProcessor.from_pretrained("/mnt/models-nfs/models/instructblip-vicuna-7b_8bit")
print("passed model load")
# processor = InstructBlipProcessor.from_pretrained("/mnt/models/instructblip-vicuna-7b_8bit", device_map="auto", torch_dtype=torch.bfloat16, use_safetensors=True, load_in_8bit=True)
print("passed processor load")
# model = InstructBlipForConditionalGeneration.from_pretrained("/mnt/models/instructblip-flan-t5-xl")
# processor = InstructBlipProcessor.from_pretrained("/mnt/models/instructblip-flan-t5-xl")

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)

# url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
# image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
# image_src = '/home/agent/workspace/shahram_in_surgery_room.jpg'
image_src = 'man_moving_trash_can.png'

image = Image.open(open(image_src, 'rb')).convert("RGB")
# prompt = "What is unusual about this image?"
prompt = "Describe this image"
inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1,
)
generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
print(generated_text)
