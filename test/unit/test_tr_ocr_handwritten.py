from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests

# load image from the IAM database
# url = 'https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg'
# image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

image = Image.open(open("/mnt/ai-nfs/workspace/screenshot.png", 'rb')).convert("RGB")


processor = TrOCRProcessor.from_pretrained('/mnt/ai-llm/models/trocr-large-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
pixel_values = processor(images=image, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("generated_text: ", generated_text)