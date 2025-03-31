
import urllib.request 
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.tokenization_mplug_owl import MplugOwlTokenizer
from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
import torch

pretrained_ckpt = '/mnt/models-nfs/models/mplug-owl-llama-7b'
model = MplugOwlForConditionalGeneration.from_pretrained(
    pretrained_ckpt,
    torch_dtype=torch.bfloat16,
    device_map="auto"
    # use_flash_attention_2=True, # doesn't support yet
)
image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
tokenizer = MplugOwlTokenizer.from_pretrained(pretrained_ckpt)
processor = MplugOwlProcessor(image_processor, tokenizer)

# We use a human/AI template to organize the context as a multi-turn conversation.
# <image> denotes an image placehold.
prompts = [
'''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <image>
Human: Describe the image.
AI: ''']

# The image paths should be placed in the image_list and kept in the same order as in the prompts.
# We support urls, local file paths and base64 string. You can custom the pre-process of images by modifying the mplug_owl.modeling_mplug_owl.ImageProcessor
# image_list = ['/home/agent/workspace/man_moving_trash_can.png', 'https://climaterwc.com/live/wp-content/uploads/2021/05/sanmateo-outdoordining-cityphoto.jpg', 'https://www.care.com/c/wp-content/uploads/sites/2/2021/04/LaurenGarcia-201944221944690550-1584x1080.jpg.webp' ]
# image_list = ['https://climaterwc.com/live/wp-content/uploads/2021/05/sanmateo-outdoordining-cityphoto.jpg' ]
image_list = ['/home/agent/workspace/shahram_in_surgery_room.jpg']

# generate kwargs (the same in transformers) can be passed in the do_generate()
generate_kwargs = {
    'do_sample': True,
    'top_k': 5,
    'max_length': 512
}
from PIL import Image

images = []
for iamge_src in image_list:
    if iamge_src.startswith('http'):
        urllib.request.urlretrieve(iamge_src, "tmp.png")
        image = Image.open("tmp.png")
    else:
        image = Image.open(iamge_src)
    images.append(image)

inputs = processor(text=prompts, images=images, return_tensors='pt')
inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    res = model.generate(**inputs, **generate_kwargs)
sentence = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
print(sentence)
