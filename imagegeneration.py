!pip install -q diffusers transformers accelerate safetensors torch

import torch
from diffusers import StableDiffusionPipeline
from IPython.display import display
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

description = input("Describe the image you want: ")

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
).to(device)

image = pipe(description).images[0]
display(image)

output_path = "generated_image.png"
image.save(output_path)

def download_output_image(output_path):
    from google.colab import files
    files.download(output_path)

download_output_image(output_path)
