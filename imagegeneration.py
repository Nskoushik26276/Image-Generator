!pip install -q diffusers transformers accelerate safetensors torch

import torch
from diffusers import StableDiffusionPipeline
from IPython.display import display

device = "cuda" if torch.cuda.is_available() else "cpu"

description = input("Describe the image you want: ")

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
).to(device)

image = pipe(description).images[0]
display(image)
