import PIL
import PIL.Image
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline

class DiffusionModel():
    def __init__(self, model_path):
        self.model_path = model_path
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(self.model_path, torch_dtype=torch.float16).to("cuda")
        self.generator = torch.Generator("cuda").manual_seed(0)
    
    def predict_next_image(self, image_path, prompt, num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=10):
        image = PIL.Image.open(image_path)
        edited_image = self.pipe(prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            image_guidance_scale=image_guidance_scale,
            guidance_scale=guidance_scale,
            generator=self.generator,
        ).images[0]
        return edited_image
