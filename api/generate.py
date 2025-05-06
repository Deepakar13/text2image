from diffusers import StableDiffusionPipeline
import torch

def load_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

def generate_image(prompt: str, guidance_scale: float = 7.5, steps: int = 50):
    pipe = load_pipeline()
    image = pipe(prompt, guidance_scale=guidance_scale, num_inference_steps=steps).images[0]
    return image