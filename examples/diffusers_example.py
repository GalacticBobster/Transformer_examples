from diffusers import AutoPipelineForText2Image

prompt = "A cat dressed up like batman"
pipeline = AutoPipelineForText2Image.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
image = pipeline(prompt).images[0]

image.show()
