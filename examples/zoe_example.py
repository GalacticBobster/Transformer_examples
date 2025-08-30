from transformers import AutoImageProcessor, ZoeDepthForDepthEstimation
import torch
import numpy as np
from PIL import Image

# Load Zoe model
image_processor = AutoImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti")
model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti")

# Load lunar image
image = Image.open("/home/ananyo/Desktop/PSL/CH3/lim/nop/data/calibrated/20230902/ch3_lim_nc1_20230902T2121108000_d_img_d32_1_228.jpg")  # downloaded NAC/WAC image

# Preprocess
inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Resize depth map back to original image size
post_processed_output = image_processor.post_process_depth_estimation(
    outputs,
    source_sizes=[(image.height, image.width)],
)
predicted_depth = post_processed_output[0]["predicted_depth"]

# Normalize depth to [0, 255] for visualization
depth = predicted_depth / predicted_depth.max()
depth_np = (depth * 255).detach().cpu().numpy().astype("uint8")
Image.fromarray(depth_np).save("depth_map.png")

