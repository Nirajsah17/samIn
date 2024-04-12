import torch
# import torchvision

import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel

import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic
# import warnings

# checkpoint either is of sam or GeoSAM ckpt["sam_decoder_multi.pth"] modelTYpe="vit_h"



import numpy as np
import os
from PIL import Image


points = np.array([[1000, 1000], [2000, 2000]])
labels = np.array([1])

def write_mask_images(masks, predictions, low_res_logits, output_dir):
    """
    Write all mask images to the system.

    Args:
    - masks (np.ndarray): The output masks in CxHxW format, where C is the number of masks,
                          and (H, W) is the original image size.
    - predictions (np.ndarray): An array of length C containing the model's predictions for the quality
                                of each mask.
    - low_res_logits (np.ndarray): An array of shape CxHxW, where C is the number of masks and H=W=256.
                                   These low resolution logits can be passed to a subsequent iteration as mask input.
    - output_dir (str): The directory where the mask images will be saved.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through each mask
    for i, mask in enumerate(masks):
        # Convert mask to image format (assuming mask values are in [0, 255] range)
        mask_image = Image.fromarray(mask.astype(np.uint8))

        # Save mask image
        mask_image.save(os.path.join(output_dir, f"mask_{i}.png"))

checkpoint = "sam_decoder_multi.pth"
model_type = "vit_h"
fileName = '5back00040.jpg'
image = cv2.imread(fileName)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device='cpu')
predictor = SamPredictor(sam)
predictor.set_image(image)
image_embedding = predictor.get_image_embedding().cpu().numpy()
print("Embedding generated")
embName = fileName.split(".")[0]+".npy"
np.save(embName,image_embedding)
print("saving",embName)

print("generating prediction")  
res = predictor.predict(point_coords=points,point_labels=labels)
write_mask_images(res[0], res[1], res[2], "images")

shape = res[0]
print("first",shape)

print("----------------------------------------------------------------------------------------------------------")
print(res)










