from ..transformations import CrossModalityTransformations
from ..utils.image_utils import load_npy_image, save_npy_image, display_image_gray, display_image_color
from ..enums import Input_Modality, Output_Modality
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


def save_presentation_images(image_name, image):
    output_path = os.path.join(r"F:\Bachelorarbeit Coding\presentation_images\08_07_24", f"{image_name}.png")
    plt.imsave(output_path, image, cmap='gray', vmin=0, vmax=255)

''
images_retina = np.load(r"F:/Bachelorarbeit Coding/datasets/retinamnist_224.npz")['train_images']
images_retina_resized = []
for i in images_retina:
    i = resize(i, (256, 256), anti_aliasing=True)
    image_min = np.min(i)
    image_max = np.max(i)
    if image_max != image_min:
        i = (i - image_min) / (image_max - image_min) * 255
    else:
        i = np.zeros_like(i)
    i = i.astype(np.uint8)
    images_retina_resized.append(i)
    break
''''''

#images_retina_resized = np.array(images_retina_resized)
# Load input image
image_to_be_transformed_1 = load_npy_image(r"F:\Bachelorarbeit Coding\data\brain_np\CT\both\ID_000a631d3.dcm.npy")
#image_to_be_transformed_2 = images_retina_resized[0]

# Create transformation instance
cross_modality_transformer = CrossModalityTransformations(
    input_modality=Input_Modality.CT, 
    output_modality=Output_Modality.MRI, 
    color_ratio_range=(0, 1), 
    artifact_ratio_range=(0, 1), 
    spatial_resolution_ratio_range=(0, 1), 
    noise_ratio_range=(0, 1)
)

# Apply transformations
output_image = cross_modality_transformer.transform(image_to_be_transformed_1)

# Display and save transformed image
display_image_gray(output_image)
save_presentation_images("brain_MRI_chemical", output_image)


# README.md Datei
# LICENSE?!
# setup.py
# requirements.txt
# inform about how to evaluate your metrics
# make rough structure of the thesis and which kind of images you want to create for a qualitative evaluation (z.B. ein Bild, was in mehrere Modalitäten umgewandelt wird, wobei das originalbild in der Mitte ist und die augmentierten Bilder außenrum im Kreis sind)
# connect to workstation
# vote in vc post
# make presentation for thursday