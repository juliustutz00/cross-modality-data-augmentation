import numpy as np
import os
import pydicom
from skimage.transform import resize
import matplotlib.pyplot as plt


def read_dicom_images_to_npy(directory, resolution):
    """
    Reads pixel_arrays of dicom (.dcm) images out of a given directory to a NumPy array. 
    The dicom files have to be normalized, meaning: same quadratic shape, 
    removed noise, similar size ratio.

    Parameters
    ----------
    directory
        Input directory with the dicom (.dcm) images.
    resolution
        Desired image resolution. May be (64, 64), (128, 128), (256, 256), or 
        (512, 512). Otherwise the default image size is kept.

    Returns
    -------
    np.ndarray
        Array of pixel_arrays of given dicom (.dcm) images.
    """
    images = []
    for filename in os.listdir(directory):
        if filename.endswith('.dcm'):
            ds = pydicom.dcmread(os.path.join(directory, filename))
            ds_pixel_array = ds.pixel_array
            if ((resolution == (64, 64)) | (resolution == (128, 128)) | (resolution == (256, 256)) | (resolution == (512, 512))):
                ds_pixel_array = resize(ds_pixel_array, resolution, anti_aliasing=True)
            image_min = np.min(ds_pixel_array)
            image_max = np.max(ds_pixel_array)
            if image_max != image_min:
                normalized_image = (ds_pixel_array - image_min) / (image_max - image_min) * 255
            else:
                normalized_image = np.zeros_like(ds_pixel_array)
            normalized_image = normalized_image.astype(np.uint8)
            images.append(normalized_image)
    return np.array(images)

def read_npy_images_to_npy(directory): 
    images = []
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            image = np.load(os.path.join(directory, filename))
            images.append(image)
            filenames.append(filename)
    images = np.array(images)
    filenames = np.array(filenames)
    return images, filenames

def load_npy_image(image_path):
    return np.load(image_path)

def save_npy_image(image_path, image: np.ndarray):
    np.save(image_path, image)

def display_image_gray(image: np.ndarray): 
    plt.figure()
    plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    plt.show()

def display_image_color(image: np.ndarray):
    plt.figure()
    plt.imshow(image)
    plt.show()

def browse_images_gray(images, filenames):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    index = 0
    ax.imshow(images[index], cmap="gray", vmin=0, vmax=255)
    plt.title(f'Image {index + 1}/{len(images)}; {filenames[index]}')

    def on_key(event):
        nonlocal index
        if event.key == 'right':
            index = (index + 1) % len(images)
        elif event.key == 'left':
            index = (index - 1) % len(images)
        ax.imshow(images[index], cmap="gray", vmin=0, vmax=255)
        plt.title(f'Image {index + 1}/{len(images)}; {filenames[index]}')
        fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

def browse_images_color(images, filenames):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    index = 0
    ax.imshow(images[index])
    plt.title(f'Image {index + 1}/{len(images)}; {filenames[index]}')

    def on_key(event):
        nonlocal index
        if event.key == 'right':
            index = (index + 1) % len(images)
        elif event.key == 'left':
            index = (index - 1) % len(images)
        ax.imshow(images[index])
        plt.title(f'Image {index + 1}/{len(images)}; {filenames[index]}')
        fig.canvas.draw()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

def load_custom_dataset(folder_path, modality):
    images = np.load(folder_path)['train_images']
    images = images[:1000]
    dataset = []
    for image in images:
        image = resize(image, (256, 256), anti_aliasing=True)
        image_min = np.min(image)
        image_max = np.max(image)
        if image_max != image_min:
            normalized_image = (image - image_min) / (image_max - image_min) * 255
        else:
            normalized_image = np.zeros_like(image)
        normalized_image = normalized_image.astype(np.uint8)
        dataset.append(normalized_image)
    dataset = np.array(dataset)
    mean_color_values = np.round(np.mean(dataset, axis=0)).astype(np.uint8)
    current_dir = os.path.dirname(__file__)
    output_path = current_dir + "/reference_images/custom/" + modality + "_256x256"
    np.save(output_path, mean_color_values)
