"""Get and preprocess data."""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import re
import glob
import cv2
from tqdm import tqdm

import zipfile
import numpy as np

import nibabel as nib

import tensorflow as tf

DATA_URL = "https://app.box.com/shared/static/l5zoa0bjp1pigpgcgakup83pzadm6wxs.zip"


def maybe_download():
    """Downloads image data and segmentation labels of the TCGA-GBM collection."""
    # Download zip file and return path
    zip_file = tf.keras.utils.get_file("TCGA-GBM.zip", DATA_URL)

    # extracted data path
    data_path = os.path.join(zip_file[:-12], "TCGA-GBM")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Extract data from zip file
    with zipfile.ZipFile(zip_file) as zf:
        zf.extractall(data_path)

    return data_path


def process_images(path, data_path):
    """Process both input images and output masks."""
    # Get patient id
    pat_id = path.split("/")[-1]

    # Get output mask
    label_file = glob.glob(os.path.join(path, "*GlistrBoost.nii.gz"))[0]
    img = nib.load(label_file)
    label_img_data = img.get_fdata()

    # Transform masks to binary segmentation
    label_img_data[label_img_data > 0] = 1.0

    # Get T1-Gd, T2, and Flair modality files
    t1Gd_file = glob.glob(os.path.join(path, "*t1Gd.nii.gz"))[0]
    t2_file = glob.glob(os.path.join(path, "*t2.nii.gz"))[0]
    flair_file = glob.glob(os.path.join(path, "*flair.nii.gz"))[0]
    input_img_data = np.empty((240, 240, 3, label_img_data.shape[-1]))

    # Input image data and scale to [0,255]
    for i, f in enumerate([t1Gd_file, t2_file, flair_file]):
        img = nib.load(f)
        img_data = img.get_fdata()
        # Rescale
        img_data *= 255.0 / img_data.max()
        # Store modality as channel
        input_img_data[:, :, i, :] = np.round(img_data)

    # Save input image as jpeg and mask as compressed sparse matrix
    file_ids = []
    for i in range(label_img_data.shape[-1]):
        file_id = f"{pat_id}-{i}"
        # Save image
        img_path = os.path.join(
            data_path + "/processed/input_images/", f"{file_id}.jpg"
        )
        cv2.imwrite(img_path, input_img_data[:, :, :, i])
        # Save mask
        mask_path = os.path.join(
            data_path + "/processed/output_masks/", f"{file_id}.jpg"
        )
        cv2.imwrite(mask_path, label_img_data[:, :, i])
        # Keep file_id
        file_ids.append(file_id)

    return file_ids


if __name__ == "__main__":

    # Download data
    print("Maybe downloading TCGA-GBM image data")
    data_path = maybe_download()

    # Make directories for processed images and masks
    os.makedirs(os.path.join(data_path, "processed/input_images"), exist_ok=True)
    os.makedirs(os.path.join(data_path, "processed/output_masks"), exist_ok=True)

    # Get paths to all image directories
    img_paths = glob.glob(os.path.join(data_path, "*/TCGA-*"))

    # Process images
    print("Processing and saving images")
    for img_path in tqdm(img_paths):
        _ = process_images(img_path, data_path)
    print("Done")
