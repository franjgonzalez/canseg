"""Data utility functions."""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import glob

from tqdm import tqdm

import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.model_selection import train_test_split

import tensorflow as tf

DATA_PATH = os.path.join(os.getenv("HOME"), ".keras/datasets/TCGA-GBM/processed")


def get_data(params):
    """Get list of paths to data and do train-test split."""

    # Get input image and label paths
    img_paths = glob.glob(os.path.join(DATA_PATH, "input_images/*.jpg"))
    base_path = os.path.join(DATA_PATH, "output_masks")
    pat_ids, label_paths = [], []
    for p in tqdm(img_paths):
        pat_id = p.split("/")[-1].split(".")[0]
        pat_ids.append(pat_id)
        path = os.path.join(base_path, f"{pat_id}.jpg")
        label_paths.append(path)

    # Do 75-25% train-test split
    train_X, test_X, train_y, test_y = train_test_split(
        img_paths, label_paths, test_size=0.25, random_state=params["random_seed"]
    )

    return train_X, test_X, train_y, test_y


def _parse_function_with_label(img_path, label_path):
    # Read input image
    img_string = tf.read_file(img_path)
    img = tf.cast(tf.image.decode_jpeg(img_string, channels=3), tf.float32)
    img /= tf.constant(255.0)
    # Read target mask
    mask_string = tf.read_file(label_path)
    label = tf.cast(tf.image.decode_jpeg(mask_string, channels=1), tf.float32)
    # Return dictionary of inputs and label
    input_dict = {"images": img}
    return input_dict, label


def _parse_function_without_label(img_path):
    # Read image
    img_string = tf.read_file(img_path)
    img = tf.cast(tf.image.decode_jpeg(img_string, channels=3), tf.float32)
    img /= tf.constant(255.0)
    # Return dictionary of inputs
    input_dict = {"images": img}
    return input_dict


def train_input_fn(img_paths, label_paths, batch_size=1):
    """Load and return batched examples."""

    # Convert inputs into tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, label_paths))
    dataset = dataset.map(_parse_function_with_label).prefetch(buffer_size=512)

    # Shuffle dataset, repeat, and make batches
    dataset = dataset.shuffle(buffer_size=2048).repeat().batch(batch_size)

    return dataset


def eval_input_fn(img_paths, label_paths, batch_size=1, shuffle=False):
    """Load and return batched examples for evaluation."""

    if label_paths is None:
        dataset = tf.data.Dataset.from_tensor_slices(img_paths)
        dataset = dataset.map(_parse_function_without_label).prefetch(buffer_size=512)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((img_paths, label_paths))
        dataset = dataset.map(_parse_function_with_label).prefetch(buffer_size=512)

    # Optionally shuffle and repeat
    if shuffle:
        dataset = dataset.shuffle(buffer_size=2048).repeat()

    # Repeat and batch dataset
    dataset = dataset.batch(batch_size)

    return dataset
