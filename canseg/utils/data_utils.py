"""Data utility functions."""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import glob

import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf

DATA_PATH = os.path.join(os.getenv("HOME"), ".keras/datasets/TCGA-GBM/processed")


def get_data():

    # Get input image and label paths
    img_paths = glob.glob(os.path.join(DATA_PATH, "input_images/*.jpg"))
    base_path = os.path.join(DATA_PATH, "output_masks")
    label_paths = []
    for p in img_paths:
        pat_id = p.split("/")[-1].split(".")[0]
        label_paths.append(os.path.join(base_path, f"{pat_id}.npz"))

    # Do 75-25% train-test split
    train_X, test_X, train_y, test_y = train_test_split(
        img_paths, label_paths, test_size=0.25, random_state=params["random_seed"]
    )

    return train_X, test_X, train_y, test_y


def train_input_fn(img_paths, label_paths, batch_size=1):
    """Load and return batched examples."""
    assert buffer_size is not None

    # Convert inputs into tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, label_paths))
    dataset = dataset.map(_parse_function_with_label).prefetch(buffer_size=512)

    # Shuffle dataset, repeat, and make batches
    dataset = dataset.shuffle(buffer_size=2048).repeat().batch(batch_size)

    return dataset


def eval_input_fn(img_paths, label_paths, batch_size=1, shuffle=False):
    """Load and return batched examples for evaluation."""

    if labels is None:
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
