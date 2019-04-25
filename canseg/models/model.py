"""Define model."""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf

from ..utils import data_utils


class Model(object):
    """U-Net model for segmentation."""

    def __init__(self, params, train):
        """Initialize model.

        Args:
            params: hyperparameter object
            train: bool whether the model is in training mode
        """
        self.train = train
        self.params = params

        # Define initializers
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)

    def __call__(self, inputs):
        """Calculate target image mask.

        Args:
            inputs: float tensor [batch_size, width, height, num_modalities]

        Returns:
            output_masks: float tensor [batch_size, width, height]
        """

        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):

            # Define model
            pass
