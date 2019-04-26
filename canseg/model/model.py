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
            output_logits: float tensor [batch_size, width, height]
        """

        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):

            # Define model
            conv1, pool1 = self.conv_conv_pool(
                inputs["images"], [32, 32], name="conv_1"
            )
            # tf.summary.histograms('conv1',conv1)
            conv2, pool2 = self.conv_conv_pool(pool1, [64, 64], name="conv_2")
            conv3, pool3 = self.conv_conv_pool(pool2, [128, 128], name="conv_3")
            conv4, pool4 = self.conv_conv_pool(pool3, [256, 256], name="conv_4")
            conv5 = self.conv_conv_pool(pool4, [512, 512], name="conv_5", pool=False)

            up6 = self.upconv_concat(conv5, conv4, 256, name="conv_6")
            conv6 = self.conv_conv_pool(up6, [256, 256], name="conv_6", pool=False)

            up7 = self.upconv_concat(conv6, conv3, 128, name="conv_7")
            conv7 = self.conv_conv_pool(up7, [128, 128], name="conv_7", pool=False)

            up8 = self.upconv_concat(conv7, conv2, 64, name="conv_8")
            conv8 = self.conv_conv_pool(up8, [64, 64], name="conv_8", pool=False)

            up9 = self.upconv_concat(conv8, conv1, 32, name="conv_9")
            conv9 = self.conv_conv_pool(up9, [32, 32], name="conv_9", pool=False)

            conv10 = tf.layers.conv2d(
                conv9, 1, (1, 1), name="conv_10", activation=None, padding="same"
            )

        return conv10

    def conv_conv_pool(self, inputs, n_filters, name, pool=True, activation=tf.nn.relu):
        """Wrapper for double convolution and max pool block"""
        layer = inputs
        with tf.variable_scope(f"layer_{name}"):
            for i, filters in enumerate(n_filters):
                layer = tf.layers.conv2d(
                    layer,
                    filters,
                    kernel_size=(3, 3),
                    activation=activation,
                    padding="same",
                    # kernel_regularizer=tf.contrib.layers.l2_regularizer(),
                    name=f"conv_{i+1}",
                )
                layer = tf.layers.batch_normalization(
                    layer, training=self.train, name=f"bn_{i+1}"
                )
                layer = activation(layer, name=f"relu_{name}_{i+1}")

            if pool is False:
                return layer

            pool = tf.layers.max_pooling2d(
                layer, (2, 2), strides=(2, 2), name=f"pool_{name}"
            )
            return layer, pool

    def upconv_concat(self, input_1, input_2, n_filters, name):
        """Wrapper for upconvolution and concatenation block"""

        upconv = tf.layers.conv2d_transpose(
            input_1,
            filters=n_filters,
            kernel_size=2,
            strides=2,
            # kernel_regularizer=tf.contrib.layers.l2_regularizer(),
            name=f"upsample_{name}",
        )

        return tf.concat([upconv, input_2], axis=-1, name=f"concat_{name}")
