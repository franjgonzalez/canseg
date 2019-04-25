"""Train and evaluate U-Net model."""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import argparse

import pickle
import numpy as np
import pandas as pd

import tensorflow as tf

from .model import model as unet_model
from .model import model_params
from .utils import data_utils


def model_fn(features, labels, mode, params):
    """Defines EstimatorSpec passed to Estimator.

    Args:
        features: dict of input features
        labels: target mask
        mode: TRAIN | EVAL | PREDICT
        params: hyperparameter object
    """

    # Create model and get output mask
    model = unet_model.Model(params, mode == tf.estimator.ModeKeys.TRAIN)
    output_masks = model(inputs=features)

    ## Predict
    # Process output_masks
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {}

    # Compute loss
    loss = None
    # Compute evaluation metrics
    accuracy = None
    metrics = {"accuracy": accuracy}
    # Add metrics to summary

    ## Evalaute
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    ## Train
    assert mode == tf.estimator.ModeKeys.TRAIN

    # Define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])

    # Define train op
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def train(estimator, params):
    """Train model.

    Args:
        estimator: estimator object
        params: hyperparameter object
    """

    # Fetch the data
    train_X, _, train_y, _ = data_utils.get_data()

    # Train the model
    estimator.train(
        input_fn=lambda: data_utils.train_input_fn(
            img_paths=train_X, label_paths=train_y, batch_size=params["batch_size"]
        ),
        steps=params["train_steps"],
    )


def evaluate(estimator, params):
    """Run model on validation set.

    Args:
        estimator: estimator object
        params: hyperparameter object
    """

    # Fetch the data
    _, test_X, _, test_y = data_utils.get_data()

    # Run evaluation
    eval_result = estimator.evaluate(
        input_fn=lambda: data_utils.eval_input_fn(
            img_paths=train_X,
            label_paths=train_y,
            batch_size=params["batch_size"],
            shuffle=True,
        )
    )


def train_and_eval(estimator, params):
    """Train and evaluate model.

    Args:
        estimator: estimator object
        params: hyperparameter object
    """

    # Fetch the data
    train_X, test_X, train_y, test_y = data_utils.get_data()

    # Define train and eval spec
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: data_utils.train_input_fn(
            img_paths=train_X, label_paths=train_y, batch_size=params["batch_size"]
        ),
        max_steps=params["train_steps"],
    )
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: data_utils.eval_input_fn(
            img_paths=train_X,
            label_paths=train_y,
            batch_size=params["batch_size"],
            shuffle=True,
        ),
        steps=params["eval_steps"],
        start_delay_secs=params["start_delay_secs"],
        throttle_secs=params["throttle_secs"],
    )

    # Train and evaluate model
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def predict(estimator, params):
    """Make predictions using trained model.

    Args:
        estimator: estimator object
        params: hyperparameter object
    """

    # Fetch the data
    _, test_X, _, _ = data_utils.get_data()

    # Make predictions
    predictions = estimator.predict(
        input_fn=lambda: data_utils.eval_input_fn(
            img_paths=train_X,
            label_paths=train_y,
            batch_size=params["batch_size"],
            shuffle=False,
        )
    )

    # Save predictions


def main(argv):
    """Main entry point."""

    # Parse arguments
    args = parser.parse_args(argv[1:])

    # Get the base parameters
    params = model_params.BASE_PARAMS

    # Check or make model directory path
    if not os.path.exists(params["model_dir"]):
        os.makedirs(params["model_dir"])

    # Build model
    estimator = tf.estimator.Estimator(
        model_fn=model_fn, params=params, model_dir=params["model_dir"]
    )

    # Train, eval, or predict
    if args.train:
        train(estimator, params)

    if args.eval:
        evaluate(estimator, params)

    if args.train_and_eval:
        train_and_eval(estimator, params)

    if args.predict:
        predict(estimator, params)


if __name__ == "__main__":

    # Create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--eval", dest="eval", action="store_true")
    parser.add_argument("--train_and_eval", dest="train_and_eval", action="store_true")
    parser.add_argument("--predict", dest="predict", action="store_true")

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
