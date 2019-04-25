"""Defines default model parameters."""

from collections import defaultdict

BASE_PARAMS = defaultdict(
    lambda: None,  # set default value to None
    random_seed=0,
    # Optimizer parameters
    learning_rate=0.001,
    batch_size=32,
    train_steps=25000,
    # Eval parameters
    eval_steps=100,
    start_delay_secs=30,
    throttle_secs=600,
    # Data and output directories
    model_dir="ckpt/",
    train_data_dir="",
    eval_data_dir="",
    infer_data_dir="",
    pred_dir="pred/",
)
