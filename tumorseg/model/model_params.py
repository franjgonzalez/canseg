"""Defines default model parameters."""

from collections import defaultdict

BASE_PARAMS = defaultdict(
    lambda: None,  # set default value to None
    random_seed=0,
    # Optimizer parameters
    learning_rate=0.00001,
    batch_size=64,
    train_steps=5000,
    # Eval parameters
    eval_steps=100,
    start_delay_secs=30,
    throttle_secs=300,
    # Data and output directories
    model_dir="ckpt/",
    pred_dir="pred/",
)
