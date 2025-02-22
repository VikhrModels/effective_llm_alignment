import logging
import sys

import datasets
import transformers
from transformers import TrainingArguments


def setup_logging(logger, training_args: TrainingArguments):
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.logging.set_verbosity_info()
    else:
        transformers.logging.set_verbosity_error()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
