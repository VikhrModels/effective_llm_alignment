import logging
import sys

import datasets
import transformers
from transformers import TrainingArguments

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

datasets.utils.logging.set_verbosity(transformers.logging.INFO)
transformers.logging.set_verbosity_info()
transformers.logging.enable_default_handler()
transformers.logging.enable_explicit_format()