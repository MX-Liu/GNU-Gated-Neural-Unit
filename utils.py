
import logging
import sys
import torch
import numpy as np
import random

def setup_logger(log_file_path):
    logger = logging.getLogger(log_file_path)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def format_confusion_matrix(cm, class_names):
    header = " " * 6 + " ".join([f"{name:<10}" for name in class_names])
    lines = [header, "-" * len(header)]
    for i, row in enumerate(cm):
        line = f"{class_names[i]:<5} " + " ".join([f"{val:<10}" for val in row])
        lines.append(line)
    return "\n" + "\n".join(lines)
