import numpy as np
import random
import torch

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_error(*message):
    print('\033[91m', 'ERROR ', *message, '\033[0m')
    raise RuntimeError


def print_ok(*message):
    print('\033[92m', *message, '\033[0m')


def print_warning(*message):
    print('\033[93m', *message, '\033[0m')


def print_info(*message):
    print('\033[96m', *message, '\033[0m')


def print_white(*message):
    print('\033[37m', *message, '\033[0m')
