#!/usr/bin/env python

import random

import numpy as np
import torch

###############################################################################


def set_seed(seed: int = 0) -> None:
    # Set a bunch of seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
