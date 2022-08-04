#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import shutil

import numpy as np
import pandas as pd
import torch

from soft_search.constants import SoftwareOutcomes
from soft_search.data import load_joined_soft_search_2022
from soft_search.label import transformer

###############################################################################


def test_transformer_train() -> None:
    # Load data
    df = load_joined_soft_search_2022()

    # Shorten dataset so training doesn't take long
    df = df[:10]

    # Train
    model = transformer.train(df, model_storage_dir="test-output-transformer/")

    # Delete the model
    shutil.rmtree(model)


def test_transformer_label() -> None:
    # Set a bunch of seeds for a semblance of reproducibility
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)

    # Starting DataFrame
    df = pd.DataFrame(
        {
            "text": [
                "software",
                "hello",
                "world",
                "algorithm",
            ],
        },
    )
    # Expected values based off above abstractText column
    expected_values = [
        SoftwareOutcomes.SoftwareNotPredicted,  # Lol, need more data / a better model
        SoftwareOutcomes.SoftwareNotPredicted,
        SoftwareOutcomes.SoftwareNotPredicted,
        SoftwareOutcomes.SoftwareNotPredicted,  # Lol, need more data / a better model
    ]

    # Run and compare
    df = transformer.label(df)
    assert transformer.TRANSFORMER_LABEL_COL in df.columns
    assert df[transformer.TRANSFORMER_LABEL_COL].tolist() == expected_values
