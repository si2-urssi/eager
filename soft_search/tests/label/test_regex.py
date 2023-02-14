#!/usr/bin/env python

import pandas as pd

from soft_search.constants import PredictionLabels
from soft_search.data.soft_search_2022 import load_soft_search_2022_training
from soft_search.label import regex
from soft_search.metrics import EvaluationMetrics

###############################################################################


def test_regex_train() -> None:
    # Load data (and sample for fast tests)
    data = load_soft_search_2022_training().sample(n=40)

    # "Train" and get eval metrics
    metrics = regex.train(data)

    assert isinstance(metrics, EvaluationMetrics)


def test_regex_label() -> None:
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
        PredictionLabels.SoftwarePredicted,
        PredictionLabels.SoftwareNotPredicted,
        PredictionLabels.SoftwareNotPredicted,
        PredictionLabels.SoftwarePredicted,
    ]

    # Run and compare
    df = regex.label(df)
    assert regex.REGEX_LABEL_COL in df.columns
    assert df[regex.REGEX_LABEL_COL].tolist() == expected_values
