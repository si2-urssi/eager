#!/usr/bin/env python

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split

from soft_search.data.soft_search_2022 import (
    SoftSearch2022DatasetFields,
    load_soft_search_2022,
)
from soft_search.label import semantic_logit
from soft_search.metrics import EvaluationMetrics

###############################################################################


def test_semantic_logit_train() -> None:
    # Load data (and sample for fast tests)
    data = load_soft_search_2022().sample(n=40)

    # Split
    train_df, test_df = train_test_split(
        data,
        stratify=data[SoftSearch2022DatasetFields.label],
    )

    # Train and get eval metrics
    model_path, logit, text_transformer, eval_metrics = semantic_logit.train(
        train_df=train_df,
        test_df=test_df,
    )

    # Basic assertions
    assert model_path.resolve(strict=True)
    assert isinstance(logit, LogisticRegressionCV)
    assert isinstance(eval_metrics, EvaluationMetrics)

    # Asserts type and that we can encode again
    assert text_transformer.encode("hello my name is eva") is not None
