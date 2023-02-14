#!/usr/bin/env python

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from soft_search.data.soft_search_2022 import (
    SoftSearch2022DatasetFields,
    load_soft_search_2022_training,
)
from soft_search.label import semantic_logit
from soft_search.metrics import EvaluationMetrics

###############################################################################


def test_semantic_logit_train() -> None:
    # Load data (and sample for fast tests)
    data = load_soft_search_2022_training().sample(n=40)

    # Split
    train_df, test_df = train_test_split(
        data,
        stratify=data[SoftSearch2022DatasetFields.label],
    )

    # Train and get eval metrics
    model_path, pipeline, eval_metrics = semantic_logit.train(
        train_df=train_df,
        test_df=test_df,
    )

    # Basic assertions
    assert model_path.resolve(strict=True)
    assert isinstance(pipeline, Pipeline)
    assert isinstance(eval_metrics, EvaluationMetrics)

    assert pipeline.predict(["this will definitely produce software"]) is not None
