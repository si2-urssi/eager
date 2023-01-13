#!/usr/bin/env python

import os
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split

from ..data import _DATA_DIR
from ..data.soft_search_2022 import SoftSearch2022DatasetFields, load_soft_search_2022
from ..seed import set_seed
from . import regex, semantic_logit, tfidf_logit, transformer
from .transformer import HUGGINGFACE_HUB_SOFT_SEARCH_MODEL

###############################################################################


def fit_and_eval_all_models(
    test_size: float = 0.2,
    seed: int = 0,
    archive: bool = False,
    train_transformer: bool = True,
) -> pd.DataFrame:
    # Set global seed
    set_seed(seed)

    # Load core data
    data = load_soft_search_2022()

    # Split the data
    train_df, test_df = train_test_split(
        data,
        test_size=test_size,
        stratify=data[SoftSearch2022DatasetFields.label],
    )

    # Run each model
    regex_metrics = regex.train(test_df)
    tfidf_logit_pipeline_path, _, tfidf_logit_metrics = tfidf_logit.train(
        train_df=train_df,
        test_df=test_df,
    )
    _, _, semantic_logit_metrics = semantic_logit.train(
        train_df=train_df,
        test_df=test_df,
    )

    # Store all metrics
    metrics = [regex_metrics, tfidf_logit_metrics, semantic_logit_metrics]

    if train_transformer:
        extra_training_args = {}
        if archive:
            extra_training_args = {
                "push_to_hub": True,
                "hub_model_id": HUGGINGFACE_HUB_SOFT_SEARCH_MODEL,
                "hub_strategy": "end",
                "hub_token": os.environ["HUGGINGFACE_TOKEN"],
            }

        _, _, _, transformer_metrics = transformer.train(
            train_df=train_df,
            test_df=test_df,
            extra_training_args=extra_training_args,
        )

        metrics.append(transformer_metrics)

    # Archive
    # We only save the tfidf-logit pipeline because the semantic pipeline
    if archive:
        shutil.copy2(
            tfidf_logit_pipeline_path,
            _DATA_DIR,
        )

    # Create dataframe with metrics
    return pd.DataFrame(
        [m.to_dict() for m in metrics],
    ).sort_values(by="f1", ascending=False)
