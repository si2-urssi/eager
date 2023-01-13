#!/usr/bin/env python

import pandas as pd
from sklearn.model_selection import train_test_split

from ..data.soft_search_2022 import SoftSearch2022DatasetFields, load_soft_search_2022
from ..seed import set_seed
from . import regex, semantic_logit, tfidf_logit, transformer

###############################################################################


def fit_and_eval_all_models(test_size: float = 0.2, seed: int = 0) -> pd.DataFrame:
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
    _, _, _, tfidf_logit_metrics = tfidf_logit.train(
        train_df=train_df,
        test_df=test_df,
    )
    _, _, _, semantic_logit_metrics = semantic_logit.train(
        train_df=train_df,
        test_df=test_df,
    )
    _, _, _, transformer_metrics = transformer.train(
        train_df=train_df,
        test_df=test_df,
    )

    # Create dataframe with metrics
    return pd.DataFrame(
        [
            regex_metrics.to_dict(),
            tfidf_logit_metrics.to_dict(),
            semantic_logit_metrics.to_dict(),
            transformer_metrics.to_dict(),
        ]
    ).sort_values(by="f1", ascending=False)
