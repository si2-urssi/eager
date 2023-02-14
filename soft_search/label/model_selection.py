#!/usr/bin/env python

import os
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split

from ..data import _DATA_DIR
from ..data.soft_search_2022 import (
    SoftSearch2022DatasetFields,
    load_soft_search_2022_training,
)
from ..seed import set_seed
from . import regex, semantic_logit, tfidf_logit, transformer
from .tfidf_logit import (
    ABSTRACT_SOURCE_TFIDF_LOGIT_PATH,
    OUTCOMES_SOURCE_TFIDF_LOGIT_PATH,
)
from .transformer import HUGGINGFACE_HUB_SOFT_SEARCH_MODEL

###############################################################################


def fit_and_eval_all_models(
    test_size: float = 0.2,
    seed: int = 0,
    archive: bool = False,
    train_transformer: bool = True,
    push_transformer: bool = False,
) -> pd.DataFrame:
    # Set global seed
    set_seed(seed)

    # Load core data
    data = load_soft_search_2022_training()

    # Run both models (prediction from abstract and prediction from outcomes)
    results = []
    for text_col in [
        SoftSearch2022DatasetFields.abstract_text,
        SoftSearch2022DatasetFields.project_outcomes,
    ]:
        # Subset / drop na for this text col
        subset = data.dropna(subset=[text_col])

        # Store the "predictive_source" column value
        predictive_source = {"predictive_source": text_col.replace("_", "-")}

        # Split the data
        train_df, test_df = train_test_split(
            subset,
            test_size=test_size,
            stratify=subset[SoftSearch2022DatasetFields.label],
        )

        # Run each model
        # Regex
        regex_metrics = regex.train(
            test_df,
            text_col=text_col,
            label_col=SoftSearch2022DatasetFields.label,
        )
        results.append(
            {
                **predictive_source,
                **regex_metrics.to_dict(),
            }
        )

        # TFIDF
        if text_col == SoftSearch2022DatasetFields.abstract_text:
            tfidf_output_path = ABSTRACT_SOURCE_TFIDF_LOGIT_PATH
        else:
            tfidf_output_path = OUTCOMES_SOURCE_TFIDF_LOGIT_PATH

        tfidf_logit_pipeline_path, _, tfidf_logit_metrics = tfidf_logit.train(
            train_df=train_df,
            test_df=test_df,
            text_col=text_col,
            label_col=SoftSearch2022DatasetFields.label,
            model_storage_path=tfidf_output_path,
        )
        results.append(
            {
                **predictive_source,
                **tfidf_logit_metrics.to_dict(),
            }
        )

        # Archive
        # We only save the tfidf-logit pipeline because it typically performs the best
        if archive:
            shutil.copy2(
                tfidf_logit_pipeline_path,
                _DATA_DIR,
            )

        # Semantic
        _, _, semantic_logit_metrics = semantic_logit.train(
            train_df=train_df,
            test_df=test_df,
            text_col=text_col,
            label_col=SoftSearch2022DatasetFields.label,
        )
        results.append(
            {
                **predictive_source,
                **semantic_logit_metrics.to_dict(),
            }
        )

        # Transformer
        if train_transformer:
            extra_training_args = {}
            if push_transformer:
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
                text_col=text_col,
                label_col=SoftSearch2022DatasetFields.label,
            )

            results.append(
                {
                    **predictive_source,
                    **transformer_metrics.to_dict(),
                }
            )

    # Create dataframe with metrics
    return (
        pd.DataFrame(results)
        .sort_values(by="f1", ascending=False)
        .reset_index(
            drop=True,
        )
    )
