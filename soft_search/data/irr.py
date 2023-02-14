#!/usr/bin/env python

from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

from .soft_search_2022 import (
    SoftSearch2022IRRDatasetFields,
    load_soft_search_2022_training_irr,
)

###############################################################################


def calc_fleiss_kappa(
    data: Union[str, Path, pd.DataFrame],
) -> float:
    """
    Calculate the Fleiss Kappa score as a metric for
    inter-rater reliability for the soft-search dataset.

    Parameters
    ----------
    data: Union[str, Path, pd.DataFrame]
        The path to the dataset (as parquet) or an in-memory DataFrame.

    Returns
    -------
    float
        The kappa statistic for the data.

    See Also
    --------
    soft_search.data.soft_search_2022.load_soft_search_2022_training_irr
        The function to load the IRR data.

    Notes
    -----
    See interpretation of Fleiss Kappa Statistic:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3900052/table/t3-biochem-med-22-3-276-4/?report=objectonly
    """
    # Assume the data is the soft-search labelled dataset
    if isinstance(data, (str, Path)):
        data = pd.read_parquet(data)

    # Sort by link to have consistent order
    sorted_data = data.sort_values(
        by=[
            SoftSearch2022IRRDatasetFields.github_link,
        ],
    )

    # Make a frame of _just_ the annotation
    annotations: List[pd.Series] = []
    for annotator_label in sorted_data[
        SoftSearch2022IRRDatasetFields.annotator
    ].unique():
        annotations.append(
            sorted_data.loc[
                sorted_data[SoftSearch2022IRRDatasetFields.annotator] == annotator_label
            ][SoftSearch2022IRRDatasetFields.include_in_definition].values
        )

    # Annotations merged together and ensured to be in subject as rows order
    annotations = pd.DataFrame(annotations).T

    # Aggregate
    agg_raters, _ = aggregate_raters(annotations)

    # Calc Kappa's and return
    return fleiss_kappa(agg_raters)


def print_irr_summary_stats(  # noqa: C901
    do_print: bool = True,
) -> float:
    """
    Print useful statistics and summary stats using the stored
    inter-rater reliability data.

    Prints:
    * Cohen's Kappa Statistic for each potential model
    * Mean number of examples for each label between the two annotators
    * The rows which differ between the two annotators

    Parameters
    ----------
    do_print: bool
        Should this function actually print the table
        Default: True (yes, print the table)

    Returns
    -------
    float
        The overall Fliess Kappa statistic.
    """
    # Load IRR data
    data = load_soft_search_2022_training_irr()

    # Sort by link to have consistent order
    sorted_data = data.sort_values(
        by=[
            SoftSearch2022IRRDatasetFields.github_link,
        ],
    )

    # Get Cohen's Kappa Stats
    kappa = calc_fleiss_kappa(sorted_data)

    def _iterrpreted_score(v: float) -> str:
        if v < 0:
            return "No agreement"
        if v < 0.2:
            return "Poor agreement"
        if v >= 0.2 and v < 0.4:
            return "Fair agreement"
        if v >= 0.4 and v < 0.6:
            return "Moderate agreement"
        if v >= 0.6 and v < 0.8:
            return "Substantial agreement"
        return "Almost perfect agreement"

    # Get just annotation series
    annotations: Dict[str, pd.Series] = {}
    link_series: Optional[pd.Series] = None
    for annotator_label in sorted_data[
        SoftSearch2022IRRDatasetFields.annotator
    ].unique():
        annotator_subset = sorted_data.loc[
            sorted_data[SoftSearch2022IRRDatasetFields.annotator] == annotator_label
        ].reset_index()
        annotations[annotator_label] = annotator_subset[
            SoftSearch2022IRRDatasetFields.include_in_definition
        ]
        if link_series is None:
            link_series = annotator_subset[SoftSearch2022IRRDatasetFields.github_link]

    # Each annotator column values as columns
    annotations_df = pd.DataFrame(
        {
            SoftSearch2022IRRDatasetFields.github_link: link_series,
            **annotations,
        },
    )

    # Print stats
    if do_print:
        print("Inter-Rater Reliability Statistics and Data Summary:")
        print("=" * 80)
        annotator_pairs = combinations(
            sorted_data[SoftSearch2022IRRDatasetFields.annotator].unique(), 2
        )

        # Run print
        print(
            f"{SoftSearch2022IRRDatasetFields.include_in_definition}: "
            f"{kappa} ({_iterrpreted_score(kappa)})"
        )
        print()

        # Get all possible diffs then drop duplicates
        diffs = []
        for anno_one, anno_two in annotator_pairs:
            diffs.append(
                annotations_df.loc[annotations_df[anno_one] != annotations_df[anno_two]]
            )
        diff_df = pd.concat(diffs)
        diff_df = diff_df.drop_duplicates(subset=["github_link"]).reset_index()

        print("Differing Labels:")
        print(diff_df)
        print()
        print("-" * 80)
        diff_df.to_csv("irr-diffs.csv", index=False)

    return kappa
