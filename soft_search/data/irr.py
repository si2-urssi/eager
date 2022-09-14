#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .soft_search_2022 import SoftSearch2022IRRDatasetFields, load_soft_search_2022_irr

###############################################################################


@dataclass
class KappaStats:
    PromisesSoftware: float
    PromisesModel: float
    PromisesAlgorithm: float
    PromisesDatabase: float


def calc_cohens_kappa(
    data: Union[str, Path, pd.DataFrame],
) -> KappaStats:
    """
    Calculate the Cohen's Kappa score as a metric for
    inter-rater reliability for the soft-search dataset.

    Parameters
    ----------
    data: Union[str, Path, pd.DataFrame]
        The path to the dataset (as CSV) or an in-memory DataFrame.

    Returns
    -------
    KappaStats
        The kappa statistics for the various columns.

    See Also
    --------
    soft_search.data.soft_search_2022.load_soft_search_2022_irr
        The function to load the IRR data.

    Notes
    -----
    See interpretation of Cohen's Kappa Statistic:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3900052/table/t3-biochem-med-22-3-276-4/?report=objectonly
    """
    # Assume the data is the soft-search labelled dataset
    if isinstance(data, (str, Path)):
        data = pd.read_csv(data)

    # Prep comparisons
    anno1 = data.loc[data["AnnotatorNum"] == 1]
    anno2 = data.loc[data["AnnotatorNum"] == 2]

    # Aggregate
    results = {}
    for col in [
        SoftSearch2022IRRDatasetFields.PromisesSoftware,
        SoftSearch2022IRRDatasetFields.PromisesModel,
        SoftSearch2022IRRDatasetFields.PromisesAlgorithm,
        SoftSearch2022IRRDatasetFields.PromisesDatabase,
    ]:
        results[col] = cohen_kappa_score(
            anno1[col].values,
            anno2[col].values,
        )

    # Calc Kappa's and return
    return KappaStats(**results)


def print_irr_summary_stats() -> None:
    """
    Print useful statistics and summary stats using the stored
    inter-rater reliability data.

    Prints:
    * Cohen's Kappa Statistic for each potential model
    * Mean number of examples for each label between the two annotators
    * The rows which differ between the two annotators
    """
    # Load IRR data
    data = load_soft_search_2022_irr()

    # Get Cohen's Kappa Stats
    kappa = calc_cohens_kappa(data)

    def _iterrpreted_score(v: float) -> str:
        if v < 0.2:
            return "No agreement"
        if v >= 0.2 and v < 0.4:
            return "Minimal agreement"
        if v >= 0.4 and v < 0.6:
            return "Weak agreement"
        if v >= 0.6 and v < 0.8:
            return "Moderate agreement"
        if v >= 0.8 and v < 0.9:
            return "Strong agreement"
        return "Almost perfect agreement"

    # Get annotator subsets
    anno1 = data.loc[data["AnnotatorNum"] == 1]
    anno2 = data.loc[data["AnnotatorNum"] == 2]

    # Print stats
    print("Inter-Rater Reliability Statistics and Data Summary:")
    print("=" * 80)
    for cat in [
        SoftSearch2022IRRDatasetFields.PromisesSoftware,
        SoftSearch2022IRRDatasetFields.PromisesModel,
        SoftSearch2022IRRDatasetFields.PromisesAlgorithm,
        SoftSearch2022IRRDatasetFields.PromisesDatabase,
    ]:
        # Get score for category
        cat_score = getattr(kappa, cat)
        # Create mini-df for row-wise comparisons
        subset = pd.DataFrame(
            {
                SoftSearch2022IRRDatasetFields.AwardNumber: (
                    anno1[SoftSearch2022IRRDatasetFields.AwardNumber]
                ),
                "anno1": anno1[cat].values,
                "anno2": anno2[cat].values,
            }
        )
        # Generate counts
        anno1_counts = subset["anno1"].value_counts().to_frame().T
        anno2_counts = subset["anno2"].value_counts().to_frame().T
        mean_counts = pd.concat([anno1_counts, anno2_counts]).mean(axis=0).to_frame().T
        diff = subset.loc[subset["anno1"] != subset["anno2"]]

        # Run print
        print(f"{cat}: {cat_score} ({_iterrpreted_score(cat_score)})")
        print()
        print("Average of Both Annotators Value Counts:")
        print(mean_counts)
        print()
        print("Differing Labels:")
        print(diff)
        print("-" * 80)
