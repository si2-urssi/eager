#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import pandas as pd
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

###############################################################################


@dataclass
class KappaStats:
    PromisesSoftware: float


def calc_fleiss_kappa(
    data: Union[str, Path, pd.DataFrame],
) -> KappaStats:
    """
    Calculate the Fleiss kappa score as a metric for
    inter-rater reliability for the soft-search dataset.

    Parameters
    ----------
    data: Union[str, Path, pd.DataFrame]
        The path to the dataset (as CSV) or an in-memory DataFrame.

    Returns
    -------
    KappaStats
        The kappa statistics for the various columns.
    """
    # Assume the data is the soft-search labelled dataset
    if isinstance(data, (str, Path)):
        data = pd.read_csv(data)

    # Prep
    software = data[
        [
            "AnnotatorOnePromisesSoftware",
            "AnnotatorTwoPromisesSoftware",
            # "AnnotatorThreePromisesSoftware",
        ]
    ]

    # Aggregate
    agg_rater_software, _ = aggregate_raters(software)

    # Calc Kappa's and return
    return KappaStats(
        PromisesSoftware=fleiss_kappa(agg_rater_software),
    )
