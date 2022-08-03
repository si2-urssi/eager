#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from datetime import datetime

from typing import Union

import logging

###############################################################################

log = logging.getLogger(__name__)

###############################################################################
# Constants

class NSFFields:
    """
    Fields that can be provided to the `get_nsf_dataset` function `dataset_fields`
    parameter.

    Usage
    -----
    >>> get_nsf_dataset(
    ...     start_date="2017-01-01",
    ...     dataset_fields=[NSFFields.id_, NSFFields.abstractText],
    ... )
    """
    id_ = "id"
    agency = "agency"
    awardeeName = "awardeeName"
    awardeeStateCode = "awardeeStateCode"
    fundsObligatedAmt = "fundsObligatedAmt"
    piFirstName = "piFirstName"
    piLastName = "piLastName"
    publicAccessMandate = "publicAccessMandate"
    date = "date"
    title = "title"
    abstractText = "abstractText"
    projectOutComesReport = "projectOutComesReport"
    piEmail = "piEmail"
    publicationResearch = "publicationResearch"
    publicationConference = "publicationConference"
    startDate = "startDate"
    expDate = "expDate"

ALL_NSF_FIELDS = [getattr(NSFFields, a) for a in dir(NSFFields) if "__" not in a]

class NSFPrograms:
    BIO = "BIO"

API_URI = (
    "https://api.nsf.gov/services/v1/awards.json?"
    "fundProgramName={program_name}"
    "&agency={agency}"
    "&dateStart={start_date}"
    "&transType={transaction_type}"
    "&printFields={dataset_fields}"
    "&projectOutcomesOnly={require_project_outcomes_doc}"
    "&offset={current_offset}"
)


###############################################################################


def get_nsf_dataset(start_date: Union[str, datetime], program_name: str = NSFPrograms.BIO, agency: str = "NSF", transaction_type: str = "Grant", dataset_fields: List[str] = ALL_NSF_FIELDS, require_project_outcomes_doc: bool = True,) -> pd.DataFrame:
    """
    Return the count of characters in the provided string.

    Parameters
    ----------
    string: str
        The string to get the count of characters for.

    Returns
    -------
    int
        The count of characters in the string.
    """
    return pd.DataFrame()
