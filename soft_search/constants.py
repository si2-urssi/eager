#!/usr/bin/env python
# -*- coding: utf-8 -*-


class SoftwareOutcomes:
    SoftwarePredicted = "software-outcome-predicted"
    SoftwareNotPredicted = "software-outcome-not-predicted"


class NSFFields:
    """
    Fields that can be provided to the `get_nsf_dataset` function `dataset_fields`
    parameter.

    Examples
    --------
    >>> soft_search.nsf.get_nsf_dataset(
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
