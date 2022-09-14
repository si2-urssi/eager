#!/usr/bin/env python
# -*- coding: utf-8 -*-


class PredictionLabels:
    SoftwarePredicted = "software-predicted"
    SoftwareNotPredicted = "software-not-predicted"
    ModelPredicted = "model-predicted"
    ModelNotPredicted = "model-not-predicted"
    AlgorithmPredicted = "algorithm-predicted"
    AlgorithmNotPredicted = "algorithm-not-predicted"
    DatabasePredicted = "database-predicted"
    DatabaseNotPredicted = "database-not-predicted"


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
    Biological_Sciences = "BIO"
    Computer_and_Information_Science_and_Engineering = "CISE"
    Education_and_Human_Resources = "EHR"
    Engineering = "ENG"
    Environmental_Research_and_Education = "ERE"
    Geosciences = "GEO"
    Integrative_Activities = "OIA"
    International_Science_and_Engineering = "OISE"
    Mathematical_and_Physical_Sciences = "MPS"
    Social_Behavioral_and_Economic_Sciences = "SBE"
    Technology_Innovation_and_Partnerships = "TIP"
