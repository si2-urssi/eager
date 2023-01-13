#!/usr/bin/env python


class PredictionLabels:
    SoftwarePredicted = "software-predicted"
    SoftwareNotPredicted = "software-not-predicted"


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
    awardeeName = "awardeeName"  # noqa: N815
    awardeeStateCode = "awardeeStateCode"  # noqa: N815
    fundsObligatedAmt = "fundsObligatedAmt"  # noqa: N815
    piFirstName = "piFirstName"  # noqa: N815
    piLastName = "piLastName"  # noqa: N815
    publicAccessMandate = "publicAccessMandate"  # noqa: N815
    date = "date"
    title = "title"
    abstractText = "abstractText"  # noqa: N815
    projectOutComesReport = "projectOutComesReport"  # noqa: N815
    piEmail = "piEmail"  # noqa: N815
    publicationResearch = "publicationResearch"  # noqa: N815
    publicationConference = "publicationConference"  # noqa: N815
    startDate = "startDate"  # noqa: N815
    expDate = "expDate"  # noqa: N815


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


DEFAULT_SEMANTIC_EMBEDDING_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
