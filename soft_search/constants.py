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
    fundProgramName = "fundProgramName"  # noqa: N815
    awardAgencyCode = "awardAgencyCode"  # noqa: N815
    fundAgencyCode = "fundAgencyCode"  # noqa: N815
    parentDunsNumber = "parentDunsNumber"  # noqa: N815
    primaryProgram = "primaryProgram"  # noqa: N815
    startDate = "startDate"  # noqa: N815
    expDate = "expDate"  # noqa: N815
    rpp = "rpp"
    awardeeCity = "awardeeCity"  # noqa: N815
    awardeeCountryCode = "awardeeCountryCode"  # noqa: N815
    awardeeCounty = "awardeeCounty"  # noqa: N815
    awardeeDistrictCode = "awardeeDistrictCode"  # noqa: N815
    awardeeZipCode = "awardeeZipCode"  # noqa: N815
    cfdaNumber = "cfdaNumber"  # noqa: N815
    coPDPI = "coPDPI"  # noqa: N815
    estimatedTotalAmt = "estimatedTotalAmt"  # noqa: N815
    dunsNumber = "dunsNumber"  # noqa: N815
    pdPIName = "pdPIName"  # noqa: N815
    perfCity = "perfCity"  # noqa: N815
    perfCountryCode = "perfCountryCode"  # noqa: N815
    perfCounty = "perfCounty"  # noqa: N815
    perfDistrictCode = "perfDistrictCode"  # noqa: N815
    perfLocation = "perfLocation"  # noqa: N815
    perfStateCode = "perfStateCode"  # noqa: N815
    perfZipCode = "perfZipCode"  # noqa: N815
    poName = "poName"  # noqa: N815
    transType = "transType"  # noqa: N815
    awardee = "awardee"
    poPhone = "poPhone"  # noqa: N815
    poEmail = "poEmail"  # noqa: N815
    awardeeAddress = "awardeeAddress"  # noqa: N815
    perfAddress = "perfAddress"  # noqa: N815
    piPhone = "piPhone"  # noqa: N815


ALL_NSF_FIELDS = [getattr(NSFFields, a) for a in dir(NSFFields) if "__" not in a]


class NSFPrograms:
    Biological_Sciences = "BIO"
    Computer_and_Information_Science_and_Engineering = "CISE"
    Education_and_Human_Resources = "EHR"
    Engineering = "ENG"
    Geosciences = "GEO"
    Integrative_Activities = "OIA"
    International_Science_and_Engineering = "OISE"
    Mathematical_and_Physical_Sciences = "MPS"
    Social_Behavioral_and_Economic_Sciences = "SBE"
    Technology_Innovation_and_Partnerships = "TIP"


ALL_NSF_PROGRAMS = [getattr(NSFPrograms, a) for a in dir(NSFPrograms) if "__" not in a]


CFDA_NUMBER_TO_PROGRAM_LUT = {
    "47.041": NSFPrograms.Engineering,
    "47.049": NSFPrograms.Mathematical_and_Physical_Sciences,
    "47.050": NSFPrograms.Geosciences,
    "47.070": NSFPrograms.Computer_and_Information_Science_and_Engineering,
    "47.074": NSFPrograms.Biological_Sciences,
    "47.075": NSFPrograms.Social_Behavioral_and_Economic_Sciences,
    "47.076": NSFPrograms.Education_and_Human_Resources,
    "47.079": NSFPrograms.International_Science_and_Engineering,
    "47.083": NSFPrograms.Integrative_Activities,
    "47.084": NSFPrograms.Technology_Innovation_and_Partnerships,
}


NSF_PROGRAM_TO_CFDA_NUMBER_LUT = {
    code: number for number, code in CFDA_NUMBER_TO_PROGRAM_LUT.items()
}


DEFAULT_SEMANTIC_EMBEDDING_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
