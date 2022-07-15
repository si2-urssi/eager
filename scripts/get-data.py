#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import pandas as pd

DATASET_FIELDS = [
    "id",
    "agency",
    "awardeeName",
    "awardeeStateCode",
    "fundsObligatedAmt",
    "piFirstName",
    "piLastName",
    "publicAccessMandate",
    "date",
    "title",
    "abstractText",
]
JOINED_FIELDS = ",".join(DATASET_FIELDS)

###############################################################################

# Iteratively get awards
current_offset = 0
award_dfs = []
while True:
    # Request with offset
    response = requests.get(
        f"https://api.nsf.gov/services/v1/awards.json?"
        f"fundProgramName=BIO"
        f"&agency=NSF"
        f"&dateStart=01/01/2019"
        f"&transType=Grant"
        f"&printFields={JOINED_FIELDS}",
        f"&offset={current_offset}",
    )

    # Parse awards
    awards = pd.DataFrame(response.json()["response"]["award"])
    award_dfs.append(awards)

    # Check length and finish
    if len(awards) < 25:
        break

    current_offset += 25
    print(f"Awards gathered: {current_offset}")

# Concat all awards
all_awards = pd.concat(award_dfs, ignore_index=True)
print(all_awards)

# Store
all_awards.to_parquet("data/nsf-awards.parquet")
all_awards.to_csv("data/nsf-awards.csv", index=False)
