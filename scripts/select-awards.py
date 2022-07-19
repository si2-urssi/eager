#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import re

###############################################################################

SOFTWARE_LIKE_PATTERNS = (
    r".*(?:software(?:\s(?:code|tool|suite|program|application|framework)s?|"
    r"(?:binar|librar)(?:y|ies))?|algorithms?|tools?).*"
)
COMPILED_SOFTWARE_LIKE_PATTERNS = re.compile(SOFTWARE_LIKE_PATTERNS)

###############################################################################

all_awards = pd.read_parquet("data/nsf-awards.parquet")
print(f"Raw dataset length: {len(all_awards)}")

matching_rows = []
for i, row in all_awards.iterrows():
    match_or_none = re.match(COMPILED_SOFTWARE_LIKE_PATTERNS, row.abstractText)
    if match_or_none:
        matching_rows.append(row)

filtered_dataset = pd.DataFrame(matching_rows)
print(f"Filtered dataset length: {len(filtered_dataset)}")

# Store
filtered_dataset.to_parquet("data/filtered-nsf-awards.parquet")
filtered_dataset.to_csv("data/filtered-nsf-awards.csv", index=False)