# soft-search

[![Build Status](https://github.com/si2-urssi/eager/workflows/CI/badge.svg)](https://github.com/si2-urssi/eager/actions)
[![Documentation](https://github.com/si2-urssi/eager/workflows/Documentation/badge.svg)](https://si2-urssi.github.io/eager)

searching for software promises in grant applications

---

## Installation

**Stable Release:** `pip install soft-search`<br>
**Development Head:** `pip install git+https://github.com/si2-urssi/eager.git`

This repository contains the library code and the paper generation code
created for our paper [Searching for Software in NSF Awards](https://si2-urssi.github.io/eager/_static/paper.html).

### Abstract
Software is an important tool for scholarly work, but software produced for research is in many cases not easily identifiable or discoverable. A potential first step in linking research and software is software identification. In this paper we present two datasets to study the identification and production of research software. The first dataset contains almost 1000 human labeled annotations of software production from National Science Foundation (NSF) awarded research projects. We use this dataset to train models that  predict software production. Our second dataset is created by applying the trained predictive models across the abstracts and project outcomes reports for all NSF funded projects between the years of 2010 and 2023. The result is an inferred dataset of software production for over 150,000 NSF awards. We release the NSF-Soft-Search dataset to aid in identifying and understanding research software production: https://github.com/si2-urssi/eager


## The NSF-Soft-Search Inferred Dataset

Please download the 500MB NSF-Soft-Search Inferred dataset from
[Google Drive](https://drive.google.com/file/d/1k0jvs47bCWT18GHOMXY6EdG5MIDdCiM2/view?usp=share_link).

The dataset is shared as a `parquet` file and can be read in Python with
```python
import pandas as pd

nsf_soft_search = pd.read_parquet("nsf-soft-search-2022.parquet")
```

Please view the
[Parquet R Documentation](https://arrow.apache.org/docs/r/reference/read_parquet.html)
for information regarding reading the dataset in R.

## Quickstart

1. Load our best model (the "TF-IDF Vectorizer Logistic Regression Model")
2. Pull award abstract texts from the NSF API
3. Predict if the award will produce software using the abstract text for each award

```python
from soft_search.constants import NSFFields, NSFPrograms
from soft_search.label import (
  load_tfidf_logit_for_prediction_from_abstract,
  load_tfidf_logit_for_prediction_from_outcomes,
)
from soft_search.nsf import get_nsf_dataset

# Load the abstract model
pipeline = load_tfidf_logit_for_prediction_from_abstract()
# or load the outcomes model
# pipeline = load_tfidf_logit_for_prediction_from_outcomes()

# Pull data
data = get_nsf_dataset(
  start_date="2022-10-01",
  end_date="2023-01-01",
  program_name=NSFPrograms.Mathematical_and_Physical_Sciences,
  dataset_fields=[
    NSFFields.id_,
    NSFFields.abstractText,
    NSFFields.projectOutComesReport,
  ],
  require_project_outcomes_doc=False,
)

# Predict
data["prediction_from_abstract"] = pipeline.predict(data[NSFFields.abstractText])
print(data[["id", "prediction_from_abstract"]])

#           id prediction_from_abstract
# 0    2238468   software-not-predicted
# 1    2239561   software-not-predicted
```

### Annotated Training Data

```python
from soft_search.data import load_soft_search_2022_training

df = load_soft_search_2022_training()
```

### Reproducible Models

| predictive_source 	| model                  	| accuracy 	| precision 	| recall   	| f1       	|
|-------------------	|------------------------	|----------	|-----------	|----------	|----------	|
| project-outcomes  	| tfidf-logit            	| 0.744898 	| 0.745106  	| 0.744898 	| 0.744925 	|
| project-outcomes  	| fine-tuned-transformer 	| 0.673469 	| 0.637931  	| 0.770833 	| 0.698113 	|
| abstract-text     	| tfidf-logit            	| 0.673913 	| 0.673960  	| 0.673913 	| 0.673217 	|
| abstract-text     	| fine-tuned-transformer 	| 0.635870 	| 0.607843  	| 0.696629 	| 0.649215 	|
| project-outcomes  	| semantic-logit         	| 0.632653 	| 0.632568  	| 0.632653 	| 0.632347 	|
| abstract-text     	| semantic-logit         	| 0.630435 	| 0.630156  	| 0.630435 	| 0.629997 	|
| abstract-text     	| regex                  	| 0.516304 	| 0.514612  	| 0.516304 	| 0.513610 	|
| project-outcomes  	| regex                  	| 0.510204 	| 0.507086  	| 0.510204 	| 0.481559 	|

To train and evaluate all of our models you can run the following:

```bash
pip install soft-search

fit-and-eval-all-models
```

Also available directly in Python

```python
from soft_search.label.model_selection import fit_and_eval_all_models

results = fit_and_eval_all_models()
```

## Annotated Dataset Creation

1. We queried GitHub for repositories with references to NSF Awards.
  - We specifically queried for the terms: "National Science Foundation", "NSF Award",
    "NSF Grant", "Supported by the NSF", and "Supported by NSF". This script is available
    with the command `get-github-repositories-with-nsf-ref`. The code for the script is
    available at the following link:
    https://github.com/si2-urssi/eager/blob/main/soft_search/bin/get_github_repositories_with_nsf_ref.py
  - Note: the `get-github-repositories-with-nsf-ref` script produces a directory of CSV
    files. This is useful for paginated queries and protecting against potential crashes
    but the actual stored data in the repo (and the data we use going forward) is
    the a DataFrame with all of these chunks concatenated together and duplicate GitHub
    repositories removed.
  - Because the `get-github-repositories-with-nsf-ref` script depends on the returned
    data from GitHub themselves, we have archived the data produced by the original run
    of this script to the repository and made it available as follows:
    ```python
    from soft_search.data import load_github_repos_with_nsf_refs_2022

    data = load_github_repos_with_nsf_refs_2022()
    ```
2. We manually labeled each of the discovered repositories as "software"
   or "not software" and cleaned up the dataset to only include awards 
   which have a valid NSF Award ID.
  - A script was written to find all NSF Award IDs within a repositories README.md file
    and check that each NSF Award ID found was valid (if we could successfully query
    that award ID using the NSF API). Only valid NSF Award IDs were kept and therefore,
    only GitHub repositories which contained valid NSF Award IDs were kept in the
    dataset. This script is available with the command
    `find-nsf-award-ids-in-github-readmes-and-link`. The code for the script is
    available at the following link:
    https://github.com/si2-urssi/eager/blob/main/soft_search/bin/find_nsf_award_ids_in_github_readmes_and_link.py
  - A function was written to merge all of the manual annotations and the NSF Award IDs
    found. This function also stored the cleaned and prepared data to the project data
    directory. The code for this function is available at the following link:
    https://github.com/si2-urssi/eager/blob/main/soft_search/data/soft_search_2022.py#L143
  - The manually labeled, cleaned, prepared, and stored data is made available with the
    following code:
     ```python
     from soft_search.data import load_soft_search_2022_training

     data = load_soft_search_2022_training()
     ```
  - Prior to the manual annotation process, we conducted multiple rounds of
    annotation trials to ensure we had agreement on our labeling definitions.
    The final annotation trial results which resulted in an inter-rater
    reliability (Fleiss Kappa score) of 0.8918 (near perfect) is available
    via the following function:
    ```python
    from soft_search.data import load_soft_search_2022_training_irr

    data = load_soft_search_2022_training_irr()
    ```
    Additionally, the code for calculating the Fleiss Kappa Statistic
    is available at the following link:
    https://github.com/si2-urssi/eager/blob/main/soft_search/data/irr.py


## Documentation

For full package documentation please visit [si2-urssi.github.io/eager](https://si2-urssi.github.io/eager).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

**MIT License**
