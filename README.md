# soft-search

[![Build Status](https://github.com/si2-urssi/eager/workflows/CI/badge.svg)](https://github.com/si2-urssi/eager/actions)
[![Documentation](https://github.com/si2-urssi/eager/workflows/Documentation/badge.svg)](https://si2-urssi.github.io/eager)

searching for software promises in grant applications

---

## Installation

**Stable Release:** `pip install soft-search`<br>
**Development Head:** `pip install git+https://github.com/si2-urssi/eager.git`

## Quickstart

1. Load our best model (the "TF-IDF Vectorizer Logistic Regression Model")
2. Pull award abstract texts from the NSF API
3. Predict if the award will produce software using the abstract text for each award

```python
from soft_search.constants import NSFFields, NSFPrograms
from soft_search.label import load_soft_search_model
from soft_search.nsf import get_nsf_dataset

# Load the model
pipeline = load_soft_search_model()

# Pull data
data = get_nsf_dataset(
    start_date="2022-05-01",
    end_date="2022-07-01",
    program_name=NSFPrograms.Computer_and_Information_Science_and_Engineering,
    dataset_fields=[NSFFields.id_, NSFFields.abstractText],
    require_project_outcomes_doc=False,
)

# Predict
data["prediction"] = pipeline.predict(data[NSFFields.abstractText])
print(data)

#                                         abstractText       id              prediction
# 0  Human AI Teaming (HAT) is an emerging and rapi...  2213827  software-not-predicted
# 1  This project furthers progress in our understa...  2213756      software-predicted
```

### Annotated Training Data

```python
from soft_search.data import load_soft_search_2022

df = load_soft_search_2022()
```

### Reproducible Models

| model                  	| accuracy 	| precision 	| recall   	| f1       	|
|------------------------	|----------	|-----------	|----------	|----------	|
| tfidf-logit            	| 0.696682 	| 0.684450  	| 0.696682 	| 0.686327 	|
| semantic-logit         	| 0.682464 	| 0.663943  	| 0.682464 	| 0.633984 	|
| fine-tuned-transformer 	| 0.710900 	| 0.576923  	| 0.616438 	| 0.596026 	|
| regex                  	| 0.526066 	| 0.537927  	| 0.526066 	| 0.531303 	|

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
     from soft_search.data import load_soft_search_2022

     data = load_soft_search_2022()
     ```
  - Prior to the manual annotation process, we conducted multiple rounds of
    annotation trials to ensure we had agreement on our labeling definitions.
    The final annotation trial results which resulted in an inter-rater
    reliability (Fleiss Kappa score) of 0.8918 (near perfect) is available
    via the following function:
    ```python
    from soft_search.data import load_soft_search_2022_irr

    data = load_soft_search_2022_irr()
    ```
    Additionally, the code for calculating the Fleiss Kappa Statistic
    is available at the following link:
    https://github.com/si2-urssi/eager/blob/main/soft_search/data/irr.py


## Documentation

For full package documentation please visit [si2-urssi.github.io/eager](https://si2-urssi.github.io/eager).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

**MIT License**
