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

## Documentation

For full package documentation please visit [si2-urssi.github.io/eager](https://si2-urssi.github.io/eager).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

**MIT License**
