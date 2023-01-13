# soft-search

[![Build Status](https://github.com/si2-urssi/eager/workflows/CI/badge.svg)](https://github.com/si2-urssi/eager/actions)
[![Documentation](https://github.com/si2-urssi/eager/workflows/Documentation/badge.svg)](https://si2-urssi.github.io/eager)

searching for software promises in grant applications

---

## Installation

**Stable Release:** `pip install soft-search`<br>
**Development Head:** `pip install git+https://github.com/si2-urssi/eager.git`

## Quickstart

```python
from soft_search.data import load_soft_search_2022

df = load_soft_search_2022()
```

## Models

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
