# soft-search

[![Build Status](https://github.com/PugetSoundClinic-PIT/soft-search/workflows/CI/badge.svg)](https://github.com/PugetSoundClinic-PIT/soft-search/actions)
[![Documentation](https://github.com/PugetSoundClinic-PIT/soft-search/workflows/Documentation/badge.svg)](https://PugetSoundClinic-PIT.github.io/soft-search)

searching for software promises in grant applications

---

## Installation

**Stable Release:** `pip install soft-search`<br>
**Development Head:** `pip install git+https://github.com/PugetSoundClinic-PIT/soft-search.git`

## Quickstart

### Apply our Pre-trained Transformer

```bash
pip install soft-search[nsf,transformer]
```

```python
from soft_search import constants, nsf
from soft_search.label import transformer
df = nsf.get_nsf_dataset(
    "2016-01-01",
    "2017-01-01",
    dataset_fields=[constants.NSFFields.abstractText],
)
predicted = transformer.label(
    df,
    apply_column=constants.NSFFields.abstractText,
)
```

## Documentation

For full package documentation please visit [PugetSoundClinic-PIT.github.io/soft-search](https://PugetSoundClinic-PIT.github.io/soft-search).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

**MIT License**
