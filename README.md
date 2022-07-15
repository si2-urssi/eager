# soft-search
searching for software promises in grant applications 

## Generating the Basic Dataset

### Python

Uses the NSF HTTP API with specifics parameters.

```bash
pip install -r scripts/requirements.txt
python scripts/get-data.py
```

Data is stored in `data/nsf-awards.csv` and `data/nsf-awards.parquet`.
(Same data, different formats).

### R

Uses AwardFindR keyword search for NSF.

```bash
Rscript scripts/requirements.r
Rscript scripts/get-data.r
```

Data is stored in `data/keyword-seard.csv`.

## Generating the Filtered Dataset

### Python

```bash
python scripts/select-awards.py
```

Data is stored in `data/filtered-nsf-awards.csv` and `data/filtered-nsf-awards.parquet`.
(Same data, different formats).