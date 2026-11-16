# Dataset Information

## Dataset Name
Bank Marketing Dataset

## Source
UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/222/bank+marketing

## Dataset File
`bank-full.csv`

## Dataset Description
The dataset contains 45,211 customer records and 17 columns, including customer information, previous marketing campaign details, and the target variable.

The target variable is `y`:

- `yes`: Customer subscribed to a term deposit.
- `no`: Customer did not subscribe.

## Loading the Dataset

The original CSV file uses a semicolon (`;`) as its delimiter.

```python
import pandas as pd

df = pd.read_csv("bank-full.csv", sep=";")

print(df.shape)
```

Expected shape:

`(45211, 17)`

## Important Note
The dataset is not stored in this GitHub repository. Download it from the official UCI source before running the notebook.

The `duration` feature is excluded from model inputs because it is only known after a marketing call, which would cause data leakage for pre-call prediction.
