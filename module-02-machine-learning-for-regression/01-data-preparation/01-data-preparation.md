# 2.2 Data preparation

```python

import pandas as pd
import numpy as np
```

## Reading Car Price Dataset

```python
df = pd.read_csv("dataset/data.csv")
```

```python
df.head()
```

```python
df.info()
```

## Data Standardization

```python
df.columns = df.columns.str.lower().str.replace(" ","_")
```

```python
df.head()
```

## Catching Categorical(Object) Values in the Dataset

```python
strings = list(df.dtypes[df.dtypes=="object"].index)
```

```python
strings
```

## Values Standardization

```python
for col in strings:
    df[col] = df[col].str.lower().str.replace(" ","_")
```

```python
df.head()
```

```python

```

## EDA

```python

```
