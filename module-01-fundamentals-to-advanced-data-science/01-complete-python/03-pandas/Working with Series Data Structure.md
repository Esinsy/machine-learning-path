# 🛠️ Working with Series Data Structure  

In this section, we will practice how to **create, explore, and manipulate** Pandas Series.  
You will see how to:  
- Create a Series from lists, NumPy arrays, or dictionaries  
- Access elements using labels or indices  
- Perform simple operations on Series  


```python
import pandas as pd
```

```python
games = pd.read_csv("DataSets/vgsalesGlobale.csv")
```

```python
games.head()
```

```python
games.dtypes
```

```python

games.Genre.describe()
```

```python

games.Genre.value_counts() 
```

```python
games.Genre.value_counts(normalize=True) 
```

```python

type(games.Genre.value_counts())
```

```python

games.Genre.value_counts().head()
```

```python

games.Genre.unique()
```

```python

pd.crosstab(games.Genre, games.Year)
```

```python

games.Global_Sales.describe().T
```

```python
games.Global_Sales.mean()
```

```python

games.Global_Sales.value_counts()
```

```python
games.Year.plot(kind="hist")
```

```python
games.Genre.value_counts()
```

```python
games.Genre.value_counts().plot(kind="bar")
```

```python

```
