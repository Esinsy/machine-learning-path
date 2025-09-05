---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# **Lesson 01: Loading Data**

This notebook covers the basics of loading datasets into Python using **pandas**.  
We’ll practice reading CSV files, exploring datasets, and preparing data for analysis.



### 📥 **Importing the Pandas Library**

In this cell, we import the **pandas** library and assign it the alias `pd`, which is the standard convention in the Python community.

Pandas provides powerful data structures such as `DataFrame` and `Series` that allow for efficient data manipulation, loading, filtering, aggregation, and much more — especially when working with CSV, Excel, SQL, or JSON data.

> ✅ `pd` is the commonly used alias and makes the code more concise.


```python
import pandas as pd
```

### 📄 **Loading the Laptop Dataset from Alexey Grigorev's GitHub Repository**

Here, we load a dataset containing information about various laptop models from an online CSV file using the `read_csv()` function from pandas.

- `pd.read_csv()` is used to read comma-separated values (CSV) files into a `DataFrame`, which is a tabular data structure with labeled axes.
- The dataset is hosted on GitHub and is accessed directly via its raw URL.
- The resulting variable `df` now holds the entire dataset and can be used for further analysis and preprocessing.

> 🔗 URL: https://raw.githubusercontent.com/alexeygrigorev/datasets/master/laptops.csv


```python
df = pd.read_csv('https://raw.githubusercontent.com/alexeygrigorev/datasets/master/laptops.csv')
```

```python
df.head()
```

```python

```
