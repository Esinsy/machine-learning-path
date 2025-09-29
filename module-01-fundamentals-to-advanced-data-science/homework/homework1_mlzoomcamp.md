## **Homework 1  - ML Zoomcamp**

```python
import pandas as pd
import numpy as np
```

```python
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
```

 ### ****

```python
pd.__version__
```

 ### **Getting the data**
For this homework, we'll use the Car Fuel Efficiency dataset.

```python
df = pd.read_csv("car_fuel_efficiency.csv")
```

```python
df.head()
```

### **Data Overview**

```python
print(df.columns.tolist())

```

```python
# check the dimensions of the dataset (rows, columns)
df.shape

# get data types of all columns
df.dtypes

# count non-null values in each column
df.count()

# count missing values in each column
df.isnull().sum()

# number of unique values in each column
df.nunique()

# correlation matrix between numerical columns
df.corr(numeric_only=True)

```

```python
print("➡️ Dataset dimensions (rows, columns):", df.shape, "\n")

print("➡️ Data types of all columns:\n", df.dtypes, "\n")

print("➡️ Non-null counts per column:\n", df.count(), "\n")

print("➡️ Missing values per column:\n", df.isnull().sum(), "\n")

print("➡️ Number of unique values per column:\n", df.nunique(), "\n")

print("➡️ Correlation matrix (numerical columns):\n", df.corr(numeric_only=True), "\n")


```

### **Q2. Records count**
How many records are in the dataset?
9704

```python
# Q2. Records count
# The number of records (rows) in the dataset
print("➡️ Number of records in the dataset:", df.shape[0])

```

### **Q3. Fuel types**
How many fuel types are presented in the dataset?

```python
# Q3. Fuel types
# Count how many unique fuel types exist in the dataset
print("➡️ Number of unique fuel types:", df['fuel_type'].nunique())
print("➡️ Fuel types are:", df['fuel_type'].unique())

```

### **Question 4. Missing values**

```python
# Q4. Missing values
# Count total missing values across the dataset
total_missing = df.isnull().sum().sum()

print("➡️ Total number of missing values in the dataset:", total_missing)

```

```python
# Missing values per column
missing_per_column = df.isnull().sum()

print("➡️ Missing values per column:\n", missing_per_column[missing_per_column > 0], "\n")

# How many columns have missing values?
missing_columns = (missing_per_column > 0).sum()
print("➡️ Number of columns with missing values:", missing_columns)

# Total missing values across the dataset
total_missing = missing_per_column.sum()
print("➡️ Total missing values in the dataset:", total_missing)

```

### **Question 5. Max fuel efficiency**

What's the maximum fuel efficiency of cars from Asia?

```python
# Max fuel efficiency for cars from Asia
asia_max_eff = df[df['origin'] == 'Asia']['fuel_efficiency_mpg'].max()

print("➡️ Maximum fuel efficiency of cars from Asia:", asia_max_eff)

```

### **Q6. Median value of horsepower**
1. Find the median value of horsepower column in the dataset.
2. Next, calculate the most frequent value of the same horsepower column.
3. Use fillna method to fill the missing values in horsepower column with the most frequent value from the previous step.
4. Now, calculate the median value of horsepower once again.
Has it changed?

Yes, it increased
Yes, it decreased
No



#### **1. Median of horsepower**

```python
median_hp_before = df['horsepower'].median()
print("➡️ Median horsepower (before fillna):", median_hp_before)
```

#### 2. **most frequent value of the same horsepower column.**

```python
# Step 2: Most frequent value (mode)
most_freq_hp = df['horsepower'].mode()[0]
print("➡️ Most frequent horsepower (mode):", most_freq_hp)
```

#### **3. Fill Missing Value in gorsepower with mode value**

```python
hp_filled = df['horsepower'].fillna(most_freq_hp)
```

#### 4. **calculate the median value of horsepower once again.**

```python
# Step 4: Median after filling (temporary, df is unchanged)
median_hp_after = hp_filled.median()
print("➡️ Median horsepower (after fillna):", median_hp_after)
```

#### **5. Compare**

```python
if median_hp_after > median_hp_before:
    print("✅ Answer: Yes, it increased")
elif median_hp_after < median_hp_before:
    print("✅ Answer: Yes, it decreased")
else:
    print("✅ Answer: No")
```

### **Q7. Sum of weights**
1. Select all the cars from Asia
2. Select only columns vehicle_weight and model_year
3. Select the first 7 values
4. Get the underlying NumPy array. Let's call it X.
5. Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. 6. Let's call the result XTX.
7. Invert XTX.
8. Create an array y with values [1100, 1300, 800, 900, 1000, 1100, 1200].
9. Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
10. What's the sum of all the elements of the result?**


#### **1. Select all the cars from Asia**

```python
df.head(20)
```

##### **Num of Asia Cars Method 1**

```python
df_asia_count = df[df["origin"] == "Asia"].shape[0]
print("Number of cars from Asia:", df_asia_count)
```

##### **Num of Asia Cars Method 2**

```python
df_asia_count = df["origin"].value_counts()["Asia"]
print("Number of cars from Asia:", df_asia_count)
```

#### **2. Select only columns vehicle_weight and model_year**

```python
asia_selected = df[df["origin"] == "Asia"][["vehicle_weight", "model_year"]]
```

#### **3. Select the first 7 values**

```python
asia_first7 = asia_selected.head(7)
```

#### **4. Get the underlying NumPy array. Let's call it X.**

```python
X = asia_first7.to_numpy()
print("➡️ First 7 rows as NumPy array (X):\n", X)
print("➡️ Shape of X:", X.shape)
```

#### **5. Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX.**

```python
# Compute XTX = X.T @ X
XTX = X.T @ X

print("➡️ XTX:\n", XTX)
print("➡️ Shape of XTX:", XTX.shape)

```

#### **6. Invert XTX**

```python
# Invert XTX
XTX_inv = np.linalg.inv(XTX) #np.linalg.inv() gives invert of square matrix

print("➡️ Inverse of XTX:\n", XTX_inv)

```

#### **7. Create an array y with values [1100, 1300, 800, 900, 1000, 1100, 1200].**

```python
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
print("y array:\n", y)
print("Shape of y:", y.shape)
```

#### **8. Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.**
![image.png](attachment:e1efbfa4-ff7f-4492-b07a-8a190bdd2a2b.png)

```python
# Compute w = (XTX)^(-1) X^T y
w = XTX_inv @ (X.T @ y)

print("➡️ w:\n", w)
print("➡️ Shape of w:", w.shape)

```

#### **9. What's the sum of all the elements of the result?**

```python
# Sum of all elements of w
sum_w = w.sum()

print("➡️ Sum of all elements in w:", sum_w)

```

```python

```
