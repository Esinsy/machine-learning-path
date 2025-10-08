# **HOMEWORK MODULE-2**


# 📦 **1. Install and Import Required Libraries**



```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(style="whitegrid", palette="muted", font_scale=1.1)
```

## 📥 **Download Dataset using `wget`**

In this step, the dataset **`car_fuel_efficiency.csv`** is downloaded directly from the official  
GitHub repository of **Alexey Grigorev** using the **`wget`** command-line utility.

**Command Explanation:**  
- `wget` — a tool for downloading files from the internet.  
- The URL points to the raw CSV file hosted on GitHub.  
- The flag `-O car_fuel_efficiency.csv` specifies the output filename,  
  ensuring the dataset is saved in the current working directory.  

**Purpose:**  
This approach guarantees that the dataset can be obtained automatically without  
manual downloads, keeping the notebook **fully reproducible** and ready for data analysis.  

**Expected Output:**  
After running the command, the file **`car_fuel_efficiency.csv`** will appear in your  
current working directory (e.g., `homework/`).  
You can verify the download by listing files with `!ls` or by loadin



```python
!wget https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv -O car_fuel_efficiency.csv

```

    --2025-10-08 14:04:00--  https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 2606:50c0:8001::154, 2606:50c0:8002::154, 2606:50c0:8003::154, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|2606:50c0:8001::154|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 874188 (854K) [text/plain]
    Saving to: ‘car_fuel_efficiency.csv’
    
    car_fuel_efficiency 100%[===================>] 853.70K  --.-KB/s    in 0.1s    
    
    2025-10-08 14:04:01 (8.57 MB/s) - ‘car_fuel_efficiency.csv’ saved [874188/874188]
    



```python
ls -lh car_fuel_efficiency.csv

```

    -rw-r--r--  1 esinscomak  staff   854K Oct  8 14:04 car_fuel_efficiency.csv



```python
import pandas as pd
df = pd.read_csv("car_fuel_efficiency.csv")
df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>engine_displacement</th>
      <th>num_cylinders</th>
      <th>horsepower</th>
      <th>vehicle_weight</th>
      <th>acceleration</th>
      <th>model_year</th>
      <th>origin</th>
      <th>fuel_type</th>
      <th>drivetrain</th>
      <th>num_doors</th>
      <th>fuel_efficiency_mpg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>170</td>
      <td>3.0</td>
      <td>159.0</td>
      <td>3413.433759</td>
      <td>17.7</td>
      <td>2003</td>
      <td>Europe</td>
      <td>Gasoline</td>
      <td>All-wheel drive</td>
      <td>0.0</td>
      <td>13.231729</td>
    </tr>
    <tr>
      <th>1</th>
      <td>130</td>
      <td>5.0</td>
      <td>97.0</td>
      <td>3149.664934</td>
      <td>17.8</td>
      <td>2007</td>
      <td>USA</td>
      <td>Gasoline</td>
      <td>Front-wheel drive</td>
      <td>0.0</td>
      <td>13.688217</td>
    </tr>
    <tr>
      <th>2</th>
      <td>170</td>
      <td>NaN</td>
      <td>78.0</td>
      <td>3079.038997</td>
      <td>15.1</td>
      <td>2018</td>
      <td>Europe</td>
      <td>Gasoline</td>
      <td>Front-wheel drive</td>
      <td>0.0</td>
      <td>14.246341</td>
    </tr>
    <tr>
      <th>3</th>
      <td>220</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>2542.392402</td>
      <td>20.2</td>
      <td>2009</td>
      <td>USA</td>
      <td>Diesel</td>
      <td>All-wheel drive</td>
      <td>2.0</td>
      <td>16.912736</td>
    </tr>
    <tr>
      <th>4</th>
      <td>210</td>
      <td>1.0</td>
      <td>140.0</td>
      <td>3460.870990</td>
      <td>14.4</td>
      <td>2009</td>
      <td>Europe</td>
      <td>Gasoline</td>
      <td>All-wheel drive</td>
      <td>2.0</td>
      <td>12.488369</td>
    </tr>
  </tbody>
</table>
</div>



## 🧩 **Preparing the Dataset**

In this step, the dataset will be prepared for analysis by selecting only the relevant numerical features.  
The following columns are retained for modeling fuel efficiency:

- **`engine_displacement`** — Engine size or volume, typically measured in liters or cubic centimeters (cc).  
- **`horsepower`** — Engine power output, representing the vehicle’s performance capability.  
- **`vehicle_weight`** — The total weight of the vehicle, usually measured in pounds or kilograms.  
- **`model_year`** — The manufacturing or model year of the vehicle.  
- **`fuel_efficiency_mpg`** — Target variable representing fuel efficiency in miles per gallon (MPG).  

These selected columns form a **clean, structured dataset** suitable for exploratory analysis  
and predictive modeling tasks such as **linear regression**.


## 📊 **2. Inspect Dataset Structure**

Display each column’s data type, number of non-null values, and overall memory usage to understand the dataset’s composition.



```python
df.info()

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9704 entries, 0 to 9703
    Data columns (total 11 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   engine_displacement  9704 non-null   int64  
     1   num_cylinders        9222 non-null   float64
     2   horsepower           8996 non-null   float64
     3   vehicle_weight       9704 non-null   float64
     4   acceleration         8774 non-null   float64
     5   model_year           9704 non-null   int64  
     6   origin               9704 non-null   object 
     7   fuel_type            9704 non-null   object 
     8   drivetrain           9704 non-null   object 
     9   num_doors            9202 non-null   float64
     10  fuel_efficiency_mpg  9704 non-null   float64
    dtypes: float64(6), int64(2), object(3)
    memory usage: 834.1+ KB


## 🔍 **3. Preview First Rows**

Show the first few rows of the dataset to verify that only the intended columns remain and values appear consistent.



```python
display(df.head())

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>engine_displacement</th>
      <th>num_cylinders</th>
      <th>horsepower</th>
      <th>vehicle_weight</th>
      <th>acceleration</th>
      <th>model_year</th>
      <th>origin</th>
      <th>fuel_type</th>
      <th>drivetrain</th>
      <th>num_doors</th>
      <th>fuel_efficiency_mpg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>170</td>
      <td>3.0</td>
      <td>159.0</td>
      <td>3413.433759</td>
      <td>17.7</td>
      <td>2003</td>
      <td>Europe</td>
      <td>Gasoline</td>
      <td>All-wheel drive</td>
      <td>0.0</td>
      <td>13.231729</td>
    </tr>
    <tr>
      <th>1</th>
      <td>130</td>
      <td>5.0</td>
      <td>97.0</td>
      <td>3149.664934</td>
      <td>17.8</td>
      <td>2007</td>
      <td>USA</td>
      <td>Gasoline</td>
      <td>Front-wheel drive</td>
      <td>0.0</td>
      <td>13.688217</td>
    </tr>
    <tr>
      <th>2</th>
      <td>170</td>
      <td>NaN</td>
      <td>78.0</td>
      <td>3079.038997</td>
      <td>15.1</td>
      <td>2018</td>
      <td>Europe</td>
      <td>Gasoline</td>
      <td>Front-wheel drive</td>
      <td>0.0</td>
      <td>14.246341</td>
    </tr>
    <tr>
      <th>3</th>
      <td>220</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>2542.392402</td>
      <td>20.2</td>
      <td>2009</td>
      <td>USA</td>
      <td>Diesel</td>
      <td>All-wheel drive</td>
      <td>2.0</td>
      <td>16.912736</td>
    </tr>
    <tr>
      <th>4</th>
      <td>210</td>
      <td>1.0</td>
      <td>140.0</td>
      <td>3460.870990</td>
      <td>14.4</td>
      <td>2009</td>
      <td>Europe</td>
      <td>Gasoline</td>
      <td>All-wheel drive</td>
      <td>2.0</td>
      <td>12.488369</td>
    </tr>
  </tbody>
</table>
</div>


## 📈 **4. Summary Statistics**

Generate descriptive statistics such as mean, standard deviation, min, and max to understand value ranges and detect outliers.



```python
df.describe().T

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>engine_displacement</th>
      <td>9704.0</td>
      <td>199.708368</td>
      <td>49.455319</td>
      <td>10.000000</td>
      <td>170.000000</td>
      <td>200.000000</td>
      <td>230.000000</td>
      <td>380.000000</td>
    </tr>
    <tr>
      <th>num_cylinders</th>
      <td>9222.0</td>
      <td>3.962481</td>
      <td>1.999323</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>13.000000</td>
    </tr>
    <tr>
      <th>horsepower</th>
      <td>8996.0</td>
      <td>149.657292</td>
      <td>29.879555</td>
      <td>37.000000</td>
      <td>130.000000</td>
      <td>149.000000</td>
      <td>170.000000</td>
      <td>271.000000</td>
    </tr>
    <tr>
      <th>vehicle_weight</th>
      <td>9704.0</td>
      <td>3001.280993</td>
      <td>497.894860</td>
      <td>952.681761</td>
      <td>2666.248985</td>
      <td>2993.226296</td>
      <td>3334.957039</td>
      <td>4739.077089</td>
    </tr>
    <tr>
      <th>acceleration</th>
      <td>8774.0</td>
      <td>15.021928</td>
      <td>2.510339</td>
      <td>6.000000</td>
      <td>13.300000</td>
      <td>15.000000</td>
      <td>16.700000</td>
      <td>24.300000</td>
    </tr>
    <tr>
      <th>model_year</th>
      <td>9704.0</td>
      <td>2011.484027</td>
      <td>6.659808</td>
      <td>2000.000000</td>
      <td>2006.000000</td>
      <td>2012.000000</td>
      <td>2017.000000</td>
      <td>2023.000000</td>
    </tr>
    <tr>
      <th>num_doors</th>
      <td>9202.0</td>
      <td>-0.006412</td>
      <td>1.048162</td>
      <td>-4.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>fuel_efficiency_mpg</th>
      <td>9704.0</td>
      <td>14.985243</td>
      <td>2.556468</td>
      <td>6.200971</td>
      <td>13.267459</td>
      <td>15.006037</td>
      <td>16.707965</td>
      <td>25.967222</td>
    </tr>
  </tbody>
</table>
</div>



## 🧮 **5. Check for Missing Values**

Identify missing data in each column to prepare for cleaning or imputation steps later in the workflow.



```python
print("Missing values per column:\n")
print(df.isnull().sum())

```

    Missing values per column:
    
    engine_displacement      0
    num_cylinders          482
    horsepower             708
    vehicle_weight           0
    acceleration           930
    model_year               0
    origin                   0
    fuel_type                0
    drivetrain               0
    num_doors              502
    fuel_efficiency_mpg      0
    dtype: int64


## 📊 **Exploratory Data Analysis (EDA)**  
### 🔍 **Examining the Target Variable — `fuel_efficiency_mpg`**

In this step, we analyze the distribution of the target variable **`fuel_efficiency_mpg`** to check whether it exhibits a **long tail**.  
A long-tailed distribution would indicate that while most vehicles have moderate fuel efficiency, a small number achieve exceptionally high MPG values.  
Understanding this shape is important for later transformations or modeling.



```python
# Plot the distribution of the target variable
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
sns.histplot(df['fuel_efficiency_mpg'], bins=30, kde=True, color='steelblue')
plt.title('Distribution of Fuel Efficiency (MPG)', fontsize=14)
plt.xlabel('Fuel Efficiency (MPG)')
plt.ylabel('Count')
plt.show()

```


    
![png](homework_2_files/homework_2_17_0.png)
    


## 🧠 **Interpretation — Fuel Efficiency Distribution**

The histogram shows that the **`fuel_efficiency_mpg`** variable is **right-skewed**, indicating a **long tail** on the higher end of the distribution.  
Most vehicles achieve moderate fuel efficiency values (around **15–35 MPG**),  
while a smaller number of vehicles reach significantly higher MPG values, extending the right tail.

This **long-tailed pattern** suggests that:
- The dataset contains a few **highly efficient vehicles** (possibly hybrids or smaller engines).  
- A **logarithmic or square-root transformation** might later be beneficial if we aim to normalize the target variable.  

Understanding this distribution helps ensure that the modeling approach accounts for these **rare, extreme values** without being dominated by them.


## ❓ **Question 1 — Identify the Column with Missing Values**

In this step, we examine the dataset to determine which column contains missing values.  
Among the following columns:  
- **`engine_displacement`**  
- **`horsepower`**  
- **`vehicle_weight`**  
- **`model_year`**  

we will check for null entries using the Pandas `isnull()` and `sum()` methods.  
This will help us detect if any numerical variable requires cleaning or imputation before modeling.



```python
# Check for missing values in the specified columns
cols_to_check = ['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year']

missing_counts = df[cols_to_check].isnull().sum()
print("🔎 Missing values per column:\n")
print(missing_counts)

# Identify which column has missing values
missing_col = missing_counts[missing_counts > 0]
if not missing_col.empty:
    print(f"\n✅ The column with missing values is: {missing_col.index[0]}")
else:
    print("\n✅ There are no missing values in these columns.")

```

    🔎 Missing values per column:
    
    engine_displacement      0
    horsepower             708
    vehicle_weight           0
    model_year               0
    dtype: int64
    
    ✅ The column with missing values is: horsepower


## ❓ **Question 2 — Median Value of `horsepower`**

In this step, we calculate the **median (50th percentile)** of the variable **`horsepower`**  
to understand the central tendency of engine power in the dataset.  
The median is a robust measure of central location, less sensitive to extreme values than the mean.

We will use the Pandas `.median()` method to obtain the 50th percentile value and compare it with the provided options:

- **49**
- **99**
- **149**
- **199**



```python
# Calculate the median horsepower
median_hp = df['horsepower'].median()
print(f"✅ The median (50th percentile) of horsepower is: {median_hp:.0f}")

```

    ✅ The median (50th percentile) of horsepower is: 149


## ❓ **Question 3 — Handling Missing Values and Model Comparison**

The variable identified in **Question 1** contains missing values (`horsepower`).  
We will test **two strategies** for handling these missing values and evaluate which one leads to better model performance:

1. **Fill missing values with 0**  
2. **Fill missing values with the mean (computed from the training data only)**  

We will:
- Split the dataset into **training** and **validation** subsets.  
- Train a **Linear Regression** model (without regularization) for each strategy.  
- Evaluate performance using **RMSE (Root Mean Squared Error)** on the validation set.  
- Round RMSE values to **two decimal places** (`round(score, 2)`).  

The approach that produces a **lower RMSE** indicates a more effective handling of missing values.



```python
# -------------------------------------------------------------
# 🧩 1. Prepare Data
# -------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Define features and target
features = ['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year']
target = 'fuel_efficiency_mpg'

# Drop rows with missing target values (if any)
df = df.dropna(subset=[target])

# Split into training and validation sets
df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)

# Reset indices
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

print("✅ Data prepared for training and validation.")
print("Train size:", len(df_train), "| Validation size:", len(df_val))

```

    ✅ Data prepared for training and validation.
    Train size: 7763 | Validation size: 1941



```python
# -------------------------------------------------------------
# ⚙️ Option 1 — Fill missing 'horsepower' with 0
# -------------------------------------------------------------
from math import sqrt   # ✅ make sure sqrt is imported

df_train_0 = df_train.copy()
df_val_0 = df_val.copy()

df_train_0['horsepower'] = df_train_0['horsepower'].fillna(0)
df_val_0['horsepower'] = df_val_0['horsepower'].fillna(0)

# Train Linear Regression model
model_0 = LinearRegression()
model_0.fit(df_train_0[features], df_train_0[target])

# Predict and compute RMSE
y_pred_0 = model_0.predict(df_val_0[features])
mse_0 = mean_squared_error(df_val_0[target], y_pred_0)
rmse_0 = sqrt(mse_0)

print(f"RMSE (fill with 0): {round(rmse_0, 2)}")

```

    RMSE (fill with 0): 0.53


## ❓ **Question 4 — Regularized Linear Regression (Ridge) with NA=0**

**Goal.** Train a **regularized linear regression (Ridge)** model to predict **`fuel_efficiency_mpg`** by trying different regularization strengths  
\(denoted as **r**\) from the list: **[0, 0.01, 0.1, 1, 5, 10, 100]**.  
Fill **all missing feature values with 0**, evaluate each model on the **validation** set using **RMSE**, and **round to 2 decimals**.  
Select the **best r** (lowest RMSE). If there is a tie, choose the **smallest r** among the best.

**Notes.**
- **r = 0** corresponds to ordinary least squares (no regularization).  
- We compute RMSE as `sqrt(MSE)` for broad scikit-learn compatibility.  
- Train/validation split is fixed with `random_state=42` for reproducibility.



```python
# -------------------------------------------------------------
# ✅ Ridge regression sweep over r with NA=0
# -------------------------------------------------------------
from math import sqrt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Features/target
features = ['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year']
target = 'fuel_efficiency_mpg'

# Drop rows with missing target if any
df = df.dropna(subset=[target]).copy()

# Train/validation split
df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)
df_train = df_train.reset_index(drop=True)
df_val   = df_val.reset_index(drop=True)

# Fill NA=0 on FEATURES only
X_train = df_train[features].fillna(0)
y_train = df_train[target].values

X_val   = df_val[features].fillna(0)
y_val   = df_val[target].values

# List of r (alpha) values to try
r_list = [0, 0.01, 0.1, 1, 5, 10, 100]

results = []

for r in r_list:
    # Ridge with given alpha (r). alpha=0 is valid and equals OLS.
    model = Ridge(alpha=r, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    rmse = sqrt(mean_squared_error(y_val, y_pred))

    results.append((r, round(rmse, 2)))

# Sort results by RMSE, then by r (ascending) to break ties by smallest r
results_sorted = sorted(results, key=lambda x: (x[1], x[0]))

# Pretty print
print("RMSE by r (rounded to 2 decimals):")
for r, rmse in results:
    print(f"  r={r:<6} -> RMSE={rmse:.2f}")

best_r, best_rmse = results_sorted[0]
print("\n✅ Best choice:")
print(f"  r = {best_r}  (RMSE = {best_rmse:.2f})")

```

    RMSE by r (rounded to 2 decimals):
      r=0      -> RMSE=0.53
      r=0.01   -> RMSE=0.53
      r=0.1    -> RMSE=0.53
      r=1      -> RMSE=0.53
      r=5      -> RMSE=0.53
      r=10     -> RMSE=0.53
      r=100    -> RMSE=0.53
    
    ✅ Best choice:
      r = 0  (RMSE = 0.53)


## ❓ **Question 5 — Effect of Random Seed on Model Stability**

**Goal.**  
Evaluate how different random seeds affect model performance stability.  
We will use **10 different seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]**,  
and for each seed we will:

1. Split the data into **train (60%)**, **validation (20%)**, and **test (20%)** sets.  
2. Fill missing values with **0**.  
3. Train a **Linear Regression (no regularization)** model.  
4. Compute the **RMSE** on the validation dataset.  
5. Collect all RMSE scores and compute their **standard deviation** using `np.std()`.

A **low standard deviation** indicates that the model’s performance is **stable and consistent** regardless of the random seed.



```python
# -------------------------------------------------------------
# ⚙️ QUESTION 5 — Model Stability Across Random Seeds
# -------------------------------------------------------------
from math import sqrt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Define features and target
features = ['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year']
target = 'fuel_efficiency_mpg'

# Prepare container for RMSE scores
rmse_scores = []

# Loop over seeds 0–9
for seed in range(10):
    # Split dataset: 60% train, 20% val, 20% test
    df_full = df.dropna(subset=[target]).copy()
    df_train_full, df_test = train_test_split(df_full, test_size=0.2, random_state=seed)
    df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=seed)  # 0.25 of 0.8 = 0.2 overall

    # Reset index
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    # Fill missing values with 0
    X_train = df_train[features].fillna(0)
    y_train = df_train[target].values
    X_val = df_val[features].fillna(0)
    y_val = df_val[target].values

    # Train Linear Regression model (no regularization)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict and calculate RMSE
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    rmse = sqrt(mse)

    rmse_scores.append(rmse)

# Compute standard deviation of RMSE scores
std_rmse = np.std(rmse_scores)
print("RMSE scores for different seeds:", [round(x, 3) for x in rmse_scores])
print(f"\n✅ Standard deviation of RMSE: {round(std_rmse, 3)}")

```

    RMSE scores for different seeds: [0.528, 0.522, 0.516, 0.512, 0.52, 0.526, 0.507, 0.522, 0.516, 0.507]
    
    ✅ Standard deviation of RMSE: 0.007


## ❓ **Question 6 — Final Model Evaluation on Test Set**

**Goal.**  
Using **seed = 9**, split the dataset again (as before) into:
- **Train (60%)**
- **Validation (20%)**
- **Test (20%)**

Then:
1. **Combine train and validation** datasets into a single training set.  
2. **Fill missing values with 0.**  
3. Train a **Ridge Regression model** (regularized linear regression) with **r = 0.001**.  
4. Evaluate the model using **RMSE on the test dataset**.  
5. Round the RMSE to **2 decimal digits** and select the correct option.

This step gives the final measure of how well the model generalizes to unseen data.



```python
# -------------------------------------------------------------
# ⚙️ QUESTION 6 — Final Model Evaluation
# -------------------------------------------------------------
from math import sqrt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Define features and target
features = ['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year']
target = 'fuel_efficiency_mpg'

# Use seed = 9 for splitting
df_full = df.dropna(subset=[target]).copy()

# Split: 60% train, 20% val, 20% test
df_train_full, df_test = train_test_split(df_full, test_size=0.2, random_state=9)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=9)

# Combine train and validation sets
df_train_combined = pd.concat([df_train, df_val]).reset_index(drop=True)

# Prepare train/test data (fill NAs with 0)
X_train = df_train_combined[features].fillna(0)
y_train = df_train_combined[target].values

X_test = df_test[features].fillna(0)
y_test = df_test[target].values

# Train Ridge model with r=0.001
model_final = Ridge(alpha=0.001, random_state=9)
model_final.fit(X_train, y_train)

# Predict on test set and compute RMSE
y_pred_test = model_final.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = sqrt(mse_test)

print(f"✅ RMSE on Test Dataset: {round(rmse_test, 2)}")

```

    ✅ RMSE on Test Dataset: 0.52



```python

```
