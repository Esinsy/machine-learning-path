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

# **Lesson 02: Introduction to NumPy**

This notebook introduces the fundamentals of NumPy, Python's powerful numerical computing library.  
We'll cover array creation, basic operations, broadcasting, indexing, and more.

📌 **Plan**:  

- 🔢 Creating arrays  
- 🧩 Multi-dimensional arrays  
- 🎲 Randomly generated arrays  
- ⚡ Element-wise operations  
  - 🔍 Comparison operations  
  - ✅ Logical operations  
- 📊 Summarizing operations  




## **🔹 Importing NumPy**
We start by importing **NumPy** with the alias `np`, which is the standard convention used in almost all projects.  
This makes it easier and faster to call NumPy functions.

📖 Official documentation: [NumPy Reference](https://numpy.org/doc/)

## **📚 Interesting Background**

NumPy (short for **Numerical Python**) was originally created in 2005 by **Travis Oliphant**, who merged two earlier Python libraries: **Numeric** and **Numarray**.  
It quickly became the **core scientific computing library in Python**, forming the foundation of libraries such as **pandas, SciPy, scikit-learn, TensorFlow, and PyTorch**.

🔗 Learn more about the history and impact of NumPy:

- [The History of NumPy (Nature article)](https://www.nature.com/articles/d41586-020-03382-2)  
- [NumPy: The fundamental package for scientific computing in Python (Official About page)](https://numpy.org/about/)



```python
import numpy as np
```

## **🔹 Creating Arrays**

NumPy arrays are the core data structure used for numerical computations.  
They are more efficient and flexible than Python lists, allowing fast operations on large datasets.

We can create arrays in multiple ways:
- From Python lists or tuples  
- Using built-in functions like `np.zeros()`, `np.ones()`, `np.arange()`, and `np.linspace()`  
- With random number generators such as `np.random.rand()`  

👉 Arrays form the foundation for all operations in NumPy.


```python
np.zeros(10)
```

```python
np.zeros(10)
```

```python
np.full(10,2.5)
```

```python
a = np.array([1,2,3,5,7,12])
a
```

```python
a[2]
```

```python
a[2] = 10
```

```python
a
```

```python
np.arange(10)
```

```python
np.arange(3,10)
```

```python
np.linspace(0,100, 11)
```

## **🔹 Multi-dimensional Arrays**

NumPy supports arrays with more than one dimension (also called **matrices** or **tensors**).  
These allow us to represent data in rows and columns, or even higher dimensions.

Common examples:
- 2D arrays → tables or matrices  
- 3D arrays → images or stacked data  
- nD arrays → tensors for advanced computations (e.g., in deep learning)

👉 Multi-dimensional arrays are the foundation of numerical computing, enabling operations on structured data.


```python
np.zeros((5,2))
```

```python
n = np.array([[1,2,3],
         [4,5,6],
         [7,8,9]
         ])
```

```python
n[0][2]
```

```python
n[0,1] = 10
```

```python
n[0]
```

```python
n[2] = [1,1,1]
```

```python
n
```

```python
n[:,1]
```

```python
n[:, 2]
```

```python
n[:, 2] = [0,1,2]
```

```python
n
```

## **🔹 Randomly Generated Arrays**

NumPy provides tools for generating arrays filled with random numbers.  
These arrays are useful for simulations, testing algorithms, or initializing model parameters.  

You can generate:
- Uniformly distributed numbers with `np.random.rand()`  
- Normally distributed numbers with `np.random.randn()`  
- Random integers with `np.random.randint()`  

👉 Random arrays are essential when working with probabilistic models, machine learning, and data sampling.


```python
np.random.rand(5,2)
```

```python
np.random.seed(10)
```

## **🔹 Element-wise Operations**

NumPy allows mathematical operations to be applied **element by element** across arrays.  
This makes computations fast and concise compared to traditional Python loops.  

Examples include:
- Arithmetic operations: `+`, `-`, `*`, `/`, `**`  
- Comparison operations: `<`, `>`, `==`, `!=`  
- Logical operations: `&`, `|`, `~`  

👉 Element-wise operations are one of the key reasons NumPy is so powerful for numerical computing.


```python
a = np.arange(5)
a
```

```python
a + 1
```

```python
b = (10 + (a * 2)) ** 2 / 100
```

```python
a + b
```

```python
a / b
```

```python
a / b +10
```

### **🔹 Comparison Operations**

NumPy supports element-wise comparison between arrays or between an array and a scalar.  
The result is a boolean array indicating whether the condition is `True` or `False`.  

Examples:
- `a > 5` → checks if each element is greater than 5  
- `a == 0` → checks equality with zero  

👉 Useful for filtering, masking, and conditional operations on data.


```python
a 
```

```python
a >= 2
```

```python
b
```

```python
a > b
```

```python
a[a>b]
```

### **🔹 Logical Operations**

Logical operations combine multiple boolean conditions element-wise.  
They return arrays of `True`/`False` values based on logical rules.  

Examples:
- `&` → logical AND  
- `|` → logical OR  
- `~` → logical NOT  

👉 Logical operations are often used with comparisons to build complex conditions, e.g. `(a > 2) & (a < 10)`.






## **🔹 Summarizing Operations**

NumPy provides functions to quickly compute summary statistics across arrays.  
These operations can be applied to the entire array or along specific axes (rows or columns).  

Common examples:
- `np.sum()` → total of all elements  
- `np.mean()` → average value  
- `np.min()`, `np.max()` → minimum and maximum values  
- `np.std()` → standard deviation  

👉 Summarizing operations are essential for understanding the overall properties of your data.


```python
a
```

```python
a.max()
```

## **Multiplication**

```python
def vector_vector_multiplication(u, v):
    assert u.shape[0] == v.shape[0]
```

```python
U = np.array([[2,4,5,6],
             [1,2,1,2],
             [3,1,2,1]])
```

```python

```
