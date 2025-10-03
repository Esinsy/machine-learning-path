# 📌 Series Data Structure  

In this section, you will learn about the **Series** data structure in Pandas.  
A *Series* is essentially a one-dimensional labeled array that can hold any type of data (integers, strings, floats, etc.).  

👉 By the end of this part, you will understand:  
- How to create a Series  
- How to access and manipulate its elements  
- Why Series is the building block for the Pandas **DataFrame**  


```python
import pandas as pd
```

```python
obj = pd.Series([1, "John", 3.5, "Hey"])
obj
```

```python
obj[0]
```

```python
obj.values
```

```python
obj2 = pd.Series([1,"John", 3.5, "Hey"], index = ["a","b","c","d"])
obj2
```

```python
obj2["b"]
```

```python
obj2.index 
```

```python
score={"Jane":90, "Bill":80, "Tom":75, "Tim":95}
names=pd.Series(score) #Convert to Series
```

```python
names
```

```python
names["Tim"]
```

```python
names[names>=85]
```

```python
names["Tom"]=60
names
```

```python

names[names<=80]=83
names
```

```python
"Tom" in names
```

```python
"Can" in names
```

```python
names/10
```

```python
names**2
```

```python
names.isnull()
```

```python

```
