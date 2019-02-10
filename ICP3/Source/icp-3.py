"""
3. Numpy

Using NumPy create random vector of size 15 having only Integers in the range 0-20.

Write a program to find the most frequent item/value in the vector list.

"""
import numpy as np

x = np.random.random_integers(20, size=(1, 15))
print("Original array:", x)

(values, counts) = np.unique(x, return_counts=True)
ind = np.argmax(counts)
print(values[ind])
