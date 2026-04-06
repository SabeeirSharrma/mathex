# 📚 Mathex Library

A pure Python mathematics library for matrix manipulation, algebra, trigonometry and more.

## 🛠 Features

- **Simplified API**: Top-level functions like `mathex.det()` and `mathex.transpose()` that accept raw list-of-lists.
- **Matrix Operations**: Matrix multiplication (`@`), Transposition, Determinant, Inverse.
- **Utility Factories**: Identity matrix, Zeros matrix, Ones matrix.
- **Readable Representation**: Formatted string output for easy debugging.

## ⚙ Usage

### Using Matrix Objects
```python
from mathex import Matrix

m1 = Matrix([[1, 2], [3, 4]])
m2 = Matrix.identity(2)

# Arithmetic and Matrix Multiplication
sum_m = m1 + m2
prod_m = m1 @ m1  

print(m1.det())       # -2.0
print(m1.transpose())
```

### Simplified Interaction (Raw Data)
```python
import mathex

data = [[1, 2], [3, 4]]

# Get answers directly for any size matrix
d = mathex.det(data)
t = mathex.transpose(data)
i = mathex.inverse(data)
```

## 📚 References

### `mathex.Matrix(data)`
Class to handle matrix operations through objects and operator overloading.

### `mathex.det(data)`
Returns the determinant of the given matrix or list-of-lists.

### `mathex.transpose(data)`
Returns the transpose of the given matrix or list-of-lists.

### `mathex.inverse(data)`
Returns the inverse of the given matrix or list-of-lists.

### `mathex.identity(n)` / `mathex.zeros(r, c)` / `mathex.ones(r, c)`
Factory functions to create common matrices.

## 🚀 Installation

Run
```bash
pip install mathex
```
in terminal.

## 📝 License

This project is made under the MIT License