from .matrix import Matrix

__version__ = "0.1.0"

__all__ = [
    "Matrix",
    "det",
    "trace",
    "transpose",
    "inverse",
    "hadamard",
    "identity",
    "zeros",
    "ones",
]

def _ensure_matrix(data):
    """Helper to convert raw list of lists to Matrix if needed."""
    if isinstance(data, Matrix):
        return data
    return Matrix(data)

def det(data):
    """Calculates the determinant of the given matrix or data."""
    return _ensure_matrix(data).det()

def transpose(data):
    """Returns the transpose of the given matrix or data."""
    return _ensure_matrix(data).transpose()

def inverse(data):
    """Returns the inverse of the given matrix or data."""
    return _ensure_matrix(data).inverse()

def hadamard(a, b):
    """Returns the element-wise (Hadamard) product of two matrices or list-of-lists."""
    return _ensure_matrix(a).hadamard(_ensure_matrix(b))

def trace(data):
    """Returns the trace (sum of main diagonal) of the given matrix or data."""
    return _ensure_matrix(data).trace()

def identity(n):
    """Returns an n x n identity matrix."""
    return Matrix.identity(n)

def zeros(rows, cols):
    """Returns a rows x cols zero matrix."""
    return Matrix.zeros(rows, cols)

def ones(rows, cols):
    """Returns a rows x cols matrix of ones."""
    return Matrix.ones(rows, cols)
