class Matrix:
    def __init__(self, data):
        if not isinstance(data, list) or not all(isinstance(row, list) for row in data):
            raise TypeError("Data must be a list of lists.")
        self.rows = len(data)
        self.cols = len(data[0]) if self.rows > 0 else 0
        if self.rows > 0 and not all(len(row) == self.cols for row in data):
            raise ValueError("All rows must have the same length.")
        self.data = [row[:] for row in data]  # Defensive copy

    def __repr__(self):
        return f"Matrix({self.data})"

    def __str__(self):
        if self.rows == 0:
            return "[]"
        max_len = max(len(str(item)) for row in self.data for item in row)
        formatted_rows = [
            "[" + " ".join(f"{str(item).rjust(max_len)}" for item in row) + "]"
            for row in self.data
        ]
        return "\n".join(formatted_rows)

    def __eq__(self, other):
        if not isinstance(other, Matrix):
            return False
        return self.data == other.data

    def __add__(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix dimensions must match for addition.")
        
        result = [[self.data[r][c] + other.data[r][c] for c in range(self.cols)] 
                  for r in range(self.rows)]
        return Matrix(result)

    def __sub__(self, other):
        if not isinstance(other, Matrix):
            return NotImplemented
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix dimensions must match for subtraction.")
        
        result = [[self.data[r][c] - other.data[r][c] for c in range(self.cols)] 
                  for r in range(self.rows)]
        return Matrix(result)

    def __mul__(self, other):
        """Scalar multiplication: Matrix * scalar."""
        if isinstance(other, (int, float)):
            result = [[self.data[r][c] * other for c in range(self.cols)]
                      for r in range(self.rows)]
            return Matrix(result)
        return NotImplemented

    def __rmul__(self, other):
        """Right-side scalar multiplication: scalar * Matrix."""
        return self.__mul__(other)

    def __neg__(self):
        """Negation: -Matrix."""
        return self * -1

    def __getitem__(self, index):
        """Row access: m[i] returns the i-th row as a list."""
        return self.data[index]

    def hadamard(self, other):
        """Element-wise (Hadamard) product of two same-shape matrices."""
        if not isinstance(other, Matrix):
            raise TypeError("Hadamard product requires another Matrix.")
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix dimensions must match for Hadamard product.")
        result = [[self.data[r][c] * other.data[r][c] for c in range(self.cols)]
                  for r in range(self.rows)]
        return Matrix(result)

    def __matmul__(self, other):
        """Matrix multiplication: (m x n) * (n x p) = (m x p)"""
        if not isinstance(other, Matrix):
            return NotImplemented
        if self.cols != other.rows:
            raise ValueError(f"Matrix dimension mismatch: ({self.rows}x{self.cols}) and ({other.rows}x{other.cols})")
        
        result = [[sum(self.data[r][k] * other.data[k][c] for k in range(self.cols)) 
                  for c in range(other.cols)] 
                  for r in range(self.rows)]
        return Matrix(result)

    def transpose(self):
        """Returns the transpose of the matrix."""
        result = [[self.data[r][c] for r in range(self.rows)] 
                  for c in range(self.cols)]
        return Matrix(result)

    @classmethod
    def identity(cls, n):
        """Returns an n x n identity matrix."""
        data = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        return cls(data)

    @classmethod
    def zeros(cls, rows, cols):
        """Returns a rows x cols zero matrix."""
        data = [[0 for _ in range(cols)] for _ in range(rows)]
        return cls(data)

    @classmethod
    def ones(cls, rows, cols):
        """Returns a rows x cols matrix of ones."""
        data = [[1 for _ in range(cols)] for _ in range(rows)]
        return cls(data)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @property
    def is_square(self):
        """True if the matrix has equal rows and columns."""
        return self.rows == self.cols

    @property
    def is_symmetric(self):
        """True if the matrix equals its own transpose (float-tolerant)."""
        if not self.is_square:
            return False
        return all(
            abs(self.data[r][c] - self.data[c][r]) < 1e-12
            for r in range(self.rows)
            for c in range(self.cols)
        )

    def trace(self):
        """Returns the sum of the main diagonal elements (square matrices only)."""
        if not self.is_square:
            raise ValueError("Trace is only defined for square matrices.")
        return sum(self.data[i][i] for i in range(self.rows))

    def row(self, i):
        """Returns row i as a plain list (0-indexed)."""
        if not (0 <= i < self.rows):
            raise IndexError(f"Row index {i} out of range for {self.rows}x{self.cols} matrix.")
        return self.data[i][:]

    def col(self, j):
        """Returns column j as a plain list (0-indexed)."""
        if not (0 <= j < self.cols):
            raise IndexError(f"Column index {j} out of range for {self.rows}x{self.cols} matrix.")
        return [self.data[i][j] for i in range(self.rows)]

    def _get_minor(self, r, c):
        """Returns the minor matrix after removing row r and column c."""
        return [row[:c] + row[c+1:] for row in (self.data[:r] + self.data[r+1:])]

    def det(self):
        """Calculates the determinant using Gaussian elimination with partial pivoting.

        Complexity: O(n³) — handles large matrices without recursion or stack overflow.
        Returns a float. For integer matrices the result is numerically exact for
        typical sizes since all operations are additions/multiplications.
        """
        if not self.is_square:
            raise ValueError("Determinant only defined for square matrices.")

        n = self.rows
        # Work on a float copy — never mutate self.data
        mat = [[float(self.data[r][c]) for c in range(n)] for r in range(n)]
        sign = 1  # Tracks parity of row swaps; each swap flips the sign

        for col in range(n):
            # Partial pivoting: find the largest absolute value in this column
            # at or below the current pivot row for numerical stability
            pivot_row = max(range(col, n), key=lambda r: abs(mat[r][col]))

            if abs(mat[pivot_row][col]) < 1e-12:
                return 0.0  # Singular matrix

            if pivot_row != col:
                mat[col], mat[pivot_row] = mat[pivot_row], mat[col]
                sign *= -1  # Each row swap negates the determinant

            # Eliminate all entries below the pivot in this column
            pivot = mat[col][col]
            for row in range(col + 1, n):
                factor = mat[row][col] / pivot
                for k in range(col, n):
                    mat[row][k] -= factor * mat[col][k]

        # Det of upper-triangular matrix = product of diagonal entries
        result = sign * 1.0
        for i in range(n):
            result *= mat[i][i]
        return result

    def inverse(self):
        """Returns the inverse using Gauss-Jordan elimination.

        Complexity: O(n³) — augments [A | I] and row-reduces to [I | A⁻¹].
        Raises ValueError for singular or non-square matrices.
        """
        if not self.is_square:
            raise ValueError("Inverse only defined for square matrices.")

        n = self.rows
        # Build augmented matrix [A | I] as floats
        aug = [
            [float(self.data[r][c]) for c in range(n)] + [1.0 if r == c else 0.0 for c in range(n)]
            for r in range(n)
        ]

        for col in range(n):
            # Partial pivoting
            pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))

            if abs(aug[pivot_row][col]) < 1e-12:
                raise ValueError("Matrix is singular (or near-singular) and cannot be inverted.")

            if pivot_row != col:
                aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

            # Scale the pivot row so that pivot = 1
            pivot = aug[col][col]
            aug[col] = [x / pivot for x in aug[col]]

            # Eliminate all other rows (above and below)
            for row in range(n):
                if row == col:
                    continue
                factor = aug[row][col]
                aug[row] = [aug[row][k] - factor * aug[col][k] for k in range(2 * n)]

        # Extract the right half — that's A⁻¹
        result = [aug[r][n:] for r in range(n)]
        return Matrix(result)