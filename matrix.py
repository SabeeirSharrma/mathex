class Matrix:
    def __init__(self, data):
        if not all(isinstance(row, list) for row in data):
            raise ValueError("Data must be a list of lists.")
        if not all(len(row) == len(data[0]) for row in data):
            raise ValueError("All rows must have the same length.")
        
        self.data = [row[:] for row in data]  # Copy data
        self.rows = len(data)
        self.cols = len(data[0]) if self.rows > 0 else 0

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
        """Element-wise multiplication (if Matrix) or scalar multiplication."""
        if isinstance(other, (int, float)):
            result = [[self.data[r][c] * other for c in range(self.cols)] 
                      for r in range(self.rows)]
            return Matrix(result)
        elif isinstance(other, Matrix):
            if self.rows != other.rows or self.cols != other.cols:
                raise ValueError("Matrix dimensions must match for element-wise multiplication.")
            result = [[self.data[r][c] * other.data[r][c] for c in range(self.cols)] 
                      for r in range(self.rows)]
            return Matrix(result)
        return NotImplemented

    def __rmul__(self, other):
        """Right-side scalar multiplication."""
        return self.__mul__(other)

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

    def _get_minor(self, r, c):
        """Returns the minor matrix after removing row r and column c."""
        return [row[:c] + row[c+1:] for row in (self.data[:r] + self.data[r+1:])]

    def det(self):
        """Calculates the determinant of the matrix (must be square)."""
        if self.rows != self.cols:
            raise ValueError("Determinant only defined for square matrices.")
        
        if self.rows == 1:
            return self.data[0][0]
        if self.rows == 2:
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        
        determinant = 0
        for c in range(self.cols):
            minor = self._get_minor(0, c)
            determinant += ((-1) ** c) * self.data[0][c] * Matrix(minor).det()
        return determinant

    def inverse(self):
        """Returns the inverse of the matrix (must be square and non-singular)."""
        if self.rows != self.cols:
            raise ValueError("Inverse only defined for square matrices.")
        
        determinant = self.det()
        if determinant == 0:
            raise ValueError("Matrix is singular and cannot be inverted.")
        
        if self.rows == 1:
            return Matrix([[1 / determinant]])

        # Adjugate matrix / Cofactor expansion
        cofactors = []
        for r in range(self.rows):
            cofactor_row = []
            for c in range(self.cols):
                minor = self._get_minor(r, c)
                cofactor_row.append(((-1) ** (r + c)) * Matrix(minor).det())
            cofactors.append(cofactor_row)
        
        adjugate = Matrix(cofactors).transpose()
        return adjugate * (1 / determinant)