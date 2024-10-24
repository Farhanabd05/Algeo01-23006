// matrixApp/src/main/java/matrix/Matrix.java
package matrix;


import java.util.Scanner;

public class Matrix {
    private double[][] data;
    private int rows;
    private int cols;
    private static final double EPSILON = 1e-10;
    // ----------------------KONSTRUKTOR DAN SELEKTOR----------------------
    
    //Konstruktor matriks dengan ukuran tertentu
    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows][cols];
    }
    //convert to matrix
    public Matrix toMatrix(double[][] data) {
        Matrix res = new Matrix(data.length, data[0].length);
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                res.setElmt(i, j, data[i][j]);
            }
        }
        return res;
    }
    //Input matriks dari user
    public void readMatrix(Scanner scanner) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                while (!scanner.hasNextDouble()) {
                    System.out.println("Input tidak valid. Masukkan angka:");
                    scanner.next(); // membuang input yang tidak valid
                }
                this.setElmt(i, j, scanner.nextDouble());

            }
        }
    }

    //buat matriks identitas
    public Matrix createIdentityMatrix() {
        Matrix identity = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            identity.setElmt(i, i, 1);
        }
        return identity;
    }

    //Copy matriks
    public Matrix copyMatrix() {
        Matrix copy = new Matrix(this.rows, this.cols);
        for (int i = 0; i < getRows(); i++) {
            for (int j = 0; j < getCols(); j++) {
                copy.data[i][j] = this.data[i][j];
            }
        }
        return copy;
    }

    //set elemen matriks pada (row, col)
    public void setElmt(int row, int col, double val) {
        if (isIdxValid(row, col)) {
            data[row][col] = val;
        } else {
            throw new IllegalArgumentException("Invalid index");
        }
    }

    //get elemen matriks pada (row, col)
    public double getElmt(int row, int col) {
        if (isIdxValid(row, col)) {
            return data[row][col];
        } else {
            throw new IllegalArgumentException("Invalid index");
        }
    }

    //get elemen diagonal matriks pada baris & kolom ke-i
    public double getElmtDiagonal(int i) {
        if (i >= 0 && i < Math.min(rows, cols)) {
            return data[i][i];
        } else {
            throw new IllegalArgumentException("Invalid diagonal index");
        }
    }

    //get jumlah row matrix
    public int getRows() {
        return this.rows;
    }


    //get matrix baris ke-row
    public double[] getRow(int row) {
        if (isRowIdxValid(row)) {
            double[] result = new double[cols];
            for (int i = 0; i < cols; i++) {
                result[i] = data[row][i];
            }
            return result;
        } else {
            throw new IllegalArgumentException("Invalid row index");
        }
    }

    //set matrix baris ke-row, input array of double
    public void setRow(int row, double[] values) {
        if (isRowIdxValid(row)) {
            for (int i = 0; i < cols; i++) {
                data[row][i] = values[i];
            }
        } else {
            throw new IllegalArgumentException("Invalid row index");
        }
    }

    //get jumlah kolom matrix
    public int getCols() {
        return this.cols;
    }


    //get matrix kolom ke-col
    public double[] getCol(int col) {
        if (isColIdxValid(col)) {
            double[] result = new double[rows];
            for (int i = 0; i < rows; i++) {
                result[i] = data[i][col];
            }
            return result;
        } else {
            throw new IllegalArgumentException("Invalid col index");
        }
    }

    //set matrix kolom ke-col, input array of double
    public void setCol(int col, double[] values) {
        if (isColIdxValid(col)) {
            for (int i = 0; i < rows; i++) {
                data[i][col] = values[i];
            }
        } else {
            throw new IllegalArgumentException("Invalid col index");
        }
    }


    //get last row index
    public int getLastRowIdx() {
        return this.rows - 1;
    }

    //get last col index
    public int getLastColIdx() {
        return this.cols - 1;
    }

    //get matrix data dalam 2d list
    public double[][] getData() {
        return this.data;
    }
    public void fillMatrix(double value) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = value;
            }
        }
    }
    //swap baris row1 dan row2
    public void swapRows(int row1, int row2) {
        if (isRowIdxValid(row1) && isRowIdxValid(row2)) {
            double[] temp = data[row1];
            data[row1] = data[row2];
            data[row2] = temp;
        } else {
            throw new IllegalArgumentException("Invalid row indices");
        }
    }

    //swap kolom col1 dan col2
    public void swapCols(int col1, int col2) {
        if (isColIdxValid(col1) && isColIdxValid(col2)) {
            double[] temp = data[col1];
            data[col1] = data[col2];
            data[col2] = temp;
        } else {
            throw new IllegalArgumentException("Invalid column indices");
        }
    }

    //----------------------DISPLAY MATRIX----------------------

    //display matriks ke layar, dengan format 2 angka di belakang koma
    //setiap elemen pada sebuah baris dipisah dengan spasi
    public void displayMatrix(){
        for(int i = 0; i < getRows(); i++){
            for(int j = 0; j < getCols(); j++){
                System.out.print(String.format("%.2f", this.getElmt(i, j)));
                if (j != getCols() - 1) {
                    System.out.print(" ");
                }
            }
            System.out.println();
        }
    }


    //----------------------ERROR HANDLING (BOOL)----------------------
    
    //cek apakah matriks bentuk persegi
    public boolean isSquare() {
        return rows == cols;
    }

    //check apakah matriks valid
    public boolean isValid() {
        return rows > 0 && cols > 0;
    }

    //check apakah index baris valid
    public boolean isRowIdxValid(int row) {
        return row >= 0 && row < rows;
    }

    //check apakah index kolom valid
    public boolean isColIdxValid(int col) {
        return col >= 0 && col < cols;
    }

    //check apakah index baris dan kolom valid
    public boolean isIdxValid(int row, int col) {
        return isRowIdxValid(row) && isColIdxValid(col);
    }

    //check apakah matriks identitas
    public boolean isMatrixIdentity() {
        if (!isSquare())
            return false;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (i == j && data[i][j] != 1)
                    return false;
                if (i != j && data[i][j] != 0)
                    return false;
            }
        }
        return true;
    }

    //check apakah size matriks sama
    public boolean isMatrixSizeEqual(Matrix m2) {
        return this.rows == m2.rows && this.cols == m2.cols;
    }

    //cek apakah matriks sama dengan matriks m2
    public boolean isMatrixEqual(Matrix m2) {
        if (this.isMatrixSizeEqual(m2)) {
            for (int i = 0; i < getRows(); i++) {
                for (int j = 0; j < getCols(); j++) {
                    if (this.data[i][j] != m2.data[i][j]) {
                        return false;
                    }
                }
            }
            return true;
        } else {
            return false;
        }
    }

    //cek apakah matriks ada inverse
    public boolean isInvertible(Matrix m2) {
        return m2.determinant() != 0;
    }

    //----------------------ARITHMETIC APPLICATION----------------------

    //tambah atau kurang matriks dengan matriks m2
    //plus = true -> tambah, plus = false -> kurang
    public Matrix addSubMatrix(Matrix m2, boolean plus) {
        Matrix res = new Matrix(this.rows, this.cols);
        if (!isMatrixSizeEqual(m2)) {
            throw new UnsupportedOperationException(
                    "Addition and subtraction only for matrices with the same dimensions");
        }
        for (int i = 0; i < getRows(); i++) {
            for (int j = 0; j < getCols(); j++) {
                res.data[i][j] = plus ? m2.data[i][j] + this.data[i][j] : this.data[i][j] - m2.data[i][j];
            }
        }
        return res;
    }

    //mengalikan matriks dengan konstanta
    public void multiplyMatrixConst(double constant) {
        for (int i = 0; i < getRows(); i++) {
            for (int j = 0; j < getCols(); j++) {
                this.data[i][j] *= constant;
            }
        }
    }

    //mengalikan matriks dengan matriks m2
    public Matrix multiplyMatrix(Matrix m2) {
        if (this.cols != m2.rows) {
            throw new IllegalArgumentException(
                    "Matrix multiplication requires the number of columns of the first matrix equal the number of rows of the second matrix.");
        }
        Matrix result = new Matrix(this.rows, m2.cols);
        for (int i = 0; i < result.getRows(); i++) {
            for (int j = 0; j < result.getCols(); j++) {
                for (int k = 0; k < this.cols; k++) {
                    result.data[i][j] += this.data[i][k] * m2.data[k][j];
                }
            }
        }
        return result;
    }

    // ----------------------MATRIX MODIFICATION----------------------
    
    //get transpose matriks ukuran rows x cols
    public Matrix transpose() {
        Matrix transposed = new Matrix(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed.data[j][i] = this.data[i][j];
            } 
        }
        return transposed;
    }

    //get transpose matriks persegi, ukuran non-persegi tidak valid
    public void transposeInPlace() {
        if (rows != cols) {
            throw new UnsupportedOperationException("In-place transpose only supported for square matrices");
        }
        for (int i = 0; i < rows; i++) {
            for (int j = i + 1; j < cols; j++) {
                double temp = data[i][j];
                data[i][j] = data[j][i];
                data[j][i] = temp;
            }
        }
    }

    //get matriks adjoint
    public Matrix adjoint(){
        return this.cofactor().transpose();
    }

    //get matriks inverse
    public Matrix inverse(){
        if (this.determinant() == 0){
            throw new UnsupportedOperationException("Determinant is zero, inverse does not exist");
        }
        Matrix result = this.adjoint();
        result.multiplyMatrixConst(1/this.determinant());
        return result;
    }

    //get matriks inverse dengan matriks eselon tereduksi
    public Matrix inverseRedRow(){
        //testing skip
//        if (this.determinantRedRow() == 0){
//            throw new UnsupportedOperationException("Determinant is zero, inverse does not exist");
//        }
        //create empty matrix as output
        Matrix inverse = new Matrix(this.rows, this.cols);

        Matrix identity = createIdentityMatrix();

        // identity.displayMatrix();
        // System.out.println("--------------------------------------------------------------------------------------------------------------------------");
        // this.displayMatrix();
        // System.out.println("--------------------------------------------------------------------------------------------------------------------------");
        Matrix combine = augmentedMatrix(this, identity);

        // Matrix combine2 = combine;
        // combine2.toRowEchelonForm();
        // combine2.displayMatrix();
        // System.out.println("--------------------------------------------------------------------------------------------------------------------------");
        combine.toReducedRowEchelonForm();

        // combine.displayMatrix();
        // System.out.println("--------------------------------------------------------------------------------------------------------------------------");

        splitAugmentedMatrix(combine, identity, inverse);

        return inverse;
    }
              
    //get augmented matriks dari matriks m1 dan m2
    //return matriks baru
    public static Matrix augmentedMatrix(Matrix m1, Matrix m2) {
        if (m1.getRows() != m2.getRows()) {
            throw new IllegalArgumentException("The number of rows in the matrices must be equal");
        }
        Matrix m = new Matrix(m1.getRows(), m1.getCols() + m2.getCols());
        for (int i = 0; i < m1.getRows(); i++) {
            for (int j = 0; j < m1.getCols(); j++) {
                m.setElmt(i, j, m1.getElmt(i, j));
            }
            for (int j = 0; j < m2.getCols(); j++) {
                m.setElmt(i, m1.getCols() + j, m2.getElmt(i, j));
            }
        }
        return m;


    }

    //split augmented matriks menjadi matriks m1 dan m2
    //tidak return matriks, langsung set m1 dan m2
    public static void splitAugmentedMatrix(Matrix m, Matrix m1, Matrix m2) {
        for (int i = 0; i < m1.getRows(); i++) {
            for (int j = 0; j < m1.getCols(); j++) {
                m1.setElmt(i, j, m.getElmt(i, j));
            }
            for (int j = 0; j < m2.getCols(); j++) {
                m2.setElmt(i, j, m.getElmt(i, m1.getCols() + j));
            }
        }
    }

    // -------------------------------------Determinant-------------------------------------------

    //get determinant matriks, hanya matriks square
    // menggunakan metode kofaktor
    public double determinant() {
        if (!isSquare()) {
            throw new UnsupportedOperationException("Determinant only supported for square matrices");
        }
        if (rows == 1)  return data[0][0];
        else if (rows == 2) return data[0][0] * data[1][1] - data[0][1] * data[1][0];
        else {
            double result = 0;
            for (int i = 0; i < rows; i++) {
                result += Math.pow(-1, i) * data[i][0] * minor(i, 0).determinant();
            }
            return result;
        }
    }

    // menggunakan metode reduction row
    public double determinantRedRow() {
        if (!isSquare()) {
            throw new UnsupportedOperationException("Determinant only supported for square matrices");
        }
        
        Matrix m = this.copyMatrix();
        int n = m.getRows();
        double det = 1.0;

        // Kasus khusus untuk matriks 1x1 dan 2x2
        if (n == 1) return getElmt(0, 0);
        if (n == 2) return getElmt(0, 0) * getElmt(1, 1) - getElmt(0, 1) * getElmt(1, 0);

        for (int pivotRow = 0; pivotRow < n; pivotRow++) {
            int pivotCol = m.findLeftmostNonZeroColumn(pivotRow);
            
            if (pivotCol == -1) return 0; // Matrixnya singular
            
            if (pivotRow != pivotCol) {
                m.swapRows(pivotRow, pivotCol);
                det *= -1; //mengalikan determinan dengan -1 karena ada swap row
            }
            
            m.ensureNonZeroPivot(pivotRow, pivotCol);
            det *= m.getElmt(pivotRow, pivotCol);

            m.dividePivotRow(pivotRow, pivotCol);
            m.subtractPivotRow(pivotRow, pivotCol);
        }

        return det;
    }


    //get minor matriks pada baris i dan kolom j
    public Matrix minor(int i,int j){
        if(!isSquare() || rows == 1){
            throw new UnsupportedOperationException("Minor only supported for square matrices");
        }
        Matrix result = new Matrix(rows - 1, cols - 1);
        int row_pos, col_pos;
        row_pos = 0;
        col_pos = 0;
        for (int k = 0; k < rows; k++) {
            for (int l = 0; l < cols; l++) {
                if (k != i && l != j) {
                    result.setElmt(row_pos, col_pos, this.getElmt(k, l));
                    col_pos++;
                }
            }
            if (k != i) {
                row_pos++;
            }
            col_pos = 0;
        }
        return result;
    }


    //get matriks kofaktor
    public Matrix cofactor(){
        if(!isSquare() || rows == 1){
            throw new UnsupportedOperationException("Cofactor only supported for square matrices");
        }
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (this.getElmt(i, j) == 0) {
                    result.setElmt(i, j, 0);
                } else {
                    result.setElmt(i, j, Math.pow(-1, i + j) * (this.minor(i, j).determinant()));
                }
            }
        }
        return result;
    }

    // ----------------------ROW ELEMENTARY OPERATIONS----------------------

    //cek apakah baris row nol semua
    //untuk operasi baris elementer solusi parametrik
    public boolean isRowAllZero(int row) {
        for (int j = 0; j < getCols(); j++) {
            if (data[row][j] != 0) {
                return false;
            }
        }
        return true;
    }
    // Fungsi utama untuk mengubah matriks menjadi bentuk eselon baris
    public void toRowEchelonForm() {
        int lead = 0;
        for (int r = 0; r < rows; r++) {
            if (lead >= cols) return;
            int i = findLeftmostNonZeroColumn(r);
            if (i != r) swapRows(i, r);
            ensureNonZeroPivot(r, lead);
            dividePivotRow(r, lead);
            subtractPivotRow(r, lead);
            lead++;
        }
    }
    // Fungsi baru untuk mencari kolom non-zero paling kiri
    public int findLeftmostNonZeroColumn(int startRow) {
        for (int j = startRow; j < cols; j++) {
            for (int i = startRow; i < rows; i++) {
                if (Math.abs(data[i][j]) > EPSILON) { // Menggunakan toleransi untuk floating-point
                    return i;
                }
            }
        }
        return startRow;
    }
     // Fungsi baru untuk menukar baris jika elemen pivot adalah nol
    public void ensureNonZeroPivot(int pivotRow, int pivotCol) {
        if (Math.abs(data[pivotRow][pivotCol]) < EPSILON) {
            for (int i = pivotRow + 1; i < rows; i++) {
                if (Math.abs(data[i][pivotCol]) > EPSILON) {
                    swapRows(pivotRow, i);
                    return;
                }
            }
        }
    }
    // Fungsi untuk membagi baris dengan pivotnya
    public void dividePivotRow(int pivotRow, int pivotCol) {
        double pivot = data[pivotRow][pivotCol];
        if (Math.abs(pivot) > EPSILON) {
            for (int j = pivotCol; j < cols; j++) {
                data[pivotRow][j] /= pivot;
            }
        }
    }
    // Fungsi untuk mengurangkan baris pivot dari baris-baris di bawahnya  
    public void subtractPivotRow(int pivotRow, int pivotCol) {
        for (int i = pivotRow + 1; i < rows; i++) {
            double factor = data[i][pivotCol];
            for (int j = pivotCol; j < cols; j++) {
                data[i][j] -= factor * data[pivotRow][j];
            }
        }
    }

    // Fungsi untuk mengubah matriks eselon baris menjadi bentuk eselon baris
    // tereduksi
    public void toReducedRowEchelonForm() {
        toRowEchelonForm();
        for (int r = rows - 1; r >= 0; r--) {
            int lead = findPivotColumn(r);
            if (lead != -1) {
                eliminateAbovePivot(r, lead);
            }
            if (isRowAllZero(r)) {
                swapRows(r, rows - 1);
            }
        }
        roundToZero();
    }

// ----------------------COLUMN ELEMENTARY OPERATIONS----------------------
    // Fungsi baru untuk mencari kolom pivot dalam suatu baris
    public int findPivotColumn(int row) {
        for (int j = 0; j < cols; j++) {
            if (Math.abs(data[row][j] - 1.0) < EPSILON) {
                return j;
            }
        }
        return -1; // Tidak ada pivot (baris semua nol)
    }
    // Fungsi untuk mengurangkan baris pivot dari baris-baris di bawahnya
    private void eliminateAbovePivot(int pivotRow, int pivotCol) {
        for (int i = pivotRow - 1; i >= 0; i--) {
            double factor = data[i][pivotCol];
            for (int j = pivotCol; j < cols; j++) {
                data[i][j] -= factor * data[pivotRow][j];
            }
        }
    }
    // Fungsi untuk membulatkan elemen bernilai sangat kecil ke nol
    private void roundToZero() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (Math.abs(data[i][j]) < EPSILON) data[i][j] = 0;
            }
        }
    }

    //get hasil jawaban SPL Gauss-Jordan [x1 .. xn]
    public double[] resFromReducedEselon(Matrix m) {
        double[] Y = new double[m.getRows()];
        for (int i = 0; i < m.getRows(); i++) {
            Y[i] = m.getElmt(i, m.getCols() - 1);
        }
        return Y;
    }

    //get hasil jawaban SPL Gauss [x1 .. xn]
    public double[] backwardSubstitution(double[][] U, double[] Y, boolean isParametric) {
        int N;
        if (U[0].length > U.length + 1) {
            N = U.length;
        } else {
            if (isParametric) {
                N = U[0].length - 1;
            } else {
                if (U[0].length != U.length) {
                    N = U[0].length;
                } else {
                    N = U.length;
                }
            }
        }
        double[] X = new double[N];

        // Inisialisasi X dengan Y
        for (int i = 0; i < N; i++) {
            X[i] = Y[i];
        }

        // Backward substitution
        for (int j = N - 1; j >= 0; j--) {
            X[j] /= U[j][j];
            for (int i = 0; i < j; i++) {
                X[i] -= U[i][j] * X[j];
            }
        }
        return X;
    }

    //Hitung rank setelah matrix dibuat jadi matrix eselon baris 
    public int computeRank() {
        Matrix gauss = new Matrix(rows, cols);
        gauss = copyMatrix();
        gauss.toRowEchelonForm();

        int rank = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (Math.abs(gauss.getElmt(i, j)) > EPSILON) {
                    rank++;
                    break;
                }
            }
        }
        return rank;
    }
}

