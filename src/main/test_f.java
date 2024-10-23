package main;
import matrix.Matrix;
import matrix.LinearSystem;
import java.util.Scanner;
import matrix.Interpolation;

public class test_f {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        // // Input ukuran matriks
        System.out.println("Masukkan jumlah baris: ");
        int rows = scanner.nextInt();
        System.out.println("Masukkan jumlah kolom: ");
        int cols = scanner.nextInt();

        // // Inisialisasi matriks
        // Matrix m = new Matrix(rows, cols);
        Matrix matrix2 = new Matrix(rows, cols);
        // // Matrix m3 = new Matrix(rows, cols);

        // // // Input elemen matriks
        // System.out.println("Masukkan elemen matriks: ");
        // m.readMatrix(scanner);
        // m.toRowEchelonForm();
        // m.displayMatrix();
        // LinearSystem ls = new LinearSystem(m);
        // ls.solveAndPrintSolutionType("Gauss");

        // // // Input elemen matriks
        System.out.println("Masukkan elemen matriks ke-2: ");
        matrix2.readMatrix(scanner);

        matrix2.toReducedRowEchelonForm();
        System.out.println("Matriks setelah diubah menjadi bentuk eselon baris tereduksi: ");
        matrix2.displayMatrix();
        LinearSystem ls2 = new LinearSystem(matrix2);
        ls2.solveAndPrintSolutionType("Gauss-Jordan");
//        matrix2.displayMatrix()
//        double det = matrix2.determinantRedRow();
//        System.out.println("det:" + det);
        // // System.out.println("Masukkan elemen matriks ke-3: ");
        // // m3.readMatrix(scanner);

        // // LinearSystem ls3 = new LinearSystem(m3);
        // // ls3.solveAndPrintSolutionType("Cramer");

        // // Matrix m4 = new Matrix(rows, cols);
        // // System.out.println("Masukkan elemen matriks ke-4: ");
        // // m4.readMatrix(scanner);

        // // LinearSystem ls4 = new LinearSystem(m4);
        // // ls4.solveAndPrintSolutionType("Inverse");
        
        // // test for interpolation
        double var;
        System.out.println("Masukkan jumlah titik: ");
        int n = scanner.nextInt();
        double[] a = new double[n];
        double[] b = new double[n];
        System.out.println("Masukkan titik-titiknya: ");
        for (int i = 0; i < n; i++) {
            a[i] = scanner.nextDouble();
            b[i] = scanner.nextDouble();
        }

        System.out.println("Masukkan titik yang ingin diinterpolasi: ");
        var = scanner.nextDouble();
        
        Interpolation interpolation = new Interpolation(a, b);
        double[] polynomial = interpolation.getPolynomial();
        System.out.print("f(x) = ");
        for (int i = polynomial.length - 1; i >= 0 ; i--) {
            if (i != 0 && polynomial[i] != 0) {
                System.out.print(polynomial[i] + "x^" + i + " + ");
            } else if (i == 0 && polynomial[i] != 0) {
                System.out.print(polynomial[i]);
            }
        }
        System.out.print(", f(" + var + ") = " +interpolation.interpolate(var));
        scanner.close();

        // Matrix X = new Matrix(16,16);
        
        // Formula:
        // mode 0: f(x,y) = Σ (j = [0:3]) Σ (i = [0:3]) aij . x^i . y^j
        // mode 1: fx(x,y) = Σ (j = [0:3]) Σ (i = [0:3]) aij . i . x^i-1 . y^j
        // mode 2: fy(x,y) = Σ (j = [0:3]) Σ (i = [0:3]) aij . j . x^i . y^j-1
        // mode 3: fxy(x,y) = Σ (j = [0:3]) Σ (i = [0:3]) aij . ij . x^i-1 . y^j-1
        // (x = [0:1], y = [0:1])

        // int x = 0;
        // int y = 0;
        // int i = 0;
        // int j = 0;
        // int mode = 0;
        // double val = 0;


        // for (int row = 0; row < 16; row++){
        //     for (int col = 0; col < 16; col++){
        //         val = 0;
        //         System.out.println("Row: " + row + " Col: " + col + " i: " + i + " j: " + j + " x: " + x + " y: " + y + " mode: " + mode);
        //         if (mode == 0){
        //             val = Math.pow(x, i) * Math.pow(y, j);
        //         } 
        //         else if (mode == 1){
        //             val = i * Math.pow(x, i-1) * Math.pow(y, j);
        //         } 
        //         else if (mode == 2){
        //             val = j * Math.pow(x, i) * Math.pow(y, j-1);
        //         } 
        //         else if (mode == 3){
        //             val = i * j * Math.pow(x, i-1) * Math.pow(y, j-1);
        //         }

        //         if (Double.isNaN(val)){
        //             val = 0;
        //         }

        //         if (i < 3){
        //             i++;
        //         }
        //         else if (i == 3 && j < 3){
        //             i = 0;
        //             j++;    
        //         }
        //         else if(i == 3 && j == 3){
        //             i = 0;
        //             j = 0;
        //         }

        //         X.setElmt(row, col, val);
        //     }

        //     if (x == 0 && y == 0){
        //         x++;
        //     }
        //     else if (x == 1 && y == 0){
        //         x = 0;
        //         y++;
        //     }
        //     else if (x == 0 && y == 1){
        //         x++;
        //     }
        //     else if (x == 1 && y == 1){
        //         x = 0;
        //         y = 0;
        //         mode++;
        //     }

        // }
        // Matrix matrix = new Matrix(16, 16); // Create an instance of the Matrix class
        // Matrix identity = matrix.createIdentityMatrix(); // Call the createIdentityMatrix() method on the instance
        // Matrix augmented = new Matrix(16, 32);
        // augmented = Matrix.augmentedMatrix(X, identity);
        // augmented.toReducedRowEchelonForm();
        // augmented.displayMatrix();
    }
}

/*
1 2 3 4 5
4 9 7 6 4
3 5 2 1 2
7 5 6 2 9
5 3 4 1 2

1 2 3 4
4 9 7 6
3 5 2 1
7 5 6 2

1 2 3
4 9 7
6 4 2

1 2
3 4
*/