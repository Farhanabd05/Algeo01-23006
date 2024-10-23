
package app.matrixapp;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.TextArea;
import javafx.scene.control.Alert;
import javafx.scene.control.Alert.AlertType;
import javafx.stage.FileChooser;
import matrix.Matrix;
import matrix.LinearSystem;
import matrix.Interpolation;
import matrix.BicubicalSpline;
import matrix.QuadraticRegressor;
import matrix.LinearRegressor;
import javafx.scene.control.ComboBox;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.List;
import java.util.Arrays;
import javafx.scene.control.*;
import java.util.ArrayList;
public class Controller {

    @FXML
    private TextArea inputArea;

    @FXML
    private TextArea outputArea;
    @FXML
    private ComboBox<String> subOperationComboBox;

    private String currentOperation = "";
    private String currentSubOperation = "";

    @FXML
    private void handleMainMenu(ActionEvent event) {
        String operation = ((MenuItem) event.getSource()).getText();
        currentOperation = operation;
        currentSubOperation = "";
        subOperationComboBox.getItems().clear();
        switch (operation) {
            case "Sistem Persamaan Linier":
                subOperationComboBox.getItems().addAll("Gauss", "GaussJordan", "MatriksBalikan", "Cramer");
                break;
            case "Determinan":
                subOperationComboBox.getItems().addAll("OBE", "Kofaktor");
                break;
            case "Matriks Balikan":
                subOperationComboBox.getItems().addAll("OBE", "Kofaktor");
                break;
            case "Regresi Linier Berganda":
                subOperationComboBox.getItems().addAll("Kuadratik", "Linier");
                break;
            case "Interpolasi Polinom":
            case "Interpolasi BicubicSpline":
            case "Interpolasi Gambar (Bonus)":
                // These operations don't have sub-menus
                subOperationComboBox.setDisable(true);
                break;
            case "Keluar":
                System.exit(0);
                break;
            default:
                showAlert("Invalid Operation", "Please select a valid operation.");
        }
        subOperationComboBox.setDisable(subOperationComboBox.getItems().isEmpty());
        outputArea.setText("Please select a sub-operation and enter the matrix or required data in the input area.");
    }

    @FXML
    private void handleSubOperation(String subOperation) {
        currentSubOperation = subOperation;
        outputArea.setText("Please enter the matrix or required data in the input area.");
    }

    @FXML
    private void handleCalculate() {
        String input = inputArea.getText();
        String subOperation = subOperationComboBox.getValue();
        if (input.isEmpty()) {
            showAlert("Invalid Input", "Please enter data in the input area.");
            return;
        }

        if (subOperation == null && !subOperationComboBox.isDisabled()) {
            showAlert("Invalid Input", "Please select a sub-operation.");
            return;
        }
        Matrix matrix;
        matrix = parseMatrix(input);

        String result = "";
        try {
            switch (currentOperation) {
                case "Sistem Persamaan Linier":
                    result = solveSPL(matrix, subOperation);
                    break;
                case "Determinan":
                    result = calculateDeterminant(matrix, subOperation);
                    break;
                case "Matriks Balikan":
                    result = calculateInverse(matrix, subOperation);
                    break;
                case "Interpolasi Polinom":
                    result = interpolatePolynomial(matrix);
                    break;
                case "Interpolasi Bicubic Spline":
                    result = interpolateBicubicSpline(matrix);
                    break;
//                case "Regresi linier berganda":
//                    result = performRegression(matrix);
//                    break;
                case "Interpolasi Gambar (Bonus)":
                    result = interpolateImage(matrix);
                    break;
                case "Keluar":
                    System.exit(0);
                default:
                    result = "Please select an operation first.";
            }
        } catch (Exception e) {
            showAlert("Calculation Error", "An error occurred during calculation: " + e.getMessage());
            return;
        }

        outputArea.setText(result);
    }

    private String solveSPL(Matrix matrix, String subOperation) {
        LinearSystem ls = new LinearSystem(matrix);
        String solutionType = ls.checkSolutionType();
        StringBuilder result = new StringBuilder();
        result.append("Tipe solusi: ").append(solutionType).append("\n");
        if (solutionType.equals("Tidak ada")) {
            result.append("Sistem tidak memiliki solusi.").append("\n");
            return result.toString();
        }
        double[] solution;
        switch (subOperation) {
            case "Gauss":
                solution = ls.gauss();
                break;
            case "GaussJordan":
                solution = ls.gaussJordan();
                break;
            case "MatriksBalikan":
                solution = ls.inverseMethodSPL(ls.getFeatures(), ls.getTarget());
                break;
            case "Cramer":
                solution = ls.CramerRule().getCol(0);
                break;
            default:
                return "Invalid SPL method selected.";
        }
        if (solutionType.equals("Unik")) {
            result.append(formatUniqueSolution(solution));
        } else if (solutionType.equals("Parametrik")) {
            result.append(formatParametricSolution(ls));
        }
        return result.toString();
    }

    private String formatUniqueSolution(double[] solution) {
        StringBuilder result = new StringBuilder("Solusi unik: \n");
        for (int i = 0; i < solution.length; i++) {
            result.append(String.format("x%d = %.4f\n", i + 1, solution[i]));
        }
        return result.toString();
    }

    private String formatParametricSolution(LinearSystem ls) {
        StringBuilder result = new StringBuilder("Solusi parametrik: \n");
        Matrix augmented = ls.augmentedMatrix(ls.getFeatures(), ls.getTarget());
        augmented.toReducedRowEchelonForm();

        int vars = augmented.getCols() - 1;
        boolean[] isFreeVariable = new boolean[vars];
        Arrays.fill(isFreeVariable, true);
        for (int i = 0; i < augmented.getRows() - 1; i++) {
            int pivotCol = augmented.findPivotColumn(i);
            if (pivotCol != -1 && pivotCol < vars) {
                isFreeVariable[pivotCol] = false;
                result.append(formatEquation(augmented, i, pivotCol, isFreeVariable));
            }
        }
        char nextParam = 's';
        for (int i = 0; i < vars; i++) {
            if (isFreeVariable[i]) {
                result.append(String.format("x%d = %c\n", i + 1, nextParam));
                nextParam++;
            }
        }
        return result.toString();
    }

    private String formatEquation(Matrix augmented, int row, int pivotCol, boolean[] isFreeVariable) {
        StringBuilder eq = new StringBuilder(String.format("x%d = ", pivotCol + 1));
        boolean firstTerm = true;

        // constant term
        double constant = augmented.getElmt(row, augmented.getCols() - 1);
        if (Math.abs(constant) > 1e-10) {
            eq.append(String.format("%.4f", constant));
            firstTerm = false;
        }

        char nextParam = 's';
        for (int j = pivotCol + 1; j < augmented.getCols() - 1; j++) {
            double coeff = -augmented.getElmt(row, j); // Negasi koefisien untuk pindah ruas
            if (Math.abs(coeff) > 1e-10) {
                if (!firstTerm) {
                    eq.append(coeff > 0 ? " + " : " - ");
                } else if (coeff < 0) {
                    eq.append("-");
                }
                if (Math.abs(Math.abs(coeff) - 1.00) > 1e-10) {
                    eq.append(String.format("%.4f", Math.abs(coeff)));
                }
                eq.append(isFreeVariable[j] ? nextParam : String.format("x%d", j + 1));
                if (isFreeVariable[j]) nextParam++;
                firstTerm = false;
            }
        }
        if (firstTerm) eq.append("0");
        return eq.append("\n").toString();
    }

    private String calculateDeterminant(Matrix matrix, String subOperation) {
        switch (subOperation) {
            case "OBE":
                return String.valueOf(matrix.determinantRedRow());
            case "Kofaktor":
                return String.valueOf(matrix.determinant());
            default:
                return "Invalid determinant method selected.";
        }
    }

    private String calculateInverse(Matrix matrix, String subOperation) {
        Matrix inverse;
        switch (subOperation) {
            case "OBE":
                inverse = matrix.inverseRedRow();
                return matrixToString(inverse);
            case "Kofaktor":
                inverse = matrix.inverse();
                return matrixToString(inverse);
            default:
                return "Invalid inverse method selected.";
        }
    }

    private String interpolatePolynomial(Matrix matrix) {
        // Implement polynomial interpolation logic
        return "Polynomial interpolation not implemented yet.";
    }

    private String interpolateBicubicSpline(Matrix matrix) {
        // Implement bicubic spline interpolation logic
        return "Bicubic spline interpolation not implemented yet.";
    }

//    private String performRegression(Matrix matrix) {
//        switch (currentSubOperation) {
//            case "Kuadratik":
//                QuadraticRegressor qr = new QuadraticRegressor(matrix);
//                return qr.regress();
//            case "Linier":
//                LinearRegressor lr = new LinearRegressor(matrix);
//                return lr.regress();
//            default:
//                return "Invalid regression type selected.";
//        }
//    }


    private String interpolateImage(Matrix matrix) {
        // Implement image interpolation logic
        return "Image interpolation not implemented yet.";
    }

    private Matrix parseMatrix(String input) {
        try {
            // Split input into lines and remove empty lines
            String[] lines = input.split("\n");

            // Count maximum columns by checking all rows
            int maxCols = 0;
            for (String line : lines) {
                String[] elements = line.trim().split("\\s+");
                maxCols = Math.max(maxCols, elements.length);
            }

            if (maxCols == 0 || lines.length == 0) {
                throw new IllegalArgumentException("Empty input or invalid matrix format");
            }

            // Create matrix with proper dimensions
            Matrix matrix = new Matrix(lines.length, maxCols);

            // Parse each line
            for (int i = 0; i < lines.length; i++) {
                String[] elements = lines[i].trim().split("\\s+");

                // Fill each row with available elements
                for (int j = 0; j < elements.length; j++) {
                    try {
                        if (!elements[j].isEmpty()) {
                            matrix.setElmt(i, j, Double.parseDouble(elements[j]));
                        }
                    } catch (NumberFormatException e) {
                        // Skip non-numeric values or set to 0
                        matrix.setElmt(i, j, 0);
                    }
                }

                // Fill remaining columns with 0 if needed
                for (int j = elements.length; j < maxCols; j++) {
                    matrix.setElmt(i, j, 0);
                }
            }

            return matrix;
        } catch (Exception e) {
            throw new IllegalArgumentException("Error parsing matrix: " + e.getMessage());
        }
    }
    private String matrixToString(Matrix matrix) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < matrix.getRows(); i++) {
            for (int j = 0; j < matrix.getCols(); j++) {
                sb.append(String.format("%.2f", matrix.getElmt(i, j)));
                if (j < matrix.getCols() - 1) sb.append(" ");
            }
            sb.append("\n");
        }
        return sb.toString();
    }

    @FXML
    private void handleReadFile() {
        FileChooser fileChooser = new FileChooser();
        File file = fileChooser.showOpenDialog(null);
        if (file != null) {
            try {
                List<String> lines = Files.readAllLines(file.toPath());
                inputArea.setText(String.join("\n", lines));
            } catch (IOException e) {
                showAlert("File Read Error", "Error reading file: " + e.getMessage());
            }
        }
    }

    private void showAlert(String title, String content) {
        Alert alert = new Alert(AlertType.ERROR);
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(content);
        alert.showAndWait();
    }

    public void init() {
        currentOperation = "";
        inputArea.setText("");
        outputArea.setText("");
        subOperationComboBox.getItems().clear();
        subOperationComboBox.setDisable(true);
    }

    public void saveOutput() {
        try {
            FileChooser fileChooser = new FileChooser();
            fileChooser.setTitle("Save Output");
            fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("Text Files", "*.txt"));
            File file = fileChooser.showSaveDialog(stage);
            if (file != null) {
                Files.write(file.toPath(), outputArea.getText().getBytes());
            }
        } catch (IOException e) {
            showAlert("File Save Error", "Error saving file: " + e.getMessage());
        }
    }
}
