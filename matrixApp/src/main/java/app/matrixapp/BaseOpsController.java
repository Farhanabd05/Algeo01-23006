
package app.matrixapp;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Node;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.TextArea;
import javafx.scene.control.Alert;
import javafx.scene.control.Alert.AlertType;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import matrix.Matrix;
import matrix.LinearSystem;
import javafx.scene.control.ComboBox;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.List;
import java.util.Arrays;
import java.util.Objects;

import javafx.scene.control.*;

public class BaseOpsController {

    private Scene scene;
    private Stage stage;
    private Parent root;
    @FXML
    public Label operationLabel;
    @FXML
    public Button backButton;
    @FXML
    private TextArea inputArea;

    @FXML
    private TextArea outputArea;
    @FXML
    private ComboBox<String> subOperationComboBox;

    private String currentOperation = "";
    private String currentSubOperation = "";

    @FXML
    private void onBackButtonClick(ActionEvent event) throws IOException {
        root = FXMLLoader.load(Objects.requireNonNull(getClass().getResource("/app/matrixapp/homeView.fxml")));
        stage = (Stage)((Node)event.getSource()).getScene().getWindow();
        double previousWidth = stage.getWidth();
        double previousHeight = stage.getHeight();
        scene = new Scene(root);
        stage.setScene(scene);
        stage.setWidth(previousWidth);
        stage.setHeight(previousHeight);
        stage.show();
    }
    @FXML
    private void handleMainMenu(ActionEvent event) {
        String operation = ((MenuItem) event.getSource()).getText();
        currentOperation = operation;
        operationLabel.setText(currentOperation);
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
        try {
            matrix = parseMatrix(input);
        } catch (IllegalArgumentException e) {
            showAlert("Invalid Input", "Unable to parse input as a matrix. Please check your input format.");
            return;
        }

        String result = "";
        try {
            result = switch (currentOperation) {
                case "Sistem Persamaan Linier" -> solveSPL(matrix, subOperation);
                case "Determinan" -> calculateDeterminant(matrix, subOperation);
                case "Matriks Balikan" -> calculateInverse(matrix, subOperation);
                default -> "Please select an operation first.";
            };
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
        String[] lines = input.split("\n");
        int rows = lines.length;
        int cols = lines[0].trim().split("\\s+").length;
        Matrix matrix = new Matrix(rows, cols);

        for (int i = 0; i < rows; i++) {
            String[] elements = lines[i].trim().split("\\s+");
            if (elements.length != cols) {
                throw new IllegalArgumentException("Inconsistent number of columns in the input.");
            }
            for (int j = 0; j < cols; j++) {
                try {
                    matrix.setElmt(i, j, Double.parseDouble(elements[j]));
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException("Invalid number format in the input.");
                }
            }
        }

        return matrix;
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
        operationLabel.setText("");
        subOperationComboBox.getItems().clear();
        subOperationComboBox.setDisable(true);
    }
}
