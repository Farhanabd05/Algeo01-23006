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
import matrix.Interpolation;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.List;
import java.util.ArrayList;
import java.util.Objects;

public class InterpolationController {
    private Scene scene;
    private Stage stage;
    private Parent root;

    @FXML
    private TextArea inputArea;

    @FXML
    private TextArea outputArea;

    @FXML
    private void onBackButtonClick(ActionEvent event) throws IOException {
        root = FXMLLoader.load(Objects.requireNonNull(getClass().getResource("/app/matrixapp/menuInterpolationView.fxml")));
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
    private void handleCalculate() {
        String input = inputArea.getText();
        if (input.isEmpty()) {
            showAlert("Invalid Input", "Please enter data in the input area.");
            return;
        }

        try {
            String[] lines = input.split("\n");
            List<Double> xList = new ArrayList<>();
            List<Double> yList = new ArrayList<>();

            for (int i = 0; i < lines.length - 1; i++) {
                String[] point = lines[i].trim().split("\\s+");
                if (point.length != 2) {
                    throw new IllegalArgumentException("Each line (except the last) should contain exactly two numbers.");
                }
                xList.add(Double.parseDouble(point[0]));
                yList.add(Double.parseDouble(point[1]));
            }

            double xToInterpolate = Double.parseDouble(lines[lines.length - 1].trim());

            double[] x = xList.stream().mapToDouble(Double::doubleValue).toArray();
            double[] y = yList.stream().mapToDouble(Double::doubleValue).toArray();

            Interpolation interpolation = new Interpolation(x, y);
            double[] coefficients = interpolation.getPolynomial();
            double interpolatedValue = interpolation.interpolate(xToInterpolate);

            StringBuilder result = new StringBuilder();
            result.append("Interpolation polynomial:\n");
            result.append("f(x) = ");
            for (int i = coefficients.length - 1; i >= 0; i--) {
                if (i < coefficients.length - 1) {
                    result.append(coefficients[i] >= 0 ? " + " : " - ");
                }
                result.append(String.format("%.4f", Math.abs(coefficients[i])));
                if (i > 0) {
                    result.append("x");
                    if (i > 1) {
                        result.append("^").append(i);
                    }
                }
            }
            result.append("\n\n");
            result.append(String.format("f(%.2f) = %.4f", xToInterpolate, interpolatedValue));

            outputArea.setText(result.toString());
        } catch (Exception e) {
            showAlert("Calculation Error", "An error occurred during calculation: " + e.getMessage());
        }
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
        inputArea.setText("");
        outputArea.setText("");
    }
}