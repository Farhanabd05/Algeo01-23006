package matrix;

public class LinearSystem extends Matrix {
    
    private Matrix features; // A in Ax=b
    private Matrix target; // b in Ax=b

    public LinearSystem(Matrix matrix) {
        super(matrix.getRows(), matrix.getCols());
        features = new Matrix(getRows(), getCols()-1 );
        for (int i = 0; i < getRows(); i++) {
            features.setRow(i, matrix.getRow(i));
        }
        target = new Matrix(getRows(), 1);
        target.setCol(0, matrix.getCol(getCols()-1));
    }

    public Matrix getFeatures() {
        return features;
    }
    
    public Matrix getTarget() {
        return target;
    }
    
    public void printFeatures() {
        features.displayMatrix();
    }

    public void printTarget() {
        target.displayMatrix();
    }

    public double[] gauss(Matrix features, Matrix target){
        features = getFeatures();
        target = getTarget();
        Matrix gaussMatrix = augmentedMatrix(features, target);
        gaussMatrix.toRowEchelonForm();
        return backwardSubstitution(gaussMatrix.getData(), target.getCol(0));
    }

    public double[] gaussJordan(Matrix features, Matrix target){
        features = getFeatures();
        target = getTarget();
        Matrix gaussJordanMatrix = augmentedMatrix(features, target);
        gaussJordanMatrix.toReducedRowEchelonForm();
        return gaussJordanMatrix.getCol(gaussJordanMatrix.getCols()-1);
    }

    public Matrix CramerRule() {
        Matrix result = new Matrix(features.getRows(), 1);
        for (int i = 0; i < features.getRows(); i++) {
            Matrix temp = features.copyMatrix();
            temp.setCol(i, target.getCol(0));
            result.setElmt(i, 0, temp.determinant() / features.determinant());
        }
        return result;
    }
}
