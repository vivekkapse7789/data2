public class MultithreadedMatrixMultiplication {

    static class RowThread extends Thread {
        int[][] a, b, result;
        int row;

        RowThread(int[][] a, int[][] b, int[][] result, int row) {
            this.a = a;
            this.b = b;
            this.result = result;
            this.row = row;
        }

        public void run() {
            for (int j = 0; j < b[0].length; j++) {
                for (int k = 0; k < b.length; k++) {
                    result[row][j] += a[row][k] * b[k][j];
                }
            }
        }
    }

    static void displayMatrix(int[][] matrix) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                System.out.print(matrix[i][j] + " ");
            }
            System.out.println();
        }
    }

    public static void main(String[] args) throws InterruptedException {

        int[][] a = {
            {2, 1, 3},
            {4, 0, 2}
        };

        int[][] b = {
            {5, 2},
            {1, 4},
            {3, 7}
        };

        if (a[0].length != b.length) {
            System.out.println("Matrix multiplication is not possible.");
            return;
        }

        int rows = a.length;
        int cols = b[0].length;

        int[][] normalResult = new int[rows][cols];
        int[][] threadResult = new int[rows][cols];

        System.out.println("Matrix A:");
        displayMatrix(a);

        System.out.println("\nMatrix B:");
        displayMatrix(b);

        long startTime = System.nanoTime();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                for (int k = 0; k < b.length; k++) {
                    normalResult[i][j] += a[i][k] * b[k][j];
                }
            }
        }

        double normalTime =
                (System.nanoTime() - startTime) / 1000000.0;

        startTime = System.nanoTime();

        RowThread[] threads = new RowThread[rows];

        for (int i = 0; i < rows; i++) {
            threads[i] = new RowThread(a, b, threadResult, i);
            threads[i].start();
        }

        for (int i = 0; i < rows; i++) {
            threads[i].join();
        }

        double threadTime =
                (System.nanoTime() - startTime) / 1000000.0;

        System.out.println("\nNormal Multiplication Result:");
        displayMatrix(normalResult);

        System.out.println("\nMultithreaded Multiplication Result:");
        displayMatrix(threadResult);

        System.out.println("\nPerformance Comparison:");
        System.out.println("Normal execution time: "
                + normalTime + " ms");
        System.out.println("Multithreaded execution time: "
                + threadTime + " ms");

        System.out.println("\nBoth results are "
                + (java.util.Arrays.deepEquals(normalResult, threadResult)
                ? "identical." : "different."));
    }

}
