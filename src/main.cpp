#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <chrono>
#include <omp.h>

using namespace std;

using Matrix = vector<vector<double>>;

Matrix generateMatrix(int n) {
    Matrix mat(n, vector<double>(n));
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 10.0);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            mat[i][j] = dis(gen);

    return mat;
}

Matrix multiplyOMP(const Matrix& A, const Matrix& B, int num_threads) {
    int n = A.size();
    Matrix C(n, vector<double>(n, 0));

    omp_set_num_threads(num_threads);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = 0; k < n; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }

    return C;
}

int main() {
    vector<int> sizes = {200, 400, 800, 1200, 1600, 2000, 2500, 3000, 3500, 4000};
    vector<int> threads = {1, 2, 4, 8, 16, 20};

    for (int n : sizes) {
        cout << "\nMatrix size: " << n << "x" << n << endl;

        Matrix A = generateMatrix(n);
        Matrix B = generateMatrix(n);

        for (int t : threads) {
            auto start = chrono::high_resolution_clock::now();

            Matrix C = multiplyOMP(A, B, t);

            auto end = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = end - start;

            cout << "Threads: " << t
                 << " | Time: " << elapsed.count() << " sec" << endl;
        }
    }

    return 0;
}