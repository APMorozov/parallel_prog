#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>

using namespace std;

using Matrix = vector<double>;

void generateMatrix(Matrix& mat, int n) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 10.0);

    for (int i = 0; i < n * n; i++)
        mat[i] = dis(gen);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<int> sizes = {200, 400, 800, 1200, 1600, 2000};

    if (rank == 0) {
        cout << "Processes: " << size << endl;
        cout << "-----------------------------" << endl;
    }

    for (int n : sizes) {

        Matrix A, B(n * n), C;

        int rows_per_proc = n / size;
        int remainder = n % size;

        int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);

        vector<int> sendcounts(size), displs(size);

        int offset = 0;
        for (int i = 0; i < size; i++) {
            int rows = rows_per_proc + (i < remainder ? 1 : 0);
            sendcounts[i] = rows * n;
            displs[i] = offset;
            offset += rows * n;
        }

        Matrix local_A(local_rows * n);
        Matrix local_C(local_rows * n, 0);

        if (rank == 0) {
            A.resize(n * n);
            generateMatrix(A, n);
            generateMatrix(B, n);
        }

        MPI_Bcast(B.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        MPI_Scatterv(A.data(), sendcounts.data(), displs.data(),
                     MPI_DOUBLE,
                     local_A.data(), local_rows * n,
                     MPI_DOUBLE,
                     0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        for (int i = 0; i < local_rows; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    local_C[i * n + j] += local_A[i * n + k] * B[k * n + j];
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double end = MPI_Wtime();

        if (rank == 0) {
            C.resize(n * n);
        }

        MPI_Gatherv(local_C.data(), local_rows * n, MPI_DOUBLE,
                    C.data(), sendcounts.data(), displs.data(),
                    MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            cout << "Size: " << n
                 << " | Time: " << (end - start) << " sec" << endl;
        }
    }

    MPI_Finalize();
    return 0;
}