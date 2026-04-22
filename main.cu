#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <cuda_runtime.h>

using namespace std;

using Vec = vector<int>;

#define CUDA_OK(x) if ((x) != cudaSuccess) { \
    cerr << "CUDA failure at line " << __LINE__ << endl; exit(1); }

Vec makeMatrix(int n, int a, int b) {
    mt19937 gen(random_device{}());
    uniform_int_distribution<int> d(a, b);

    Vec m(n * n);
    for (int i = 0; i < n * n; ++i)
        m[i] = d(gen);

    return m;
}

void dumpMatrix(const Vec& m, int n, const string& name) {
    ofstream f(name);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            f << m[i*n + j] << " ";
        f << "\n";
    }
}

__global__ void matmulKernel(const int* A, const int* B, int* C, int n) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < n && c < n) {
        int acc = 0;
        for (int k = 0; k < n; ++k)
            acc += A[r*n + k] * B[k*n + c];

        C[r*n + c] = acc;
    }
}

int main() {
    int n = 1000;
    int bx = 16, by = 16;

    Vec A = makeMatrix(n, 0, 9);
    Vec B = makeMatrix(n, 0, 9);
    Vec C(n*n, 0);

    size_t sz = n*n*sizeof(int);

    int *dA, *dB, *dC;
    CUDA_OK(cudaMalloc(&dA, sz));
    CUDA_OK(cudaMalloc(&dB, sz));
    CUDA_OK(cudaMalloc(&dC, sz));

    CUDA_OK(cudaMemcpy(dA, A.data(), sz, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dB, B.data(), sz, cudaMemcpyHostToDevice));

    dim3 blk(bx, by);
    dim3 grd((n+bx-1)/bx, (n+by-1)/by);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    cudaEventRecord(t0);

    matmulKernel<<<grd, blk>>>(dA, dB, dC, n);

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);

    CUDA_OK(cudaMemcpy(C.data(), dC, sz, cudaMemcpyDeviceToHost));

    long long ops = 2LL * n * n * n;

    cout << "N = " << n << endl;
    cout << "Time = " << ms << " ms" << endl;

    ofstream out("results.csv", ios::app);
    out << n << "," << bx << "x" << by << "," << ops << "," << ms << "\n";

    dumpMatrix(C, n, "out.txt");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}