#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <chrono>
#include <cstdio>

using namespace std;


vector<vector<double>> generateMatrix(int n) {
    vector<vector<double>> mat(n, vector<double>(n));
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 10.0);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            mat[i][j] = dis(gen);

    return mat;
}

void writeMatrix(const string& filename, const vector<vector<double>>& mat) {
    ofstream out(filename);
    int n = mat.size();
    out << n << endl;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            out << mat[i][j] << " ";
        }
        out << endl;
    }
}

vector<vector<double>> readMatrix(const string& filename) {
    ifstream in(filename);
    int n;
    in >> n;

    vector<vector<double>> mat(n, vector<double>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            in >> mat[i][j];

    return mat;
}

vector<vector<double>> multiply(const vector<vector<double>>& A,
                                const vector<vector<double>>& B) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0));

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j];

    return C;
}

int main() {
    int n;
    cout << "Введите размерность матриц: ";
    cin >> n;

    auto A = generateMatrix(n);
    auto B = generateMatrix(n);

    writeMatrix("A.txt", A);
    writeMatrix("B.txt", B);

    auto start = chrono::high_resolution_clock::now();

    auto C = multiply(A, B);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    writeMatrix("C.txt", C);
    
    cout << "Matrix size: " << n << "x" << n << endl;
    cout << "Operations: " << n * n * n << endl;
    cout << "Time: " << elapsed.count() << " seconds" << endl;

    FILE* pipe = popen("python3 check.py", "r");
    if (!pipe) {
        cerr << "Error running Python script" << endl;
        return 1;
    }

    char buffer[128];
    string result = "";

    while (fgets(buffer, sizeof(buffer), pipe) != NULL) {
        result += buffer;
    }

    pclose(pipe);

    if (!result.empty() && result.back() == '\n')
        result.pop_back();

    cout << "Python check result: " << result << endl;

    if (result != "True") {
    cerr << "Verification FAILED!" << endl;
    exit(1);
    } else {
        cout << "Verification OK!" << endl;
    }

    return 0;
}