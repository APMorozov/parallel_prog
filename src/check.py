import numpy as np

def read_matrix(filename):
    with open(filename, 'r') as f:
        n = int(f.readline())
        data = []
        for _ in range(n):
            row = list(map(float, f.readline().split()))
            data.append(row)
    return np.array(data)

if __name__ == "__main__":
    A = read_matrix("A.txt")
    B = read_matrix("B.txt")
    C_cpp = read_matrix("C.txt")

    C_py = np.dot(A, B)

    if np.allclose(C_cpp, C_py, atol=1e-6):
        print("True")
    else:
        print("False")