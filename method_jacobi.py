import numpy as np

def jacobi_method(A, B, nx, ny, nz, tol=1e-10, max_iter=1000):
    """
    Método iterativo de Jacobi para resolver sistemas lineares Ax = B.

    Parâmetros:
    A: numpy.ndarray
        Matriz de coeficientes (quadrada).
    B: numpy.ndarray
        Matriz ou vetor independente.
    nx, ny, nz: int
        Dimensões da matriz de saída.
    tol: float
        Tolerância para convergência (erro máximo).
    max_iter: int
        Número máximo de iterações permitidas.

    Retorna:
    numpy.ndarray
        Matriz de resultados com dimensões nx, ny, nz.
    """
    # Validação inicial
    n = A.shape[0]
    if A.shape[1] != n or B.shape[0] != n:
        raise ValueError("Dimensões de A e B são inconsistentes.")

    # Matriz inicial de resultados
    X = np.zeros((n, 1))
    X_new = np.zeros_like(X)

    for it in range(max_iter):
        for i in range(n):
            # Calcula a soma excluindo o elemento diagonal
            sigma = sum(A[i, j] * X[j, 0] for j in range(n) if j != i)
            X_new[i, 0] = (B[i] - sigma) / A[i, i]

        # Verifica convergência
        if np.linalg.norm(X_new - X, ord=np.inf) < tol:
            break

        # Atualiza X
        X = X_new.copy()

    else:
        print("Aviso: O método de Jacobi não convergiu.")

    # Redimensiona o resultado para nx, ny, nz
    return X_new.reshape(nx, ny, nz)