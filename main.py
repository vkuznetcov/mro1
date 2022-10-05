import math
import numpy as np
from matplotlib import pyplot as plt




def generate_vector(M, B, n, N):  # пункт 1
    a = np.zeros((2, 2))
    x = np.zeros((n, N))

    a[0, 0] = np.sqrt(B[0, 0])
    a[0, 1] = 0
    a[1, 0] = B[0, 1] / np.sqrt(B[0, 0])
    a[1, 1] = np.sqrt(B[1, 1] - B[0, 1] ** 2 / B[0, 0])

    # print (a)

    for i in range(0, N):
        x[:, i] = np.column_stack(np.dot(a, np.random.normal(0, 1, size=(n, 1))) + M)  # 3стр
    return x  # случайный вектор с норм законом распределения


def estimate_math_expectation(x, N, n):  # оценка мат ожидания
    estimate = np.zeros((2, 1))

    for i in range(0, n):
        for j in range(0, N):
            estimate[i, :] += x[i, j]  # стр 5 1 формула

    return estimate / N


def correlation_matrix_estimate(x, M, N, n):  # оценка корреляционной матрицы
    estimate = np.dot(x, x.transpose())
    return estimate / N - np.dot(M, M.transpose())  # стр 5 2 формула


def Bhatacharya(M1, M2, B1, B2):  # cтр 6 (1)
    return (0.25 * np.dot(np.dot((M1 - M2).transpose(), np.linalg.inv((B1 + B2) / 2)), (M1 - M2))
            + 0.5 * math.log(np.linalg.det((B1 + B2) / 2) / np.sqrt(np.linalg.det(B1) * np.linalg.det(B2))))


def Mahalanobis(M1, M2, B):  # cтр 6 (2)
    return np.dot(np.dot((M1 - M2).transpose(), np.linalg.inv(B)), (M1 - M2))


if __name__ == '__main__':
    n = 2
    N = 200

    M1 = np.array([[0], [0]])
    M2 = np.array([[1], [-1]])
    M3 = np.array([[1], [1]])

    # кор матрица
    B1 = np.array([[0.2, 0.1], [0.1, 0.2]])
    B2 = np.array([[0.05, 0.03], [0.03, 0.05]])
    B3 = np.array([[0.05, 0.005], [0.005, 0.05]])

    Y1 = generate_vector(M1, B2, n, N)
    Y2 = generate_vector(M2, B2, n, N)

    X1 = generate_vector(M1, B1, n, N)
    X2 = generate_vector(M2, B2, n, N)
    X3 = generate_vector(M3, B3, n, N)

    fig, axes = plt.subplots(1, 2)
    axes[0].scatter(Y1[0, :], Y1[1, :], c='purple')
    axes[0].scatter(Y2[0, :], Y2[1, :], c='blue')
    axes[0].set_title('С равными коррел. матрицами')
    axes[1].scatter(X1[0, :], X1[1, :], c='yellow')
    axes[1].scatter(X2[0, :], X2[1, :], c='purple')
    axes[1].scatter(X3[0, :], X3[1, :], c='pink')
    axes[1].set_title('С разными коррел. матрицами')
    plt.show()

    m1 = estimate_math_expectation(Y1, N, n)
    m2 = estimate_math_expectation(Y2, N, n)
    m3 = estimate_math_expectation(X1, N, n)
    m4 = estimate_math_expectation(X2, N, n)
    m5 = estimate_math_expectation(X3, N, n)

    print('\nОценка математического ожидания:\nМ1 по Y1:', m1, 'М2 по Y2:', m2, 'М1 по X1:', m3, 'М2 по X2:', m4,
          'М3 по X3:', m5, sep='\n')

    print('\nОценка корреляционной матрицы:\nB2 по Y1:', correlation_matrix_estimate(Y1, m1, N, n), 'B2 по Y2:',
          correlation_matrix_estimate(Y2, m2, N, n),
          'B1 по X1:', correlation_matrix_estimate(X1, m3, N, n), 'В2 по X2:',
          correlation_matrix_estimate(X2, m4, N, n), 'В3 по X3:', correlation_matrix_estimate(X3, m5, N, n),
          sep='\n')

    print('\nРасстояние Бхатачария 1-2: ', Bhatacharya(m1, m2, B1, B2), '\nРасстояние Бхатачария 2-3: ',
          Bhatacharya(M2, M3, B2, B3),
          '\nРасстояние Бхатачария 1-3: ', Bhatacharya(M1, M3, B1, B3))

    print('\nРасстояние Махаланобиса: ', Mahalanobis(M1, M2, B3))

    np.savetxt("Y1.csv", Y1, delimiter='|')
    np.savetxt("Y2.csv", Y2, delimiter='|')
    np.savetxt("X1.csv", X1, delimiter='|')
    np.savetxt("X2.csv", X2, delimiter='|')
    np.savetxt("X3.csv", X3, delimiter='|')
