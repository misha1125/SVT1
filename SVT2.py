import numpy as np
import scipy as sp
import scipy.sparse as sparse
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import nquad
import scipy.sparse.linalg as slin

dx = 1e-1
dy = 0.1

# для красоты
X_bounds = (0, 200)
Y_bounds = (-100, 100)

N = 400


c_0 = 1
a = 20

h = 200 / N
t_h = 1*h

T_N = int(50/t_h)

B = 1


P = h / dx  # Peclet number
print(P)
supg = True
if supg and P >= 1:
    delta_e = (h - dx) / 2
else:
    delta_e = 0


def to_coord(i, j):
    return i * 200 / N, j * 200 / N - 100


def bounds_conditions(i, j):
    if abs(to_coord(i, j)[1]) < a and i == 0:
        return c_0
    return 0


def test_inside(i, j):
    return 0 < i < N and 0 < j < N


def test_in(i, j):
    return 0 <= i <= N and 0 <= j <= N


def to_multiindex(i, j):
    return (i-1) + (N-1) * (j-1)




def integrate(i, j, k, l):
    integral = np.array([[1 / 36, 1 / 9, 1 / 36],
                                 [1 / 9, 4 / 9, 1 / 9],
                                 [1 / 36, 1 / 9, 1 / 36]]).T * (h ** 2)
    reg_integral = np.array([[1 / 12, 0, -1 / 12],
                                   [1 / 3, 0, -1 / 3],
                                   [1 / 12, 0, -1 / 12]]).T
    delta_x = k - i
    delta_y = l - j
    if abs(delta_x) <= 1 and abs(delta_y) <= 1:
        return (integral[1 + delta_x, 1 + delta_y]+ delta_e*h*reg_integral[1+delta_x, 1+delta_y])
    return 0


def integrate_with_diff_operator(i, j, k, l):
    integral_x = np.array([[-1 / 6, 1 / 3, -1 / 6],
                                     [-2 / 3, 4 / 3, -2 / 3],
                                     [-1 / 6, 1 / 3, -1 / 6]]).T
    integral_y = np.array([[-1 / 6, -2 / 3, -1 / 6],
                                     [1 / 3, 4 / 3, 1 / 3],
                                     [-1 / 6, -2 / 3, -1 / 6]]).T
    delta_x = k - i
    delta_y = l - j
    if abs(delta_x) <= 1 and abs(delta_y) <= 1:
        return (integral_x[1 + delta_x, 1 + delta_y] * dx + integral_y[1 + delta_x, 1 + delta_y] * dy)*t_h
    return 0


def integrate_with_b(i, j, k, l):
    integral = np.array([[-1 / 12, 0, 1 / 12],
                       [-1 / 3, 0, 1 / 3],
                       [-1 / 12, 0, 1 / 12]]).T * h
    reg_int = np.array([[-1 / 6, 1 / 3, -1 / 6],
              [-2 / 3, 4 / 3, -2 / 3],
              [-1 / 6, 1 / 3, -1 / 6]]).T

    delta_x = k - i
    delta_y = l - j
    if abs(delta_x) <= 1 and abs(delta_y) <= 1:
        return B *(integral[1 + delta_x, 1 + delta_y] + delta_e*reg_int[1 + delta_x, 1 + delta_y]) * t_h
    return 0


def create_matrix():


    def add_cheme(i, j, k, l):
        if test_in(i, j) and test_in(k, l):
            val = integrate(i, j, k, l)
            C_1 = integrate(i, j, k, l)
            C_2 = integrate_with_b(i, j, k, l)
            C_3 = integrate_with_diff_operator(i, j, k, l)
            if test_inside(k, l):
                if val != 0:
                    row.append(to_multiindex(i, j))
                    col.append(to_multiindex(k, l))
                    values.append(C_1 + C_2 + C_3)
                    b[to_multiindex(i, j)] += (C_1) * last_solution[k-1, l-1]
            else:
                b[to_multiindex(i, j)] += -(C_2 + C_3)*bounds_conditions(k, l)

    b = np.zeros(shape=(N-1) ** 2)
    row = []
    col = []
    values = []
    last_solution = np.zeros((N - 1, N - 1))
    for t in range(T_N):
        row = []
        col = []
        values = []
        b = np.zeros(shape=(N-1) ** 2)

        for i in range(1, N):
            for j in range(1, N):
                for k in range(i - 1, i + 2):
                    for l in range(j - 1, j + 2):
                        add_cheme(i, j, k, l)
        row = np.array(row)
        col = np.array(col)
        values = np.array(values)


        matrix = sparse.csc_matrix((values, (row, col)), shape=((N-1) ** 2, (N-1) ** 2))
        #precond = slin.LinearOperator((N**2,N**2), slin.spilu(matrix).solve)
        #x, info = slin.cg(matrix, b, M = precond, tol=1e-16)
        #x = slin.spsolve(matrix, b)
        ILU = slin.spilu(matrix, fill_factor=1.0, drop_tol=1e-3)
        prec = slin.LinearOperator(((N - 1) ** 2, (N - 1) ** 2), matvec=ILU.solve)
        x, info = slin.gmres(matrix, b, M=prec, tol=1e-6)
        print(np.linalg.norm(matrix @ x - b), info)
        last_solution = x.reshape((N-1, N-1)).T

    return last_solution


x = create_matrix().T


bound_matrix = np.zeros((N-1, N-1))
for i in range(1, N):
    for j in range(1, N):
        bound_matrix[i-1,j-1] = bounds_conditions(i, j)

# sns.heatmap(x, yticklabels=np.arange(-100, 100, 200/N), xticklabels=np.arange(0, 200, 200/N))
sns.heatmap(x[:, :], cmap="viridis")
#plt.imshow(x)
plt.show()

#sns.heatmap(bound_matrix.T, cmap="viridis")
#plt.show()
