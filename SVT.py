import scipy.sparse as spr
import scipy.sparse.linalg as slin
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dx = 1
dy = 1
h = 1/32
N = int(1/h)
def C(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def f(x, y):
    return np.pi * np.pi * (dx + dy) * np.sin(np.pi * x) * np.sin(np.pi * y)

def N1(y):
    return np.pi * np.sin(np.pi * y)

def N2(x):
    return np.pi * np.sin(np.pi * x)

def matrix_index_to_coord(ind):
    return ind//(N+1), ind % (N+1)

def coord_to_matrix_index(x, y):
    return x * (N + 1) + y

data = []
row_ind = []
col_ind = []
right_part = [0] * ((N+1) * (N+1))

# i, j - номер уравнения
# k, n - добавляемый элемент
def add_point_to_matrix(val, i, j, k, n):
    data.append(val)
    row_ind.append(coord_to_matrix_index(i, j))
    col_ind.append(coord_to_matrix_index(k, n))

# Добавляем разностное уравнение
for i in range(1, N):
    for j in range(1, N):
        right_part[coord_to_matrix_index(i, j)] = (f(i * h, j * h))

        add_point_to_matrix(-dx / h ** 2, i, j, i + 1, j)
        add_point_to_matrix(2 * (dx+dy) / h ** 2, i, j, i, j)
        add_point_to_matrix(-dx / h ** 2, i, j, i - 1, j)

        add_point_to_matrix(-dy / h ** 2, i, j, i, j + 1)
        add_point_to_matrix(-dy / h ** 2, i, j, i, j - 1)

# Добавляем граничные условия первого рода
for i in range(0, N+1):
    j = 0
    add_point_to_matrix(1, i, j, i, j)


for j in range(0, N+1):
    i = N
    add_point_to_matrix(1, i, j, i, j)
for j in range(0, N+1):
    i = 0
    add_point_to_matrix(1, i, j, i, j)


#Добавляем граничные условия второго рода
for j in range(1, N):
    i = 0
    right_part[coord_to_matrix_index(i, j)] = f(i*h, j*h) / 2 + N1(j * h) / h * dx
    add_point_to_matrix(-dx / h**2, i, j, i, j)
    add_point_to_matrix(dx / h**2, i, j, i+1, j)

    add_point_to_matrix(-dy / 2 / h ** 2, i, j, i, j + 1)
    add_point_to_matrix(dy / h ** 2, i, j, i, j)
    add_point_to_matrix(-dy / 2 / h ** 2, i, j, i, j - 1)


for i in range(1, N):
    j = N
    right_part[coord_to_matrix_index(i, j)] = f(i*h, j*h) / 2 + N2(i * h) / h * dy
    add_point_to_matrix(-dy / h**2, i, j, i, j)
    add_point_to_matrix(dy / h**2, i, j, i, j - 1)

    add_point_to_matrix(-dx / 2 / h ** 2, i, j, i + 1, j)
    add_point_to_matrix(dx / h ** 2, i, j, i, j)
    add_point_to_matrix(-dx / 2 / h ** 2, i, j, i - 1, j)


data = np.array(data ,dtype=float)
row_ind = np.array(row_ind, dtype=int)
col_ind = np.array(col_ind, dtype=int)
right_part = np.array(right_part)



# решаем систеу с разреженной матрицей
total_csr = spr.csr_matrix((data, (row_ind,col_ind )))
print(slin.norm(total_csr - total_csr.transpose()))
#print(total_csr.shape, right_part.shape)
x, info = slin.bicg(A = total_csr, b = right_part,x0=np.zeros(right_part.shape), tol=1e-6)
print(info, np.linalg.norm(total_csr@x - right_part))
#преобразем x обратно в решение и вычисляем ошибку

x = x.reshape((N+1, N+1))
max_er = 0.0
avr_error = 0.0
for i in range(0, N+1):
    for j in range(0, N+1):
        error = np.abs(C(i*h, j*h) - x[i, j])
        max_er = max(max_er, error)
        avr_error += error*error

sns.heatmap(x)
plt.show()

sol = x.copy()
for i in range(0, N+1):
    for j in range(0, N+1):
        sol[i,j] = C(i*h, j*h)

sns.heatmap(sol)
plt.show()

sns.heatmap(np.abs(sol-x))
plt.show()

print("max error", max_er, "avr sqrt", np.sqrt(avr_error)/(N+1)**2)