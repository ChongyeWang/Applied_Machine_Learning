from sklearn.decomposition import PCA
import csv
import numpy as np
import numpy.linalg as la
from sklearn.metrics import mean_squared_error as mse


def generate_matrix(name):
    result = []
    with open(name, 'r') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        count = 0
        for line in data:
            if count == 0:
                count += 1
                continue
            list = []
            for integer in line:
                list.append(float(integer))
            result.append(list)
    return result

def dim_reduction(original, iris, dim, is_iris):
    X = []
    mean = 0
    if is_iris == True:
        X = np.cov(iris.T)
        mean = iris.mean(axis=0)
    else:
        X = np.cov(original.T)
        mean = original.mean(axis=0)
    eigenvalue, eigenvector = la.eig(X)
    idx = eigenvalue.argsort()[::-1]
    eigenvalue = eigenvalue[idx]
    eigenvector = eigenvector[:,idx]
    mse = 0
    new_x = []
    for i in range(len(original)):
        x = np.zeros(4)
        for j in range(dim):
            x = x + np.dot(eigenvector[:, j].T, original[i] - mean) * eigenvector[:, j]
        x = x + mean
        new_x.append(x)
        mse += la.norm(iris[i] - x)**2
    mse = mse / len(original)
    return new_x, mse

iris = np.array(generate_matrix('iris.csv'))
dataI = np.array(generate_matrix('dataI.csv'))
dataII = np.array(generate_matrix('dataII.csv'))
dataIII = np.array(generate_matrix('dataIII.csv'))
dataIV = np.array(generate_matrix('dataIV.csv'))
dataV = np.array(generate_matrix('dataV.csv'))

result = np.zeros((5, 10))
all_data = [dataI, dataII, dataIII, dataIV, dataV]
count = 0
for i in range(0, 5):
    data = all_data[count]
    count += 1
    for j in range(0, 5):
        result[i][j] = dim_reduction(data, iris, j, True)[1]
        result[i][j + 5] = dim_reduction(data, iris, j, False)[1]


with open('chongye2-numbers.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    writer.writerow(['0N', '1N', '2N', '3N', '4N', '0c', '1c', '2c', '3c', '4c'])
    writer.writerows(result)

reconstrution_II = dim_reduction(all_data[1], iris, 2, False)[0]

with open('chongye2-recon.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    writer.writerow(['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width'])
    writer.writerows(reconstrution_II)
