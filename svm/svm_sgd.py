import numpy as np
from numpy import random
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from numpy.random import shuffle
import matplotlib.pyplot as plt
import numpy.linalg as la
import sklearn.preprocessing as proc
import csv

def get_accuracy(a, b, X_test, y_test):
    size = len(y_test)
    count = 0
    for i in range(size):
        x = X_test[i]
        real = y_test[i]

        x = np.array(x)
        x = x.reshape(1, 6)

        prediction = x.dot(a.T) + b

        if prediction > 0 and real == 1:
            count += 1
        elif prediction < 0 and real == -1:
            count += 1
    return count / size

data = []
with open("train.txt") as file:
    data = [line.split() for line in file]


X = []
y = []
for line in data:
    numerical = [int(line[0][:-1]), int(line[2][:-1]), int(line[4][:-1]), \
      int(line[10][:-1]), int(line[11][:-1]), int(line[12][:-1])]

    X.append(numerical)
    if line[14] == '<=50K':
        y.append(-1)
    else:
        y.append(1)


a = random.dirichlet(np.ones(6)*1000, size = 1)
b = 0


#scale X
scaler = StandardScaler()
X = scaler.fit_transform(X)

X = X - np.mean(X)

#10% test data and 90% train data
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)


lambdas = [0.001, 0.01, 0.1, 1]

dict_accuracy = {}
for lamb in lambdas:
    dict_accuracy[lamb] = []


dict_a = {}
for lamb in lambdas:
    dict_a[lamb] = []

dict_b = {}
for lamb in lambdas:
    dict_b[lamb] = []

a = 0
b = 0

for lamb in lambdas:

    #a = random.dirichlet(np.ones(6)*1000, size = 1)
    a = np.zeros(6)
    b = 0

    for epoch in range(50):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

        if epoch == 49:
            result = get_accuracy(a, b, X_test, y_test)
            print(str(lamb) + ' : ' + str(result))

        shuffle(X_train)

        validation_train = X_train[0:50]
        validation_test = y_train[0:50]

        train_data = X_train[51:]
        train_test = y_train[51:]

        m = 1
        n = 50
        step_size = m / (0.01 * epoch + n)

        for step in range(500):

            if step % 30 == 0:
                accuracy = get_accuracy(a, b, validation_train, validation_test)

                dict_accuracy[lamb].append(accuracy)
                dict_a[lamb].append(a)
                dict_b[lamb].append(b)

            # current index randomly chosen
            curr = random.randint(0, len(train_data))

            curr_train = np.array(train_data[curr])

            curr_train = curr_train.reshape(1, 6)

            curr_val = (curr_train.dot(a.T) + b) * train_test[curr]

            if curr_val >= 1:
                a = a - np.dot(a, lamb) * step_size
            else:
                a = a - step_size * (np.dot(a, lamb) - np.dot(train_data[curr], train_test[curr]))
                b = b - (step_size * (-train_test[curr]))


'''

x_val = [i for i in range(1, 851)]

# dict_accuracy
fig = plt.figure()
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)

fig.tight_layout()

y1 = dict_accuracy[0.001]
y2 = dict_accuracy[0.01]
y3 = dict_accuracy[0.1]
y4 = dict_accuracy[1]

ax1.plot(x_val, y1, color='m')
ax2.plot(x_val, y2, color='g')
ax3.plot(x_val, y3, color='r')
ax4.plot(x_val, y4, color='b')

ax1.set_xlabel('lambda = 0.001')
ax2.set_xlabel('lambda = 0.01')
ax3.set_xlabel('lambda = 0.1')
ax4.set_xlabel('lambda = 1')

plt.show()

#########################################

a_norm = {}
for lamb in lambdas:
    a_norm[lamb] = []

for lamb in dict_a:
    curr_list = dict_a[lamb]
    for curr in curr_list:
        norm = la.norm(curr, 2)
        a_norm[lamb].append(norm)

plt.plot(x_val, a_norm[0.001], label = 'lambda is 0.001', color = 'b')
plt.plot(x_val, a_norm[0.01], label = 'lambda is 0.01', color = 'r')
plt.plot(x_val, a_norm[0.1], label = 'lambda is 0.01', color = 'g')
plt.plot(x_val, a_norm[1], label = 'lambda is 1', color = 'm')
plt.legend()
plt.show()


'''


lamb = 0.001

a = random.dirichlet(np.ones(6)*1000, size = 1)


b = 0

data = []
with open("train.txt") as file:
    data = [line.split() for line in file]

X = []
y = []
for line in data:
    numerical = [int(line[0][:-1]), int(line[2][:-1]), int(line[4][:-1]), \
      int(line[10][:-1]), int(line[11][:-1]), int(line[12][:-1])]

    X.append(numerical)
    if line[14] == '<=50K':
        y.append(-1)
    else:
        y.append(1)


#scale X
scaler = StandardScaler()
X = scaler.fit_transform(X)

X = X - np.mean(X)

for epoch in range(30):

    if epoch == 29:
        result = get_accuracy(a, b, X_test, y_test)
        print(str(lamb) + ' : ' + str(result))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

    shuffle(X_train)

    validation_train = X_train[0:50]
    validation_test = y_train[0:50]

    train_data = X_train[51:]
    train_test = y_train[51:]

    m = 1
    n = 50
    step_size = m / (0.01 * epoch + n)

    for step in range(300):
        # current index randomly chosen
        curr = random.randint(0, len(train_data))
        curr_train = np.array(train_data[curr])
        curr_train = curr_train.reshape(1, 6)
        curr_val = (curr_train.dot(a.T) + b) * train_test[curr]

        if curr_val >= 1:
            a = a - np.dot(a, lamb) * step_size
        else:
            a = a - step_size * (np.dot(a, lamb) - np.dot(train_data[curr], train_test[curr]))
            b = b - (step_size * (-train_test[curr]))


data = []
with open("test.txt") as file:
    data = [line.split() for line in file]

X = []
y = []
for line in data:
    numerical = [int(line[0][:-1]), int(line[2][:-1]), int(line[4][:-1]), \
      int(line[10][:-1]), int(line[11][:-1]), int(line[12][:-1])]
    X.append(numerical)




prediction = []
for k in X:
    numerical = np.array(k)
    estimate = numerical.dot(a.T) + b
    #print(estimate)
    if estimate < 0:
        prediction.append('<=50K')
    else:
        prediction.append('>50K')


index_final = []
for i in range(len(prediction)):
    index_final.append(["'" + str(i) + "'", prediction[i]])

with open('output.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    writer.writerow(['Example', 'Label'])
    writer.writerows(index_final)
