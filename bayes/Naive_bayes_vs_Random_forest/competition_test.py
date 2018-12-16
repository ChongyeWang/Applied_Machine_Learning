import matplotlib.pyplot as plt
from sklearn.svm import SVC
import csv
import numpy as np
from scipy.misc import imresize
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
import matplotlib.pyplot as plt

def get_middle(matrix):
    middle = []
    for i in range(4, 24):
        for j in range(4, 24):
            index = i * 28 + j
            middle.append(matrix[index])

    return np.array(middle).astype(np.float)


def stretch(matrix):
    reshaped_matrix = matrix.reshape(20, 20)
    left = 20
    right = -1
    up = 20
    down = -1
    for i in range(20):
        for j in range(20):
            if reshaped_matrix[i][j] != 0:
                if i < up: up = i
                if i > down: down = i
                if j < left: left = j
                if j > right: right = j

    height = down - up + 1
    width = right - left + 1
    stretched_matrix = reshaped_matrix[left:right + 1, up:down + 1]
    new_stretched = imresize(stretched_matrix, (20, 20))
    new_stretched = new_stretched.reshape(1, 400)[0]


    return new_stretched

train_data = []
train_label = []
test_data = []
test_label = []

#read train data
with open('train.csv', 'r') as csvfile:
    data = csv.reader(csvfile, delimiter=',')
    count = 0
    for row in data:
        if count == 0:
            count += 1
            continue
        curr = np.array(row[2:]).astype(np.float)
        train_data.append(curr)
        train_label.append(row[1])

#read test data
with open('val.csv', 'r') as csvfile:
    data = csv.reader(csvfile, delimiter=',')
    count = 0
    for row in data:
        if count == 0:
            count += 1
            continue
        curr = np.array(row[1:]).astype(np.float)
        test_data.append(curr)
        test_label.append(row[0])



train_data = np.array(train_data).astype(np.float)
train_label = np.array(train_label).astype(np.float)
test_data = np.array(test_data).astype(np.float)
test_label = np.array(test_label).astype(np.float)

#20 * 20 train_data
middle_train_data = []
for i in range(len(train_data)):
    middle_train_data.append(get_middle(train_data[i]))
middle_train_data = np.array(middle_train_data).astype(np.float)

#20 * 20 test_data
middle_test_data = []
for i in range(len(test_data)):
    middle_test_data.append(get_middle(test_data[i]))
middle_test_data = np.array(middle_test_data).astype(np.float)


#stretched matrix
stretched_matrix = []
for i in range(len(middle_train_data)):
    line = middle_train_data[i]
    stretched_matrix.append(stretch(line))
stretched_matrix = np.array(stretched_matrix).astype(np.float)


################### competition #####################

#read test data
competition = []
with open('test.csv', 'r') as csvfile:
    data = csv.reader(csvfile, delimiter=',')
    for row in data:
        line = np.array(row).astype(np.float)
        competition.append(row)
competition = np.array(competition).astype(np.float)

#20 * 20 test_data
middle_competition_data = []
for i in range(len(competition)):
    middle_competition_data.append(get_middle(competition[i]))
middle_competition_data = np.array(middle_competition_data).astype(np.float)


clf = GaussianNB()
clf.fit(middle_train_data, train_label)



#The predicted result
result = clf.predict(middle_competition_data)
final = []
for i in result:
    final.append(int(i))



index_final = []
for i in range(0, 20000):
    index_final.append([i, final[i][0]])

with open('output.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    writer.writerow(['ImageId', 'Label'])
    writer.writerows(index_final)
