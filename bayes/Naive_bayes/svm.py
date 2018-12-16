import matplotlib.pyplot as plt
from sklearn.svm import SVC
import csv

data = []
target = []

data2 = []
target2 = []

size = 0

with open('pima-indians-diabetes.csv', 'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    size = 614

    count = 0

    for row in plots:
        if count < size:
            data.append(row[0:7])
            target.append(row[8])
        else:
            data2.append(row[0:7])
            target2.append(row[8])
        count += 1

model = SVC()
model.fit(data, target)
result = model.score(data2, target2)
print(result)#0.64
