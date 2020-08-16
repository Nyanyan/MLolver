import pandas as pd
import numpy as np
from sklearn import svm, metrics, model_selection
from sklearn.neighbors import KNeighborsClassifier
import pickle

load_mode = True

filename = 'model.sav'
if load_mode:
    knn = pickle.load(open(filename, 'rb'))
else:
    allData = pd.read_csv('data.csv')
    '''
    plt.scatter(allData['cp'], allData['num'])
    plt.show()
    '''
    data = allData[[str(i) for i in range(54)]]
    label = allData['num']

    #train_data, test_data, train_label, test_label = model_selection.train_test_split(data, label)
    train_data = data
    train_label = label
    '''
    clf = svm.SVC()
    clf.fit(train_data, train_label)
    pre = clf.predict(test_data)

    print('learning done')

    ac_score = metrics.accuracy_score(test_label, pre)
    print("正解率 =", ac_score)
    '''
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(train_data, train_label)
    pickle.dump(knn, open(filename, 'wb'))
    print('learning done')

print('getting test data')

test_data = []
with open('data_test.csv', 'r') as f:
    f.readline()
    for _ in range(100):
        test_data.append([int(i) for i in f.readline().replace('\n', '').split(',')])

print('testing')

ans = 0
for i in range(100):
    newdata = np.array([test_data[i][:54]])
    answer = test_data[i][54]
    prediction = knn.predict(newdata)[0]
    if abs(prediction - answer):
        ans += 1
ans /= 100
print('average error', ans)