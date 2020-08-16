import pandas as pd
import numpy as np
from sklearn import svm, metrics, model_selection
#from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import pickle

load_mode = False

filename = 'model.sav'
if load_mode:
    clf = pickle.load(open(filename, 'rb'))
else:
    allData = pd.read_csv('data.csv')
    '''
    plt.scatter(allData['cp'], allData['num'])
    plt.show()
    '''
    data = allData[[str(i) for i in range(324)]]
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
    '''
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(train_data, train_label)
    '''
    clf = tree.DecisionTreeClassifier(max_depth=150)
    clf = clf.fit(train_data, train_label)

    pickle.dump(clf, open(filename, 'wb'))
    print('learning done')

print('getting test data')

test_data = []
with open('data_test.csv', 'r') as f:
    f.readline()
    for _ in range(100):
        test_data.append([int(i) for i in f.readline().replace('\n', '').split(',')])

print('testing')

ans = 0
ans2 = 0
for i in range(100):
    newdata = np.array([test_data[i][:324]])
    answer = test_data[i][324]
    prediction = clf.predict(newdata)[0]
    ans += abs(prediction - answer)
    if prediction == answer:
        ans2 += 1
ans /= 100
ans2 /= 100
print('average error', ans)
print('correct ratio', ans2)