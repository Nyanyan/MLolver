import pandas as pd
import numpy as np
from sklearn import svm, metrics, model_selection
from sklearn.neighbors import KNeighborsClassifier
import pickle

load_mode = False

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
    '''
    ac_score = metrics.accuracy_score(test_label, knn)
    print("正解率 =", ac_score)
    '''
newdata = np.array([[0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 5, 1, 1, 5, 1, 1, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 3, 3, 0, 3, 3, 0, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 3, 5, 5, 3, 5, 5, 3]])
print("newdata.shape: {}".format(newdata.shape))

prediction = knn.predict(newdata)
print("Predicted target name: {}".format(prediction))