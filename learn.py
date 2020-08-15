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
    data = allData[[str(i) for i in range(40)]]
    label = allData['num']

    train_data, test_data, train_label, test_label = model_selection.train_test_split(data, label)

    clf = svm.SVC()
    clf.fit(train_data, train_label)
    pre = clf.predict(test_data)

    print('learning done')

    ac_score = metrics.accuracy_score(test_label, pre)
    print("正解率 =", ac_score)
    
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(train_data, train_label)
    pickle.dump(knn, open(filename, 'wb'))

newdata = np.array([[6, 2, 3, 5, 4, 7, 0, 1, 2, 1, 2, 0, 2, 1, 1, 0, 0, 8, 2, 9, 10, 6, 1, 3, 4, 5, 11, 7, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0]])
print("newdata.shape: {}".format(newdata.shape))

prediction = knn.predict(newdata)
print("Predicted target name: {}".format(prediction))