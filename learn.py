import pandas as pd
import numpy as np
from sklearn import svm, metrics, model_selection
from sklearn.neighbors import KNeighborsClassifier

allData = pd.read_csv('data.csv')
'''
plt.scatter(allData['cp'], allData['num'])
plt.show()
'''
data = allData[['cp', 'co', 'ep', 'eo']]
label = allData['num']

train_data, test_data, train_label, test_label = model_selection.train_test_split(data, label)

clf = svm.SVC()
clf.fit(train_data, train_label)
pre = clf.predict(test_data)

ac_score = metrics.accuracy_score(test_label, pre)
print("正解率 =", ac_score)