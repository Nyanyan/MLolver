from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
import numpy as np
from numpy import loadtxt

#X, y, l = load_csv()
dataset = loadtxt('data.csv', delimiter=',')
'''
print(dataset[0])
X = [[[] for _ in range(6)] for _ in range(len(dataset))]
for i in range(len(dataset)):
    for color in range(6):
        for face in range(6):
            X[i][color].append(dataset[i,color * 54 + face * 9:color * 54 + face * 9 + 9])
X = np.array(X)
'''
X = dataset[:, 0:324].reshape(len(dataset), 3, 3, 36)
#print(X[0])
y = dataset[:,324:345]
#print(y[0])

model = Sequential()
model.add(Dense(100, input_shape=(3, 3, 36), activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(21, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(X, y, epochs=100, batch_size=10)

dataset = loadtxt('data_test.csv', delimiter=',')
test_X = dataset[:,0:324].reshape(len(dataset), 3, 3, 36)
test_y = dataset[:,324:345]
prediction = model.predict_classes(test_X)

correct_ratio = 0
error_average = 0
ans = [0 for _ in range(21)]
for i in range(len(dataset)):
    ans[prediction[i]] += 1
    predicted = -1
    for j in range(21):
        if test_y[i][j] == 1:
            predicted = j
            break
    if prediction[i] == predicted:
        correct_ratio += 1
    error_average += abs(prediction[i] - predicted)
correct_ratio /= len(dataset)
error_average /= len(dataset)
print(correct_ratio)
print(error_average)
print(ans)

model.save('model.h5', include_optimizer=False)