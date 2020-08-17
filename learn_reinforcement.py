from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from numpy import loadtxt

#X, y, l = load_csv()
dataset = loadtxt('data.csv', delimiter=',')
X = dataset[:,0:324]
y = dataset[:,324:345]

model = Sequential()
model.add(Dense(100, input_dim=324, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(21, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=100, batch_size=10)

dataset = loadtxt('data_test.csv', delimiter=',')
test_X = dataset[:,0:324]
test_y = dataset[:,324:345]
prediction = model.predict_classes(test_X)

correct_ratio = 0
error_average = 0
ans = [0 for _ in range(25)]
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