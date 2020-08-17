from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from numpy import loadtxt

def load_csv():
    file = []
    with open('data.csv', 'r') as f:
        f.readline()
        while True:
            try:
                file.append([int(i) for i in f.readline().replace('\n', '').split(',')])
            except:
                break
    data_all = [file[i][:324] for i in range(len(file))]
    target = [file[i][324] for i in range(len(file))]
    data = np.array([[[[arr[i + j * 9 + k * 54] for i in range(9)] for j in range(6)] for k in range(6)] for arr in data_all])
    return data, target, len(file)

#X, y, l = load_csv()
dataset = loadtxt('data.csv', delimiter=',')
X = dataset[0:100,0:324]
y = dataset[0:100,324]

model = Sequential()
model.add(Dense(12, input_dim=324, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=100, batch_size=10)

dataset = loadtxt('data_test.csv', delimiter=',')
test_X = dataset[:,0:324]
test_y = dataset[:,324]
prediction = model.predict_classes(test_X)

correct_ratio = 0
error_average = 0
ans = [0 for _ in range(25)]
for i in range(len(dataset)):
    ans[prediction[i][0]] += 1
    if prediction[i][0] == test_y[i]:
        correct_ratio += 1
    error_average += abs(prediction[i][0] - test_y[i])
correct_ratio /= len(dataset)
error_average /= len(dataset)
print(correct_ratio)
print(error_average)
print(ans)