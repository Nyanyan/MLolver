import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPooling3D, Convolution3D
import numpy as np
from numpy import loadtxt

dataset = loadtxt('data.csv', delimiter=',')
input_shape = (36, 3, 3, 1)
X = dataset[:,0:324]
X = X.reshape(-1, input_shape[0], input_shape[1], input_shape[2], input_shape[3])
print(X.shape)
y = dataset[:,324]
y = keras.utils.to_categorical(y, 21)
print(y.shape)

model = Sequential()
model.add(Conv3D(256, kernel_size=(36, 3, 3), input_shape=input_shape, activation='relu'))
model.add(Conv3D(128, (36, 3, 3), activation='relu'))
model.add(MaxPooling3D(pool_size=(3, 3, 3)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(21, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(X, y, epochs=20, batch_size=10)

dataset = loadtxt('data_test.csv', delimiter=',')
test_X = dataset[:,0:324] #.reshape(len(dataset), 3, 3, 36)
test_y = dataset[:,324:345]
prediction = model.predict_classes(test_X)

correct_ratio = 0
error_average = 0
ans = [0 for _ in range(21)]
predicted_ans = [0 for _ in range(21)]
for i in range(len(dataset)):
    ans[prediction[i]] += 1
    predicted = -1
    for j in range(21):
        if test_y[i][j] == 1:
            predicted = j
            break
    predicted_ans[predicted] += 1
    if prediction[i] == predicted:
        correct_ratio += 1
    error_average += abs(prediction[i] - predicted)
correct_ratio /= len(dataset)
error_average /= len(dataset)
print('correct ratio', correct_ratio)
print('average error', error_average)
print(ans)
print(predicted_ans)

model.save('model.h5', include_optimizer=False)