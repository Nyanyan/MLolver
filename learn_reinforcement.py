import keras
from keras.models import Sequential
from keras.layers.convolutional import MaxPooling3D
from keras.layers import Dense, Dropout, Flatten, Conv3D, Activation
from keras.callbacks import EarlyStopping
import numpy as np
from numpy import loadtxt

dataset = loadtxt('data.csv', delimiter=',')
input_shape = (36, 3, 3, 1)
X = dataset[:,0:324]
X = X.reshape(-1, input_shape[0], input_shape[1], input_shape[2], input_shape[3])
X = X.astype('float32')
y = dataset[:,324]
y = keras.utils.to_categorical(y, 21)

model = Sequential()
model.add(Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='same', input_shape=X.shape[1:]))
model.add(Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='same'))
model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
model.add(Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu', padding='same'))
model.add(Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu', padding='same'))
model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
model.add(Flatten())
model.add(Dropout(rate=0.2))
model.add(Dense(units=1024, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(units=1024, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(units=21, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
'''
model = Sequential()
model.add(Conv3D(filters=2, kernel_size=(3, 3, 3), activation='relu', padding='same', input_shape=X.shape[1:]))
model.add(Conv3D(filters=2, kernel_size=(3, 3, 3), activation='relu', padding='same'))
model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
model.add(Conv3D(filters=2, kernel_size=(3, 3, 3), activation='relu', padding='same'))
model.add(Conv3D(filters=2, kernel_size=(3, 3, 3), activation='relu', padding='same'))
model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
model.add(Flatten())
model.add(Dropout(rate=0.2))
model.add(Dense(units=2, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(units=2, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(units=21, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
'''

#early_stopping =  EarlyStopping(monitor='val_loss', min_delta=0.0, patience=2)


print(model.summary())

model.fit(X, y, epochs=10, batch_size=10)

dataset = loadtxt('data_test.csv', delimiter=',')
test_X = dataset[:,0:324]
test_X = test_X.reshape(-1, input_shape[0], input_shape[1], input_shape[2], input_shape[3])
test_y = dataset[:,324]
test_y = keras.utils.to_categorical(test_y, 21)
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