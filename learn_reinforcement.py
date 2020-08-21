import keras
from keras.models import Sequential
#from keras.layers.convolutional import MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Flatten, Conv2D, Activation, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping
import numpy as np
from numpy import loadtxt

dataset = loadtxt('data.csv', delimiter=',')
input_shape = (36, 3, 3)
X = dataset[:,0:324]
X = X.reshape(-1, input_shape[0], input_shape[1], input_shape[2])
X = X.astype('float32')
y = dataset[:,324]
y = keras.utils.to_categorical(y, 21)


model = Sequential()
model.add(Conv2D(filters=128, kernel_size=1, activation='relu', padding='same', input_shape=X.shape[1:]))
for _ in range(4):
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
#model.add(Conv2D(filters=64, kernel_size=5, activation='relu', padding='same'))
#model.add(Conv3D(filters=8, kernel_size=3, activation='relu', padding='same'))
#model.add(Flatten())
#model.add(Dropout(rate=0.2))
#model.add(Dense(units=2048, activation='relu'))
model.add(Dropout(rate=0.2))
#model.add(Dense(units=1024, activation='relu'))
#model.add(Dropout(rate=0.2))
#model.add(GlobalAveragePooling2D(data_format="channels_last"))
model.add(GlobalAveragePooling2D())
model.add(Dense(units=21, activation='sigmoid'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


print(model.summary())

model.fit(X, y, epochs=15, batch_size=10)


model = keras.models.load_model('model.h5', compile=False)


dataset = loadtxt('data_test.csv', delimiter=',')
test_X = dataset[:,0:324]
test_X = test_X.reshape(-1, input_shape[0], input_shape[1], input_shape[2])
test_y = dataset[:,324]
test_y = keras.utils.to_categorical(test_y, 21)

test_loss, test_acc = model.evaluate(X, y, verbose=0)
print(test_loss, test_acc)

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

# model.save('model.h5', include_optimizer=False)
