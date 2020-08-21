import keras
from keras.models import Sequential
#from keras.layers.convolutional import MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Flatten, Conv2D, Activation, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping
import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt

model_num = 1

dataset = loadtxt('data.csv', delimiter=',')
input_shape = (36, 3, 3)
X = dataset[:,0:324]
X = X.reshape(-1, input_shape[0], input_shape[1], input_shape[2])
X = X.astype('float32')
y = dataset[:,324]
y = keras.utils.to_categorical(y, 21)

models = []
history = []

for _ in range(model_num):
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=1, activation='relu', padding='same', input_shape=X.shape[1:]))
    for _ in range(1):
        model.add(BatchNormalization())
        model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
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

    history.append(model.fit(X, y, epochs=10, batch_size=32))

    models.append(model)


#model = keras.models.load_model('model.h5', compile=False)


dataset = loadtxt('data_test.csv', delimiter=',')
test_X = dataset[:,0:324]
test_X = test_X.reshape(-1, input_shape[0], input_shape[1], input_shape[2])
test_y = dataset[:,324]
test_y = keras.utils.to_categorical(test_y, 21)

for i in range(model_num):
    test_loss, test_acc = models[i].evaluate(X, y, verbose=0)
    print('model', i, test_loss, test_acc)

prediction = []
for i in range(model_num):
    prediction.append(models[i].predict_classes(test_X))
    correct_ratio = 0
    error_average = 0
    ans = [0 for _ in range(21)]
    predicted_ans = [0 for _ in range(21)]
    for j in range(len(dataset)):
        ans[prediction[i][j]] += 1
        predicted = -1
        for k in range(21):
            if test_y[j][k] == 1:
                predicted = k
                break
        predicted_ans[predicted] += 1
        if prediction[i][j] == predicted:
            correct_ratio += 1
        error_average += abs(prediction[i][j] - predicted)
    correct_ratio /= len(dataset)
    error_average /= len(dataset)
    print('model', i)
    print('correct ratio', correct_ratio)
    print('average error', error_average)
    print(ans)
    print(predicted_ans)

prediction_merged = [int(round(sum(prediction[i][j] for i in range(model_num)) / model_num)) for j in range(len(dataset))] # soft
correct_ratio = 0
error_average = 0
ans = [0 for _ in range(21)]
predicted_ans = [0 for _ in range(21)]
for j in range(len(dataset)):
    ans[prediction_merged[j]] += 1
    predicted = -1
    for k in range(21):
        if test_y[j][k] == 1:
            predicted = k
            break
    predicted_ans[predicted] += 1
    if prediction_merged[j] == predicted:
        correct_ratio += 1
    error_average += abs(prediction_merged[j] - predicted)
correct_ratio /= len(dataset)
error_average /= len(dataset)
print('merged')
print('correct ratio', correct_ratio)
print('average error', error_average)
print(ans)
print(predicted_ans)

acc = []
loss = []
for i in range(model_num):
    models[i].save('model' + str(i) + '.h5', include_optimizer=False)

    acc.append(history[i].history['accuracy'])
    #val_acc = history.history['val_acc']
    loss.append(history[i].history['loss'])
    #val_loss = history.history['val_loss']
epochs = range(len(acc[0]))

# 1) Accracy Plt
for i in range(model_num):
    plt.plot(epochs, acc[i] ,label = 'training acc' + str(i))
#plt.plot(epochs, val_acc, 'b' , label= 'validation acc')
plt.title('Training and Validation acc')
plt.legend()

plt.figure()

# 2) Loss Plt
for i in range(model_num):
    plt.plot(epochs, loss[i] ,label = 'training loss' + str(i))
#plt.plot(epochs, val_loss, 'b' , label= 'validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()
