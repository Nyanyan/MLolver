'''
Corner
   B
  0 1 
L 2 3 R
   F

   F
  4 5
L 6 7 R
   B


Edge
top layer
    B
    0
L 3   1 R
    2
    F

middle layer
4 F 5 R 6 B 7

bottom layer
    F
    8
L 11  9 R
    10
    B
'''


fac = [1 for _ in range(15)]
for i in range(1, 15):
    fac[i] = fac[i - 1] * i

def cmb(n, r):
    return fac[n] // fac[r] // fac[n - r]

class Cube:
    def __init__(self, cp=list(range(8)), co=[0 for _ in range(8)], ep=list(range(12)), eo=[0 for i in range(12)]):
        self.Cp = cp
        self.Co = co
        self.Ep = ep
        self.Eo = eo
    
    def move_cp(self, mov):
        surface = [[3, 1, 7, 5], [0, 2, 4, 6], [0, 1, 3, 2], [4, 5, 7, 6], [2, 3, 5, 4], [1, 0, 6, 7]]
        res = [i for i in self.Cp]
        mov_type = mov // 3
        mov_amount = mov % 3
        for i in range(4):
            res[surface[mov_type][(i + mov_amount + 1) % 4]] = self.Cp[surface[mov_type][i]]
        return res
    
    def move_co(self, mov):
        surface = [[3, 1, 7, 5], [0, 2, 4, 6], [0, 1, 3, 2], [4, 5, 7, 6], [2, 3, 5, 4], [1, 0, 6, 7]]
        pls = [2, 1, 2, 1]
        res = [i for i in self.Co]
        mov_type = face(mov)
        mov_amount = mov % 3
        for i in range(4):
            res[surface[mov_type][(i + mov_amount + 1) % 4]] = self.Co[surface[mov_type][i]]
            if axis(mov) != 1 and mov_amount != 1:
                res[surface[mov_type][(i + mov_amount + 1) % 4]] += pls[(i + mov_amount + 1) % 4]
                res[surface[mov_type][(i + mov_amount + 1) % 4]] %= 3
        return res
    
    def move_ep(self, mov):
        surface = [[1, 6, 9, 5], [3, 4, 11, 7], [0, 1, 2, 3], [8, 9, 10, 11], [2, 5, 8, 4], [0, 7, 10, 6]]
        res = [i for i in self.Ep]
        mov_type = face(mov)
        mov_amount = mov % 3
        for i in range(4):
            res[surface[mov_type][(i + mov_amount + 1) % 4]] = self.Ep[surface[mov_type][i]]
        return res
    
    def move_eo(self, mov):
        surface = [[1, 6, 9, 5], [3, 4, 11, 7], [0, 1, 2, 3], [8, 9, 10, 11], [2, 5, 8, 4], [0, 7, 10, 6]]
        res = [i for i in self.Eo]
        mov_type = face(mov)
        mov_amount = mov % 3
        for i in range(4):
            res[surface[mov_type][(i + mov_amount + 1) % 4]] = self.Eo[surface[mov_type][i]]
        if axis(mov) == 2 and mov_amount != 1:
            for i in surface[mov_type]:
                res[i] += 1
                res[i] %= 2
        return res
    
    def move(self, mov):
        return Cube(cp=self.move_cp(mov), co=self.move_co(mov), ep=self.move_ep(mov), eo=self.move_eo(mov))
    
    def idx(self):
        res = [0 for _ in range(324)]
        for i in range(6):
            res[i * 9 + 4 + i * 54] = 1
        corner_colors = [[0, 4, 3], [0, 3, 2], [0, 1, 4], [0, 2, 1], [5, 4, 1], [5, 1, 2], [5, 3, 4], [5, 2, 3]]
        edge_colors = [[0, 3], [0, 2], [0, 1], [0, 4], [1, 4], [1, 2], [3, 2], [3, 4], [5, 1], [5, 2], [5, 3], [5, 4]]
        corner_stickers = [[0, 36, 29], [2, 27, 20], [6, 9, 38], [8, 18, 11], [45, 44, 15], [47, 17, 24], [51, 35, 42], [53, 26, 33]]
        edge_stickers = [[1, 28], [5, 19], [7, 10], [3, 37], [12, 41], [14, 21], [30, 23], [32, 39], [46, 16], [50, 25], [52, 34], [48, 43]]
        for corner_idx in range(8):
            corner = self.Cp[corner_idx]
            co = self.Co[corner_idx]
            for i, j in enumerate(corner_stickers[corner_idx]):
                color = corner_colors[corner][(i - co) % 3]
                res[j + color * 54] = 1
        for edge_idx in range(12):
            edge = self.Ep[edge_idx]
            eo = self.Eo[edge_idx]
            for i, j in enumerate(edge_stickers[edge_idx]):
                color = edge_colors[edge][(i - eo) % 2]
                res[j + color * 54] = 1
        return res

def face(twist):
    return twist // 3

def axis(twist):
    return twist // 6


import keras
from keras.models import Sequential
#from keras.layers.convolutional import MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Flatten, Conv2D, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping
import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt
from random import randint

def generate_p(depth, l_twist, cube):
    if depth == 0:
        return cube
    twist = randint(0, 17)
    while twist // 3 == l_twist // 3:
        twist = randint(0, 17)
    cube = cube.move(twist)
    return generate_p(depth - 1, twist, cube)

def generate_data(num):
    res_x = []
    res_y = []
    for _ in range(num):
        depth = randint(1, 20)
        res_x.append(generate_p(depth, -10, Cube()).idx())
        res_y.append(depth)
    res_x = np.array(res_x).reshape(num, input_shape[0], input_shape[1], input_shape[2])
    res_x = res_x.astype('float32')
    res_y = keras.utils.to_categorical(np.array(res_y), 21)
    return res_x, res_y

model_num = 1
input_shape = (36, 3, 3)
'''
dataset = loadtxt('data.csv', delimiter=',')
X = dataset[:,0:324]
X = X.reshape(-1, input_shape[0], input_shape[1], input_shape[2])
X = X.astype('float32')
y = dataset[:,324]
y = keras.utils.to_categorical(y, 21)
'''
models = []
history = []

for _ in range(model_num):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=1, activation='relu', padding='same', input_shape=input_shape))
    for _ in range(10):
        model.add(BatchNormalization())
        model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(Dropout(rate=0.2))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=21, activation='sigmoid'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    print(model.summary())

    x, y = generate_data(10000)
    history.append(model.fit(x, y, epochs=75, batch_size=256))

    models.append(model)


#model = keras.models.load_model('model.h5', compile=False)

'''
dataset = loadtxt('data_test.csv', delimiter=',')
test_X = dataset[:,0:324]
test_X = test_X.reshape(-1, input_shape[0], input_shape[1], input_shape[2])
test_y = dataset[:,324]
test_y = keras.utils.to_categorical(test_y, 21)
'''
l = 2000
test_X, test_y = generate_data(l)
prediction = []
for i in range(model_num):
    prediction.append(models[i].predict_classes(test_X))
    correct_ratio = 0
    error_average = 0
    ans = [0 for _ in range(21)]
    predicted_ans = [0 for _ in range(21)]
    for j in range(l):
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
    correct_ratio /= l
    error_average /= l
    print('model', i)
    print('correct ratio', correct_ratio)
    print('average error', error_average)
    print(ans)
    print(predicted_ans)

prediction_merged = [int(round(sum(prediction[i][j] for i in range(model_num)) / model_num)) for j in range(l)] # soft
correct_ratio = 0
error_average = 0
ans = [0 for _ in range(21)]
predicted_ans = [0 for _ in range(21)]
for j in range(l):
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
correct_ratio /= l
error_average /= l
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
    plt.plot(epochs, acc[i], label = 'training acc' + str(i))
#plt.plot(epochs, val_acc, 'b' , label= 'validation acc')
plt.title('Training and Validation acc')
plt.legend()

plt.figure()

# 2) Loss Plt
for i in range(model_num):
    plt.plot(epochs, loss[i], label = 'training loss' + str(i))
#plt.plot(epochs, val_loss, 'b' , label= 'validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()
