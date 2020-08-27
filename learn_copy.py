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
        '''
        res_2 = [-1 for _ in range(324)]
        for face in range(6):
            for color in range(6):
                for y in range(3):
                    for x in range(3):
                        res_2[y * 108 + x * 36 + face * 6 + color] = res[face * 54 + color * 9 + y * 3 + x]
        return res_2

        res_edge = []
        for i in range(12):
            tmp = [0 for _ in range(12)]
            tmp[self.Ep[i]] = 1
            res_edge.extend(tmp)
        eo = []
        for i in range(12):
            eo.append(self.Eo[i])
        res_edge.extend(eo)
        return res_edge
        '''

def face(twist):
    return twist // 3

def axis(twist):
    return twist // 6

import tensorflow as tf
import keras
'''
from keras.models import Sequential, Model
#from keras.layers.convolutional import MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.applications.resnet50 import ResNet50
from keras import optimizers
'''
from keras.callbacks import ModelCheckpoint
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


def generate_data_test(num):
    res_x = []
    res_y = []
    for _ in range(num):
        depth = randint(1, 20)
        res_x.append(generate_p(depth, -10, Cube()).idx())
        res_y.append(depth)
    res_x = np.array(res_x).reshape(num, 36, 3, 3)
    res_x = res_x.astype('float32')
    return res_x, res_y

def generate_data_m(depth):
    res_x = np.array(generate_p(depth, -10, Cube()).idx()).reshape(36, 3, 3)
    return res_x


#!pip install funcy
#!pip install pathlib
from funcy   import *
from pathlib import *


def computational_graph():
    def add():
        return tf.keras.layers.Add()

    def batch_normalization():
        return tf.keras.layers.BatchNormalization()

    def conv(filter_size, kernel_size=3):
        return tf.keras.layers.Conv2D(filter_size, kernel_size, padding='same', use_bias=False, kernel_initializer='he_normal')

    def dense(unit_size):
        return tf.keras.layers.Dense(unit_size, use_bias=False, kernel_initializer='he_normal')

    def global_average_pooling():
        return tf.keras.layers.GlobalAveragePooling2D()

    def relu():
        return tf.keras.layers.ReLU()

    ####

    def residual_block(width):
        return rcompose(ljuxt(rcompose(batch_normalization(),
                                       conv(width),
                                       batch_normalization(),
                                       relu(),
                                       conv(width),
                                       batch_normalization()),
                              identity),
                        add())

    #W = 1024
    W = 32
    H = 2

    return rcompose(conv(W, 1),
                    rcompose(*repeatedly(partial(residual_block, W), H)),
                    global_average_pooling(),
                    dense(1),
                    relu())






def create_model():
    result = tf.keras.Model(*juxt(identity, computational_graph())(tf.keras.Input(shape=(36, 3, 3))))

    result.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    result.summary()

    return result

def create_generator(batch_size):
    while True:
        xs = []
        ys = []

        for i in range(batch_size):
            step = randint(1, 20)

            xs.append(generate_data_m(step))
            ys.append(step)

        yield np.array(xs), np.array(ys)

print(Cube().idx())

model_path = Path('./cost.h5')

if not model_path.exists():
    model = create_model()
    # checkpointの設定
    checkpoint = ModelCheckpoint(
                        filepath="./model_during/model-{epoch:02d}-{mean_absolute_error:.2f}.h5",
                        monitor='mean_absolute_error',
                        save_best_only=True,
                        period=1,
                    )
    history = model.fit_generator(create_generator(100), steps_per_epoch=10, epochs=100, callbacks=[checkpoint]) # 1000, 10, 5000
    tf.keras.models.save_model(model, 'cost.h5')
    tf.keras.backend.clear_session()
    acc = history.history['mean_absolute_error']
    loss = history.history['loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, label = 'training error')
    plt.title('Training error')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, label = 'training loss')
    plt.title('Training loss')
    plt.legend()

    plt.show()
else:
    model = tf.keras.models.load_model('cost.h5')
#model_path.parent.mkdir(exist_ok=True)



l = 1000
plt_x = []
plt_y = []
test_X, test_y = generate_data_test(l)
error_average = 0
correct_ratio = [0 for _ in range(25)]
pre_tmp = 0
for j in range(l):
    tmp = int(j / l * 10)
    if pre_tmp != tmp:
        pre_tmp = tmp
        for _ in range(tmp):
            print('=', end='')
        for _ in range(10 - tmp):
            print('.', end='')
        print('')
    prediction = float(model.predict(np.array([test_X[j]]),batch_size=10)[0][0])
    prediction_int = int(round(prediction))
    predicted = test_y[j]
    plt_x.append(predicted)
    plt_y.append(prediction)
    correct_ratio[abs(predicted - prediction_int)] += 1
    #print(prediction, predicted)
    error_average += abs(predicted - prediction)
error_average /= l
#correct_ratio /= l
print('correct ratio', correct_ratio)
print('average error', error_average)
#print('average error moves', error_average * 20)

plt.scatter(plt_x, plt_y)
plt.plot(range(20), range(20), label = 'Theoretical Value')
plt.title('Training error')
plt.figure()
plt.show()