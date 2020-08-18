import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Activation, Flatten


# MNISTを学習用に正規化する関数
def get_mnist(shape):
    # MNISTデータを読込む
    (x_train, t_train), (x_test, t_test) = mnist.load_data()

    # MNISTデータを3次元に成形する
    x_train = x_train.reshape(60000, shape[0], shape[1], shape[2])
    x_test = x_test.reshape(10000, shape[0], shape[1], shape[2])

    # 画像ピクセル数値を0~255 → 0.0~1.0に正規化
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # one_hot_labelに変換
    t_train = keras.utils.to_categorical(t_train, 10)
    t_test = keras.utils.to_categorical(t_test, 10)

    # 60000件の訓練データの10000件を検証用データとする
    x_validate = x_train[50000:]
    x_train = x_train[:50000]
    t_validate = t_train[50000:]
    t_train = t_train[:50000]

    return (x_train, t_train), (x_validate, t_validate), (x_test, t_test)


# モデルを構築する関数
def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(3, 3)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# --- ここからがMain処理 ---

# MNIST画像のサイズ形式
input_shape = (28, 28, 1)
# エポック数
epochs = 5
# ミニバッチ
batch_size = 512

# MNIST取得
(x_train, t_train), (x_validate, t_validate), (x_test, t_test) = get_mnist(input_shape)

# モデル構築
model = create_model(input_shape)

# モデル要約出力
model.summary()

# 訓練
history = model.fit(x_train, t_train, batch_size=batch_size,
                    epochs=epochs, verbose=1, validation_data=(x_validate, t_validate))

# テスト
score = model.evaluate(x_test, t_test, verbose=1)
# テスト結果
print('loss:', score[0])
print('accuracy:', score[1])