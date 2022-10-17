import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Стандартизация

x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

# Создание структуры нейросети

model = keras.Sequential([
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2), strides=2),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')])

print(model.summary())  # Вывод структуры нейросети в консоль

#  Компиляция нейросети. Оптимизатор - Адам, критерий качества - категориальная кроссэнтропия

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#  Запуск процесса обучения. 80% - обучающая выборка, 20% - выборка валидации

his = model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)
model.evaluate(x_test, y_test_cat)

#  Распознавание всей тестовой выборки

pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)

#  Выделение неверных вариантов распознавания

mask = pred == y_test

x_false = x_test[~mask]
y_false = pred[~mask]

print(x_false.shape)

'''
#  Вывод первых 5 неверных изображений

for i in range(5):
    print('Значение сети: ' + str(y_false[i]))
    plt.imshow(x_false[i], cmap=plt.cm.binary)
    plt.show()
'''
