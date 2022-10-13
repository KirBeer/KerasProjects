import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense

c = np.array([-40, -10, 0, 8, 15, 22, 38])  # Вектор входных значений
f = np.array([-40, 14, 32, 46, 59, 72, 100])  # Вектор выходных значений

model = keras.Sequential()  # Моделирование нейросети
model.add(Dense(units=1, input_shape=(1,), activation='linear'))
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.1))

history = model.fit(c, f, epochs=500, verbose=0)  # Обучение нейросети
print("Обучение завершено")

print(model.predict([100]))  # Тест для вычисления градусов фаренгейта при 100 градусах цельсия
print(model.get_weights())  # Отображение найденных весов

plt.plot(history.history['loss'])  # Построение графика изменения ошибки
plt.grid(True)
plt.show()