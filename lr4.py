import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10


# Загрузка и предобработка данных CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


# Определение модели
input_layer = keras.Input(shape=(32, 32, 3), name="image_input")  # Входной слой для изображений размером 32x32 с 3 цветами

conv_layer_1 = layers.Conv2D(32, 3, activation="relu", padding="same")(input_layer)  # Первая свертка
conv_layer_2 = layers.Conv2D(32, 3, activation="relu", padding="same")(conv_layer_1)  # Вторая свертка
max_pooling_block_1 = layers.MaxPooling2D(2)(conv_layer_2)  # Макс-пулинг для уменьшения размерности

conv_layer_3 = layers.Conv2D(64, 3, activation="relu", padding="same")(max_pooling_block_1)  # Третья свертка
conv_layer_4 = layers.Conv2D(64, 3, activation="relu", padding="same")(conv_layer_3)  # Четвертая свертка
residual_block_1_resized = layers.Conv2D(64, 1, padding="same")(max_pooling_block_1)  # Приведение max_pooling_block_1 к размеру (16, 16, 64) для добавления к выходу четвертого свертки
residual_block_2_output = layers.add([conv_layer_4, residual_block_1_resized])  # Сложение выходов для остаточного блока

conv_layer_5 = layers.Conv2D(128, 3, activation="relu", padding="same")(residual_block_2_output)  # Пятая свертка
conv_layer_6 = layers.Conv2D(128, 3, activation="relu", padding="same")(conv_layer_5)  # Шестая свертка
residual_block_2_resized = layers.Conv2D(128, 1, padding="same")(residual_block_2_output)  # Приведение residual_block_2_output к размеру (8, 8, 128) для добавления к выходу шестого свертки
residual_block_3_output = layers.add([conv_layer_6, residual_block_2_resized])  # Сложение выходов для остаточного блока

global_average_pooling_output = layers.GlobalAveragePooling2D()(residual_block_3_output)
dense_layer = layers.Dense(128, activation="relu")(global_average_pooling_output)  # Полносвязный слой с уменьшением количества нейронов
dropout_layer = layers.Dropout(0.5)(dense_layer)  # Слой Dropout для предотвращения переобучения
output_layer = layers.Dense(10, activation='softmax')(dropout_layer)  # Выходной слой с softmax


# Создание модели
model = keras.Model(input_layer, output_layer)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=32,
          epochs=30,
          validation_split=0.2)

# Оценка модели на тестовом наборе данных
score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
