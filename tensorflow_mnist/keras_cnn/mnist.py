import keras
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
import tensorflow as tf


model = models.Sequential()

model.add(layers.Conv2D(32, (5, 5), padding="same", activation='relu', input_shape=(28, 28, 1))) 

model.add(layers.MaxPooling2D((2, 2), strides=2))

model.add(layers.Conv2D(64, (5, 5), padding="same", activation='relu')) 

model.add(layers.MaxPooling2D((2, 2), strides=2))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dropout(0.4))

model.add(layers.Dense(10, activation='softmax'))

model.summary()

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    keras.callbacks.TensorBoard(
        log_dir='my_log_dir'
	) 
]

history = model.fit(train_images, train_labels,
					epochs=20,
					batch_size=100,
					validation_split=0.2,
					callbacks=callbacks)


