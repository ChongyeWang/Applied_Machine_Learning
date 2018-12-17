from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt


# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)


# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)


# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))


autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))


############ select 20 images ##############
dict = {}
for i in range(0, 10):
    dict[i] = []
for i in range(len(x_test)):
    key = y_test[i]
    if len(dict[key]) < 2:
        dict[key].append(i)
    sum = 0
    for j in dict:
        sum += len(dict[j])
    if sum == 20:
        break


selected_images = {}
for i in range(0, 10):
    selected_images[i] = []
for i in range(10):
    selected_images[i].append(x_test[dict[i][0]])
    selected_images[i].append(x_test[dict[i][1]])


start = []
for i in range(10):
    start.append(selected_images[i][0])
    start.append(selected_images[i][1])
start = np.array(start)


# encode and decode some digits
# note that we take them from the *test* set
encoded_images = encoder.predict(start)


result = []
for i in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]:
    l1 = np.array(encoded_images[i] + (1 / 8) * (encoded_images[i + 1] - encoded_images[i]))
    l2 = np.array(encoded_images[i] + (2 / 8) * (encoded_images[i + 1] - encoded_images[i]))
    l3 = np.array(encoded_images[i] + (3 / 8) * (encoded_images[i + 1] - encoded_images[i]))
    l4 = np.array(encoded_images[i] + (4 / 8) * (encoded_images[i + 1] - encoded_images[i]))
    l5 = np.array(encoded_images[i] + (5 / 8) * (encoded_images[i + 1] - encoded_images[i]))
    l6 = np.array(encoded_images[i] + (6 / 8) * (encoded_images[i + 1] - encoded_images[i]))
    l7 = np.array(encoded_images[i] + (7 / 8) * (encoded_images[i + 1] - encoded_images[i]))

    result += [l1, l2, l3, l4, l5, l6, l7]

result = np.array(result)


decoded_imgs = decoder.predict(result)


decoded_images = []

j = 0
for i in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]:

    decoded_images.append(start[i])
    for k in range(j, j + 7):
        decoded_images.append(decoded_imgs[k])
    j += 7
    decoded_images.append(start[i + 1])


fig = plt.figure(figsize=(10, 10))
for i in range(1, 91):
    ax = fig.add_subplot(10, 9, i)
    ax.imshow(np.reshape(decoded_images[i - 1], (28, 28)))
plt.show()



############### different digit ###############

dict = {}
for i in range(10):
    dict[i] = []

j = 0
for i in range(10):
    while j < len(x_test):
        if len(dict[i]) == 0:
            dict[i].append(j)
            j += 1
        elif len(dict[i]) == 1:
            if y_test[j] == y_test[dict[i][0]]:
                j += 1
                continue
            else:
                dict[i].append(j)
                j += 1
                break

selected_images = {}
for i in range(0, 10):
    selected_images[i] = []
for i in range(10):
    selected_images[i].append(x_test[dict[i][0]])
    selected_images[i].append(x_test[dict[i][1]])
    

start = []
for i in range(10):
    start.append(selected_images[i][0])
    start.append(selected_images[i][1])
start = np.array(start)


# encode and decode some digits
# note that we take them from the *test* set
encoded_images = encoder.predict(start)


result = []
for i in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]:
    l1 = np.array(encoded_images[i] + (1 / 8) * (encoded_images[i + 1] - encoded_images[i]))
    l2 = np.array(encoded_images[i] + (2 / 8) * (encoded_images[i + 1] - encoded_images[i]))
    l3 = np.array(encoded_images[i] + (3 / 8) * (encoded_images[i + 1] - encoded_images[i]))
    l4 = np.array(encoded_images[i] + (4 / 8) * (encoded_images[i + 1] - encoded_images[i]))
    l5 = np.array(encoded_images[i] + (5 / 8) * (encoded_images[i + 1] - encoded_images[i]))
    l6 = np.array(encoded_images[i] + (6 / 8) * (encoded_images[i + 1] - encoded_images[i]))
    l7 = np.array(encoded_images[i] + (7 / 8) * (encoded_images[i + 1] - encoded_images[i]))

    result += [l1, l2, l3, l4, l5, l6, l7]

result = np.array(result)


decoded_imgs = decoder.predict(result)


decoded_images = []

j = 0
for i in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]:

    decoded_images.append(start[i])
    for k in range(j, j + 7):
        decoded_images.append(decoded_imgs[k])
    j += 7
    decoded_images.append(start[i + 1])


fig = plt.figure(figsize=(10, 10))
for i in range(1, 91):
    ax = fig.add_subplot(10, 9, i)
    ax.imshow(np.reshape(decoded_images[i - 1], (28, 28)))
plt.show()


