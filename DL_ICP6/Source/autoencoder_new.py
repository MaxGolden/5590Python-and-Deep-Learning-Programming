from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt
import pickle
from keras.callbacks import TensorBoard
from keras import regularizers
from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
# this is our input placeholder
input_img = Input(shape=(784,))

# activity_regularizer=regularizers.l1(0.01)

# "encoded" is the encoded representation of the input , activity_regularizer=regularizers.l1(1e-7)
encoded = Dense(8*encoding_dim, activation='relu', activity_regularizer=regularizers.l1(1e-7))(input_img)

# 4. add hidden layers
encoder_layer_h1 = Dense(4*encoding_dim, activation='relu')
encoder_layer_h2 = Dense(2*encoding_dim, activation='relu')
encoder_layer_h3 = Dense(encoding_dim, activation='relu')

# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoder_layer_h3(encoder_layer_h2(encoder_layer_h1(encoded))))

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
autoencoder.summary()

# this model maps an input to its encoded representation
encoder = Model(input_img, encoder_layer_h3(encoder_layer_h2(encoder_layer_h1(encoded))))
# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


# J:\5590Dl\DL\ICP6\max_v\autoEn_logs
# cd /path/to/log
# tensorboard --logdir=./
tensorboard = TensorBoard(log_dir='./autoEn_new_logs', histogram_freq=0,
                          write_graph=True, write_images=False)


hist = autoencoder.fit(x_train, x_train,
                       epochs=50,
                       batch_size=128,
                       shuffle=True,
                       validation_data=(x_test, x_test))

#loss, ac = autoencoder.evaluate(x_test, x_test, verbose=0)
#print("The loss is: ", loss)
#print("The accuracy is: ", ac)
# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

hist = hist.history

# # acc
# plt.plot(hist['acc'])
# plt.plot(hist['val_acc'])
# plt.title('acc')
# plt.ylabel('acc')
# plt.xlabel('epoch')
# plt.legend(['training', 'test'], loc='upper left')
# plt.show()

# # acc
# plt.plot(hist['loss'])
# plt.plot(hist['val_loss'])
# plt.title('loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['training', 'test'], loc='upper left')
# plt.show()

# displaying original and reconstructed image
n = 10  # how many digits we will display
plt.figure(figsize=(18, 4))
for i in range(n):
    # display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display encode
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(encoded_imgs[i].reshape(8, 4))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# model_ori = open('autoEn_max.pckl', 'wb')
# pickle.dump(hist, model_ori)
# model_ori.close()

