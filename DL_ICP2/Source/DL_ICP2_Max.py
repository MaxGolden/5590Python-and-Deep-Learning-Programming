from keras import Sequential
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.utils import to_categorical

(train_images,train_labels), (test_images, test_labels) = mnist.load_data()
#display the first image in the training data
# plt.imshow(train_images[0, :, :], cmap='gray')
# plt.title('Ground Truth : {}'.format(train_labels[0]))
# plt.show()

#process the data
#1. convert each image of shape 28*28 to 784 dimensional which will be fed to the network as a single feature
dimData = np.prod(train_images.shape[1:])
train_data = train_images.reshape(train_images.shape[0], dimData)
test_data = test_images.reshape(test_images.shape[0], dimData)

#convert data to float and scale values between 0 and 1
train_data = train_data.astype('float')
test_data = test_data.astype('float')
#scale data
train_data /=255.0
test_data /=255.0
#change the labels frominteger to one-hot encoding
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

#creating network
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(dimData,)))
model.add(Dense(512, activation='relu'))
# One more dense
# model.add(Dense(128, activation='sigmoid'))

model.add(Dense(10, activation='softmax'))

model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

history = model.fit(train_data, train_labels_one_hot,
                    batch_size=256,
                    epochs=10,  # 20
                    verbose=1,
                    validation_data=(test_data, test_labels_one_hot))

predict1 = model.predict_classes(test_data[[0], :])
# print(predict1)

[test_loss, test_acc] = model.evaluate(test_data, test_labels_one_hot)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))
print("======================")
print(history.history.keys())


# DL_ICP2 - question1
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# acc
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# DL_ICP2 - question2
predict_test = model.predict_classes(test_data[[30], :])
print("The prediction of the 30th in the test dataset is: ", predict_test)

plt.imshow(test_images[30, :, :], cmap='gray')
plt.title('Ground Truth : {}'.format(test_labels[30]))
plt.show()

# DL_ICP2 - question3
# With


