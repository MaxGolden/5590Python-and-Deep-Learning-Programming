# Simple CNN model for CIFAR-10
import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model
import pickle
from keras.callbacks import TensorBoard
from keras.models import model_from_yaml
import matplotlib.pyplot as plt

K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
# print(num_classes)

# load YAML and create model
yaml_file = open('model2.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("model2.h5")
print("Loaded model from disk")

yaml_file1 = open('model1.yaml', 'r')
loaded_model_yaml1 = yaml_file1.read()
yaml_file1.close()
loaded_model1 = model_from_yaml(loaded_model_yaml1)
# load weights into new model
loaded_model1.load_weights("model1.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
epochs = 20
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
score = loaded_model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

loaded_model1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
score = loaded_model1.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model1.metrics_names[1], score[1]*100))

# evaluation of the model
# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))

# ResultsList = loaded_model.predict_on_batch(X_test[0:4])
# print(ResultsList)
# print(y_test[0:4])

# retrieve:
f = open('model2.pckl', 'rb')
history = pickle.load(f)
f.close()
f1 = open('model1.pckl', 'rb')
history1 = pickle.load(f1)
f1.close()
# DL_ICP2 - question1
plt.plot(history1['val_loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['model_ori', 'model_new'], loc='upper left')
plt.show()

# acc
plt.plot(history1['val_acc'])
plt.plot(history['val_acc'])
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['model_ori', 'model_new'], loc='upper left')
plt.show()
