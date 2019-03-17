import pandas
from keras.models import Sequential
from keras.layers.core import Dense, Activation

# load dataset
from sklearn.model_selection import train_test_split
import pandas as pd
dataset = pd.read_csv("diabetes.csv", header=None).values
# print(dataset)
import numpy as np
X_train, X_test, Y_train, Y_test = train_test_split(dataset[:, 0:8], dataset[:, 8],
                                                    test_size=0.25, random_state=87)

np.random.seed(155)
my_first_nn = Sequential()  # create model
my_first_nn.add(Dense(20, input_dim=8, activation='relu'))  # hidden layer
# my_first_nn.add(Dense(20, input_dim=8, activation='relu'))  # hidden layer
# my_first_nn.add(Dense(20, input_dim=8, activation='relu'))  # hidden layer

#my_first_nn.add(Dense(10, activation='sigmoid')) # output layer
my_first_nn.add(Dense(1, activation='sigmoid')) # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100, verbose=0,
                                     initial_epoch=0)
print(my_first_nn.summary())
loss, ac = my_first_nn.evaluate(X_test, Y_test, verbose=0)
print("The loss is: ", loss)
print("The accuracy is: ", ac)
"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 20)                180       
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 21        
=================================================================
Total params: 201
Trainable params: 201
Non-trainable params: 0
_________________________________________________________________
None
0.6976303805907568

Process finished with exit code 0
"""