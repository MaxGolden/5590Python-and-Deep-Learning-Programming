import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
import pickle
from keras.callbacks import TensorBoard
from sklearn.preprocessing import LabelEncoder
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import model_from_yaml


def createmodel():
    # load YAML and create model
    yaml_file = open('model_SA.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights("model_SA.h5")
    print("Loaded model from disk")
    # predict data
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return loaded_model


labelencoder = LabelEncoder()
integer_encoded = labelencoder.fit_transform(data['sentiment'])
y = to_categorical(integer_encoded)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# print(X_train.shape, Y_train.shape)
# print(X_test.shape, Y_test.shape)

# J:\5590Dl\DL\ICP5\Max_v\SA_logs
#tensorboard = TensorBoard(log_dir='./SA_logs', histogram_freq=0,write_graph=True, write_images=False)

batch_size = 32
hist = model.fit(X_train, Y_train,
                 epochs=7,
                 batch_size=batch_size,
                 verbose=2,
                 callbacks=[tensorboard])

score, acc = model.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)

model = KerasClassifier(build_fn=createmodel, verbose=0)
batch_size = [32, 64]
epochs = [1, 2, 3, 4]
param_grid = dict(batch_size=batch_size, epochs=epochs)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(X_train, Y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))