#basic libraries
import numpy as np
import sys

from keras.layers import LSTM, Dense, TimeDistributed,Activation,Input
from ast import literal_eval
import numpy
import pandas
import scipy
import joblib
import keras
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

#Deep learning modules
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

from keras import models
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA

import tensorflow as tf


text_file = open("/home/superuser/SequenceEncoding/Representations/tfidf_repr.dat", "r")
instances_concat = text_file.readlines()
num_steps = -1
instances = []
for instance in instances_concat:
    vec = literal_eval(instance)
    num_steps = len(vec)
    input_dim = len(vec[0])
    instances.append(vec)

text_file.close()
labels_file = open("/home/superuser/SequenceEncoding/Representations/tfidf_labels.dat", "r")
labels  = labels_file.readlines()
labels_file.close()

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)
labels = integer_encoded


def learning_model(): #Neural Network Architecture#
    global input_dim
    global num_steps
    global input
    model = Sequential()
    print 'dim '+str(input_dim)
    print 'steps '+str(num_steps)
    input_shape=(num_steps,input_dim)
    model.add(LSTM(200, input_shape=input_shape,dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(200))
    model.add(Dense(1,activation='softmax'))
    model.compile(loss="binary_crossentropy", optimizer="adam",metrics=['accuracy'])
    print model.summary()
    return model

# fix random seed for reproducibility

epochs = 2
seed = 40
batches = 64


#partition dataset
(x_train, x_test,y_train,y_test) = train_test_split(instances,labels, test_size=0.2, random_state=seed)
non_linear_model = learning_model()
es = EarlyStopping(monitor='loss', mode='min', verbose=0)
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
non_linear_model.fit(x =x_train,y=y_train,epochs=epochs, batch_size=batches,validation_data=(x_test,y_test),callbacks=[es])
# Final evaluation of the model

y_pred =non_linear_model.predict(x_test)
from sklearn.metrics import f1_score
#print set(y_test) - set(y_pred)
print 'f1 score macro ' + str(f1_score(y_test, y_pred, average='macro'))
print 'f1 score micro ' + str(f1_score(y_test, y_pred, average='micro'))
