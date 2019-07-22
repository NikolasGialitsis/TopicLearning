#basic libraries
import numpy as np
import sys

from keras.layers import LSTM, Dense, TimeDistributed,Activation,Input
from ast import literal_eval
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
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
from keras.layers import Flatten
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
from sklearn.dummy import  DummyClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA

import tensorflow as tf


text_file = open("/home/superuser/SequenceEncoding/Representations/prob_repr.dat", "r")
instances_concat = text_file.readlines()
num_steps = -1
instances = []
for instance in instances_concat:
    vec = literal_eval(instance)
    num_steps = len(vec)
    input_dim = len(vec[0])
    instances.append(vec)

text_file.close()
labels_file = open("/home/superuser/SequenceEncoding/Representations/prob_labels.dat", "r")
labels  = labels_file.readlines()
labels_file.close()

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)
labels = integer_encoded
print 'Instances num = '+ str(len(instances))
print 'Labels num = ' + str(len(labels))
def learning_model(): #Neural Network Architecture#
    global input_dim
    global num_steps
    global input
    model = Sequential()
    print 'dim '+str(input_dim)
    print 'steps '+str(num_steps)
    input_shape=(num_steps,input_dim)
    #model.add(LSTM(10, input_shape=input_shape,dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(10,input_shape=input_shape))
    model.add(Dense(10))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss="binary_crossentropy", optimizer="adam",metrics=['accuracy'])
    return model

# def learning_model(): #Neural Network Architecture#
#     global input_dim
#     global num_steps
#     global input
#     model = Sequential()
#     print 'dim '+str(input_dim)
#     print 'steps '+str(num_steps)
#     input_shape=(num_steps,input_dim)
#     model.add(LSTM(200, input_shape=input_shape,dropout=0.2, recurrent_dropout=0.2))
#     model.add(Dense(200))
#     model.add(Dense(2,activation='softmax'))
#     model.compile(loss="binary_crossentropy", optimizer="adam",metrics=['accuracy'])
#     print model.summary()
#     return model

# fix random seed for reproducibility

epochs = 1
seed = 111
batches = 64

#partition dataset


skf = StratifiedKFold(n_splits=10)
from sklearn.metrics import f1_score
instances = np.array(instances)
labels = np.array(labels)
for train_index, test_index in skf.split(instances, labels):
    #print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = instances[train_index], instances[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    print '---- XTRAIN ----\n'+str(x_train[0])
    print '---- YTRAIN ----\n'+str(y_train[0])

    non_linear_model = learning_model()
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0,patience=5)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    non_linear_model.fit(x =x_train,y=y_train,epochs=epochs, batch_size=batches,validation_data=(x_test,y_test),callbacks=[es])
    # Final evaluation of the model
    print non_linear_model.summary()
    y_pred =non_linear_model.predict(x_test)

    #print set(y_test) - set(y_pred)
    print ('f1 score macro ' + str(f1_score(y_test, y_pred, average='macro')))
    print ('f1 score micro ' + str(f1_score(y_test, y_pred, average='micro')))
    print ('Confusion matrix:\n' + str(confusion_matrix(y_test, y_pred)))


#(x_train, x_test,y_train,y_test) = train_test_split(instances,labels, test_size=0.2, random_state=seed)
# non_linear_model = learning_model()



