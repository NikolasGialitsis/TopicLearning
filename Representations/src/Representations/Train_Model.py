#basic libraries
import numpy as np
import sys
from sklearn.metrics import f1_score
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dropout, LSTM, Dense, TimeDistributed,Activation,Input
from ast import literal_eval
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn import tree
import numpy
import pandas
import scipy
import joblib
import keras
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from sklearn.naive_bayes import GaussianNB
#Deep learning modules
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras import models
import pickle
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
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
from sklearn.tree import DecisionTreeClassifier

import tensorflow as tf
num_steps = 8
input_dim = 0
seed = 111
def LSTM_NN(): #Neural Network Architecture#
    model = Sequential()
    input_shape=(num_steps,input_dim)
    print("lstm input shape:", input_shape)
    model.add(LSTM(200,return_sequences=False))  # returns a sequence of vectors of dimension 32
    model.add(Dense(200))
    model.add(Dense(1,input_dim=32, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
def main():


    repr = "tfidf"
    for arg in sys.argv:
        if arg == "-prob":
            repr = "prob"
    print('Training model on '+repr+' representation...')
    text_file = open("/home/superuser/SequenceEncoding/Representations/"+repr+"_repr_train.dat", "r")
    print('...opened file')
    instances_concat = text_file.readlines()
    text_file.close()

    instances = []
    print('...append instances' )
    count = 0

    for instance in instances_concat:
        global num_steps
        global input_dim
        print('\tInstance:'+str(count))
        count = count + 1
        vec = literal_eval(instance)
        num_steps = len(vec)
        input_dim = len(vec[0])
        instances.append(vec)
    classifiers = [
        #KNeighborsClassifier(3),
        #SVC(kernel="rbf", C=0.025, probability=True),
        #NuSVC(probability=True),
        #LSTM_NN(),
        DecisionTreeClassifier(random_state=seed),
        RandomForestClassifier(random_state=seed),
        AdaBoostClassifier(random_state=seed),
        GradientBoostingClassifier(random_state=seed),
        GaussianNB(),
        LinearDiscriminantAnalysis(),

        DummyClassifier(strategy="stratified",random_state=seed)
        #QuadraticDiscriminantAnalysis()

    ]
    text_file.close()
    print('...open label')
    labels_file = open("/home/superuser/SequenceEncoding/Representations/"+repr+"_labels_train.dat", "r")
    labels  = labels_file.readlines()
    labels_file.close()

    print ('...close label')
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    labels = integer_encoded
    print('Instances num = '+ str(len(instances)))
    print('Labels num = ' + str(len(labels)))


    #partition dataset
    print(sys.argv)
    validate = False
    only_nn = False
    for arg in sys.argv:
        if  arg == "-validate":
            validate = True
            continue
        if arg == "-only_nn":
            only_nn = True
            continue


    if validate == True:
        print("=== Validation mode ===")
        n_splits = 10
        skf = StratifiedKFold(n_splits=n_splits)


        instances = np.array(instances)
        labels = np.array(labels)
        for clf in classifiers:

            #print("+++ Starting fold...")
            micro_sum = 0.0
            macro_sum = 0.0
            name = non_linear_model.__class__.__name__
            if only_nn and name is not 'Sequential':
                continue
            for train_index, test_index in skf.split(instances, labels):
                x_train , x_test = [] , []
                if(name == 'Sequential'):

                    x_train, x_test = np.array(instances[train_index]), np.array(instances[test_index])
                    x_train, x_test = x_train[:, :num_steps, :],  x_test[:,:num_steps,:]
                else:
                    flattenedInstances = np.array([np.ndarray.flatten(xVec) for xVec in instances])
                    x_train, x_test = np.array(flattenedInstances[train_index]), np.array(flattenedInstances[test_index])
                y_train, y_test = np.array(labels[train_index]), np.array(labels[test_index])

                non_linear_model = clf
                name = non_linear_model.__class__.__name__
                print(name)
                non_linear_model.fit(x_train, y_train)

                #print("+ Fitting model... Done.")
                x_test = np.array(x_test)
                y_test = np.array(y_test)
                #print("+ Predicting...")
                y_pred = non_linear_model.predict(x_test)
                #print("+ Predicting... Done.")

                micro_sum = micro_sum + f1_score(y_test,y_pred,average="micro")
                macro_sum = macro_sum + f1_score(y_test,y_pred,average="macro")

            print(str(name))
            print('==================================')
            print('f1 score macro ' + str(macro_sum/n_splits))
            print('f1 score micro ' + str(micro_sum/n_splits))
            print('==================================')
    else:
        print("=== Training mode ===")
        for cnf in classifiers:

            instances = np.array(instances)
            labels = np.array(labels)
            non_linear_model = cnf
            print(cnf)
            name = non_linear_model.__class__.__name__
            print('-- Training with ' +name)
            if only_nn and name is not 'Sequential':
                continue
            y_train = np.array(labels)
            if(name == 'Sequential'):
                x_train = np.array(instances)
                x_train = x_train[:, :num_steps, :]
                non_linear_model.fit(x_train, y_train,batch_size=32,epochs=10,shuffle=False)
            else:
                flattenedInstances = np.array([np.ndarray.flatten(xVec) for xVec in instances])
                x_train = np.array(flattenedInstances)
                non_linear_model.fit(x_train, y_train)
            # save model to file
            pickle.dump(non_linear_model, open("TrainedModels/"+name + ".pickle", "wb"))
if __name__ == '__main__':
    print 'main init'
    main()
