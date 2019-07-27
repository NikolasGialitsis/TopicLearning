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


classifiers = [
    #KNeighborsClassifier(3),
    #SVC(kernel="rbf", C=0.025, probability=True),
    #NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    DummyClassifier(strategy="stratified",random_state=111)
    #QuadraticDiscriminantAnalysis()
    ]

def main():
    repr = "tfidf"
    for arg in sys.argv:
        if arg == "-prob":
            repr = "prob"
    print('Training model on '+repr+' representation...')
    text_file = open("/home/superuser/SequenceEncoding/Representations/"+repr+"_repr_train.dat", "r")
    instances_concat = text_file.readlines()
    time_steps = 10
    num_steps = -1
    instances = []
    for instance in instances_concat:
        vec = literal_eval(instance)
        num_steps = len(vec)
        instances.append(vec)

    text_file.close()
    labels_file = open("/home/superuser/SequenceEncoding/Representations/"+repr+"_labels_train.dat", "r")
    labels  = labels_file.readlines()
    labels_file.close()


    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    labels = integer_encoded
    print('Instances num = '+ str(len(instances)))
    print('Labels num = ' + str(len(labels)))
    def learning_model(): #Neural Network Architecture#
        global input_dim
        global num_steps
        global input_shape

        model = Sequential()
        print('dim '+str(input_dim))
        print('steps '+str(num_steps))
        input_shape=(time_steps,input_dim)
        print("lstm input shape:", input_shape)
        model.add(LSTM(10, input_shape=input_shape, dropout=0.2, recurrent_dropout=0.2))

        #     model.add(Dense(100))
        #     model.add(Dropout(0.2))
        #     model.add(Dense(100))
        model.add(Dense(1, activation='softmax'))
        model.compile(loss="binary_crossentropy", optimizer="adam",metrics=['accuracy'])
        model.summary()
        return model



    # fix random seed for reproducibility

    epochs = 100
    seed = 111
    batches = 64

    #partition dataset
    print(sys.argv)
    validate = False
    if  len(sys.argv) > 1 and sys.argv[1] == "-validate":
        validate = True

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

            for train_index, test_index in skf.split(instances, labels):
                #print("TRAIN:", train_index, "TEST:", test_index)
                #print("+ Flattening instances...")
                flattenedInstances = np.array([np.ndarray.flatten(xVec) for xVec in instances])
                #print("+ Flattening instances... Done.")
                x_train, x_test = np.array(flattenedInstances[train_index]), np.array(flattenedInstances[test_index])
                #x_train, x_test = np.array(instances[train_index]), np.array(instances[test_index])
                #x_train, x_test = x_train[:,:time_steps,:],  x_test[:,:time_steps,:]
                y_train, y_test = np.array(labels[train_index]), np.array(labels[test_index])
                #print('---- XTRAIN 1st instance ----\n'+str(x_train[0]))
                #print('---- YTRAIN 1st instance  ----\n'+str(y_train[0]))


                input_dim = x_train.shape[-1]
                #print("Shape:" + str(x_train.shape))

                #print("+ Fitting model...")

                non_linear_model = clf
                name = non_linear_model.__class__.__name__
                print(name)
                non_linear_model.fit(x_train, y_train)


                #================================
                # Decision Tree Classifier : almost equals to dummy [stratified]
                #================================
                #non_linear_model = tree.DecisionTreeClassifier(criterion = "gini",\
                #                                              random_state = seed,max_depth=3, min_samples_leaf=5)
                #non_linear_model.fit(x_train, y_train)


                #================================
                # Gaussian Classifier : below baseline [with flattened instances]
                #================================
                #non_linear_model = GaussianNB()
                #non_linear_model.fit(x_train, y_train)

                #================================
                # Keras NN classifier : equals baseline
                #================================
                #non_linear_model = KerasClassifier(build_fn= learning_model, epochs=epochs)
                #es = EarlyStopping(monitor='loss', mode='min', verbose=0,patience=5)
                #non_linear_model.fit(x=x_train,y=y_train,validation_data=(x_test,y_test),callbacks=[es])

                #================================
                # Dummy classifier
                #================================
                #non_linear_model = DummyClassifier(strategy="most_frequent",random_state=seed)
                #non_linear_model.fit(X=x_train,y=y_train)

                #print("+ Fitting model... Done.")
                x_test = np.array(x_test)
                y_test = np.array(y_test)
                #print("+ Predicting...")
                y_pred = non_linear_model.predict(x_test)
                #print("+ Predicting... Done.")

                micro_sum = micro_sum + f1_score(y_test,y_pred,average="micro")
                macro_sum = macro_sum + f1_score(y_test,y_pred,average="macro")
            name = non_linear_model.__class__.__name__
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

            flattenedInstances = np.array([np.ndarray.flatten(xVec) for xVec in instances])
            x_train = np.array(flattenedInstances)
            y_train = y_test = np.array(labels)
            name = non_linear_model.__class__.__name__
            print('-- Training with ' +name)
            non_linear_model.fit(x_train, y_train)
            # save model to file
            pickle.dump(non_linear_model, open("TrainedModels/"+name + ".pickle", "wb"))

if __name__ == '__main__':
    main()
