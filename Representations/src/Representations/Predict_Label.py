#basic libraries
import numpy as np
import sys
from sklearn.metrics import f1_score
from keras.wrappers.scikit_learn import KerasClassifier
from ast import literal_eval
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn import tree
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
import pickle
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
import sklearn
from sklearn.linear_model import LinearRegression
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


classifiers = [
    DecisionTreeClassifier(),
    KNeighborsClassifier()
    #SVC(),
    #NuSVC(),
    #QuadraticDiscriminantAnalysis()
    #Sequential(),
    #RandomForestClassifier(),
    #AdaBoostClassifier(),
    #GradientBoostingClassifier()
    #GaussianNB(),
    #LinearDiscriminantAnalysis(),
    #SGDClassifier(),
    #DummyClassifier(),
    #sklearn.linear_model.LogisticRegression()
]


def main():
    repr = "tfidf"
    for arg in sys.argv:
        if arg == "-prob":
            repr = "prob"
    print('Selected '+repr+ " representation")
    print('Loading test data...')
    text_file = open("/home/superuser/SequenceEncoding/Representations/"+repr+"_repr_test.dat", "r")
    instances_concat = text_file.readlines()
    instances = []
    input_n_features = 0
    topic_num = 0
    instances_num = 0;
    for instance in instances_concat:
        vec = literal_eval(instance)
        input_n_features = len(vec)
        topic_num = len(vec[0])

        instances.append(vec)
        instances_num = instances_num+1
    print(input_n_features,topic_num)
    print('\tinstances num = '+str(instances_num))
    text_file.close()
    labels_file = open("/home/superuser/SequenceEncoding/Representations/"+repr+"_labels_test.dat", "r")
    labels  = labels_file.readlines()
    labels_file.close()
    print('...Done\n')
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    labels = integer_encoded


    labels = np.array(labels)
    name = classifiers[0].__class__.__name__

    loaded_model = pickle.load(open(repr+"_TrainedModels/"+name+".pickle", "rb"))
    model_n_features = 0
    exclude_classifiers = ['Sequential','KNeighborsClassifier','SVC']
    model_n_features = loaded_model.n_features_/(topic_num)
    model_n_features = (int)(model_n_features)

    print('model n_features = '+str(model_n_features))
    print('input n_features = '+str(input_n_features))
    print('values per term = '+str(topic_num))
    #add padding so that the input and the model features are compatible
    if input_n_features < model_n_features:
        for index in xrange(0,len(instances)):
            sentence = instances[index]
            for x in xrange(input_n_features,model_n_features):
                word_vec = np.zeros(topic_num)
                sentence.append(word_vec)
    

    instances = np.array(instances)

    flattenedInstances = np.array([np.ndarray.flatten(xVec) for xVec in instances])

    for clf in classifiers:
        x_test = instances
        print('input shape ' + str(x_test.shape) + '\n')
        name = clf.__class__.__name__
        if(name is 'Sequential'):
            x_test = x_test[:, :model_n_features, :]
        else:
            x_test = np.array(flattenedInstances)

        y_test = np.array(labels)
        print('Loading saved trained model ...')
        # load model from file

        print(name)
        print('transformed shape '+ str(x_test.shape) + '\n')
        loaded_model = pickle.load(open(repr+"_TrainedModels/"+name+".pickle", "rb"))
        # make predictions for test data
        print('Make predictions...')
        y_pred = loaded_model.predict(x_test)
        print('shape of predicted labels :'+str( y_pred.shape))
        print y_pred
        if name is not 'Sequential': #need to figure out why it doesnt work for NN
            pickle.dump(np.ndarray(shape=(y_pred.shape),dtype=int,buffer=y_pred), open("results/"+repr+"_"+name + ".pickle", "wb"))

        print('...Done\n')
        micro_sum = f1_score(y_test,y_pred,average="micro")
        macro_sum = f1_score(y_test,y_pred,average="macro")
        print('\tPrediction results')
        print('==================================')
        print('f1 score macro ' + str(macro_sum))
        print('f1 score micro ' + str(micro_sum))
        print('==================================')

if __name__ == '__main__':
    main()
