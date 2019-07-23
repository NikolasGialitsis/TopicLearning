import pickle
from ast import literal_eval
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score

print('Loading test data...')
text_file = open("/home/superuser/SequenceEncoding/Representations/tfidf_repr.dat", "r")
instances_concat = text_file.readlines()
instances = []
for instance in instances_concat:
    vec = literal_eval(instance)
    num_steps = len(vec)
    instances.append(vec)

text_file.close()
labels_file = open("/home/superuser/SequenceEncoding/Representations/tfidf_labels.dat", "r")
labels  = labels_file.readlines()
labels_file.close()
print('...Done\n')
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)
labels = integer_encoded

instances = np.array(instances)
labels = np.array(labels)
non_linear_model = GradientBoostingClassifier()

flattenedInstances = np.array([np.ndarray.flatten(xVec) for xVec in instances])
x_test = np.array(flattenedInstances)
y_test = np.array(labels)

print('Loading saved trained model...')
# load model from file
loaded_model = pickle.load(open("non_linear_model.pickle", "rb"))
print('...Done\n')

# make predictions for test data
print('Make predictions...')
y_pred = loaded_model.predict(x_test)
print('...Done\n')
micro_sum = f1_score(y_test,y_pred,average="micro")
macro_sum = f1_score(y_test,y_pred,average="macro")
print('\tPrediction results')
print('==================================')
print('f1 score macro ' + str(macro_sum))
print('f1 score micro ' + str(micro_sum))
print('==================================')