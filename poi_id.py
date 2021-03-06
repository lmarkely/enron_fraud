#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### Include all quantitative features. In addition, 'std_from_poi' and
### 'std_to_poi' are standardized feature (see details below).
features_list = ['poi','salary','bonus','expenses','exercised_stock_options',
                 'other','restricted_stock', 'shared_receipt_with_poi',
                 'std_from_poi','std_to_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
# Add new features: std_from_poi and std_to_poi by dividing the message
# to/from poi by the total sent or received messages, respectively.
data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
data_dict.pop('LOCKHART EUGENE E')
missing_data = 0
feature_zeros = {}
for key in data_dict:
    for subkey in data_dict[key]:
        if data_dict[key][subkey] == 'NaN':
            feature_zeros[subkey] = feature_zeros.get(subkey,0)+1
            missing_data += 1
    if (type(data_dict[key]['from_poi_to_this_person']) == int and
        type(data_dict[key]['from_messages']) == int):
        data_dict[key]['std_from_poi'] = \
        (data_dict[key]['from_poi_to_this_person']/
         data_dict[key]['from_messages'])
    else:
        data_dict[key]['std_from_poi'] = 0
    if (type(data_dict[key]['from_this_person_to_poi']) == int and
        type(data_dict[key]['to_messages']) == int):
        data_dict[key]['std_to_poi'] = \
        (data_dict[key]['from_this_person_to_poi']/
         data_dict[key]['to_messages'])
    else:
        data_dict[key]['std_to_poi'] = 0
my_dataset = data_dict
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# The followings are the major steps in the analysis:
# A. Visualize the data using dimensionality reduction PCA and LDA to gain
#    further insight into the data
# B. Algorithm selection using repeated nested cross validation to choose
#    the algorithm that has highest accuracy
# C. Model selection using repeated cross validation to identify the best
#    hyperparameter values
#The following codes will provide only the final answers. Please refer to
#poi_id_modified.py or Enron_fraud.ipynb for all the codes used in this project.

from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import numpy as np

# For simplicity, rename features as X and labels as y
X = features
y = np.array(labels)
### First, explore the dataset.
### Identify the total number of data points.
print 'Total number of data points:',np.shape(X)[0]
print 'Total number of features:', np.shape(X)[1]
print 'Total number of data points with all zeros:',\
       len(data_dict)-np.shape(X)[0]
print 'Total number of missing feature values:', missing_data
print 'Total number of POI:', sum(labels)
print 'Total number of non-POI', np.shape(X)[0] - sum(labels)
print feature_zeros

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

clf_knn = KNeighborsClassifier(n_neighbors=5)
pipe_knn = Pipeline([['sc',MinMaxScaler()],
                     ['kbest',SelectKBest(chi2,k=2)],
                     ['clf',clf_knn]])
clf = pipe_knn
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
