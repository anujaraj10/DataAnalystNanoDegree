#!/usr/bin/python

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from time import time
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from tester import dump_classifier_and_data
from feature_format import featureFormat, targetFeatureSplit
from sklearn.preprocessing import MinMaxScaler

sys.path.append("/home/aunja/ANUJA/data_analyst_nano/ud120-projects/tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary','deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees','shared_receipt_with_poi' ,'percentage_to_poi','percentage_from_poi']
 # You will need to use more features
initial_features_list = ['poi','salary','deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'shared_receipt_with_poi','to_messages','from_messages', 'from_poi_to_this_person','from_this_person_to_poi']

### Load the dictionary containing the dataset
data_dict = pickle.load(open("/home/aunja/ANUJA/data_analyst_nano/ud120-projects/final_project/final_project_dataset.pkl", "r") )

# The Total data point will not help us in the evaluation
identified_outliers = ["TOTAL"]

print "Original Length", len(data_dict)
for outlier in identified_outliers:
    data_dict.pop(outlier)    

keys = data_dict.keys()

print "Length after Outlier", len(data_dict)    


### Task 2: Remove outliers
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)
### plot features
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )
plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

outliers = []
for key in data_dict:
    val = data_dict[key]['salary']
    if val == 'NaN':
        continue
    outliers.append((key, int(val)))

outliers_final = (sorted(outliers,key=lambda x:x[1],reverse=True)[:4])
### print top 4 salaries
print outliers_final

### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("/home/aunja/ANUJA/data_analyst_nano/ud120-projects/final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]

data_dict.pop('TOTAL',0)

data = featureFormat(data_dict, features)

print data.max()
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )
plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()


### Task 3: Create new feature(s)
def dict_to_list(key,normalizer):
    new_list = []

    for i in data_dict:
        if data_dict[i][key] == "NaN" or data_dict[i][normalizer] == "NaN":
            new_list.append(0)
        elif data_dict[i][key] >= 0:
            new_list.append(float(data_dict[i][key])/float(data_dict[i][normalizer]))
    return new_list

### create two lists of new features
fraction_from_poi_email = dict_to_list("from_poi_to_this_person","to_messages")
fraction_to_poi_email = dict_to_list("from_this_person_to_poi","from_messages")

### insert new features into data_dict
count = 0
for i in data_dict:
    data_dict[i]["fraction_from_poi_email"] = fraction_from_poi_email[count]
    data_dict[i]["fraction_to_poi_email"] = fraction_to_poi_email[count]
    count += 1
features_list = ["poi", "fraction_from_poi_email", "fraction_to_poi_email"] 
 ### store to my_dataset for easy export below
my_dataset = data_dict
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


#plot new features
def graph_scatter_with_poi(var1, var2):
    for name in data_dict:
        point = data_dict[name]
        poi = point['poi']
        x = point[var1]
        y = point[var2]

        if poi:
            plt.scatter( x, y, color='red')
        else:
            plt.scatter( x, y, color='blue')
    plt.xlabel(var1)
    plt.ylabel(var2)

plt.figure(1, figsize=(16, 5))
plt.subplot(1,2,1) 
graph_scatter_with_poi('from_poi_to_this_person', 'from_this_person_to_poi')
plt.subplot(1,2,2) 
graph_scatter_with_poi('fraction_from_poi_email', 'fraction_to_poi_email')
plt.show()


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clfGB = GaussianNB()
from sklearn.svm import SVC
clfSV = SVC(kernel='rbf',C=100)
from sklearn.tree import DecisionTreeClassifier
clfDT= DecisionTreeClassifier()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
#
clfGB.fit(features_train,labels_train)
predGB=clfGB.predict(features_test)
clfSV.fit(features_train,labels_train)
predSV=clfSV.predict(features_test)
clfDT.fit(features_train,labels_train)
predDT=clfDT.predict(features_test)
from sklearn.metrics import classification_report
target_names = ['Not PoI', 'PoI']
print "GaussianNB"
print( classification_report(labels_test, predGB, target_names=target_names))
print "Support Vector Classifier"
print( classification_report(labels_test, predSV, target_names=target_names))
print "Decision Tree"
print( classification_report(labels_test, predDT, target_names=target_names))

from sklearn.grid_search import GridSearchCV
param_grid = {'min_samples_split': np.arange(2, 10)}
tree = GridSearchCV(DecisionTreeClassifier(), param_grid)
tree.fit(features_train, labels_train)
print(tree.best_params_)
clf=tree


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
### features_list is a list of strings, each of which is a feature name
### first feature must be "poi", as this will be singled out as the label
features_list = ["poi", "fraction_from_poi_email", "fraction_to_poi_email", 'shared_receipt_with_poi']


### store to my_dataset for easy export below
my_dataset = data_dict


### these two lines extract the features specified in features_list
### and extract them from data_dict, returning a numpy array
data = featureFormat(my_dataset, features_list)


### split into labels and features (this line assumes that the first
### feature in the array is the label, which is why "poi" must always
### be first in features_list
labels, features = targetFeatureSplit(data)


### machine learning goes here!
### please name your classifier clf for easy export below

### deploying feature selection
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)

### use KFold for split and validate algorithm
from sklearn.cross_validation import KFold
kf=KFold(len(labels),3)
for train_indices, test_indices in kf:
    #make training and testing sets
    features_train= [features[ii] for ii in train_indices]
    features_test= [features[ii] for ii in test_indices]
    labels_train=[labels[ii] for ii in train_indices]
    labels_test=[labels[ii] for ii in test_indices]

from sklearn.tree import DecisionTreeClassifier

t0 = time()

clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
score = clf.score(features_test,labels_test)
print 'accuracy before tuning ', score

print "Decision tree algorithm time:", round(time()-t0, 3), "s"


### use manual tuning parameter min_samples_split
t0 = time()
clf = DecisionTreeClassifier(min_samples_split=7)
clf = clf.fit(features_train,labels_train)
pred= clf.predict(features_test)
print("done in %0.3fs" % (time() - t0))

acc=accuracy_score(labels_test, pred)

print "Validating algorithm:"
print "accuracy after tuning = ", acc

# function for calculation ratio of true positives
# out of all positives (true + false)
print 'precision = ', precision_score(labels_test,pred)

# function for calculation ratio of true positives
# out of true positives and false negatives
print 'recall = ', recall_score(labels_test,pred)


### dump your classifier, dataset and features_list so
### anyone can run/check your results
pickle.dump(clf, open("my_classifier.pkl", "w") )
pickle.dump(data_dict, open("my_dataset.pkl", "w") )
pickle.dump(features_list, open("my_feature_list.pkl", "w") )

dump_classifier_and_data(clf, my_dataset, features_list)

