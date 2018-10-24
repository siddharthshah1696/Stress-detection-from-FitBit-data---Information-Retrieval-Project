import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import math
import numpy as np
import pandas as pd
import nltk
import random
import pickle
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
from scipy.stats.stats import pearsonr
from pylab import rcParams
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, svm
import pandas as pd
from sklearn.naive_bayes import *
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier



df = pd.read_csv("Dataset-sentment analysis.csv")
X = np.array(df.drop(['Stress Level'], 1))
# X = preprocessing.scale(X)
y = np.array(df['Stress Level'])


# split the data into test set and training set

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.08)

print (len(X_train), len(X_test), len(y_train), len(y_test))

# applied various learning algorithms to this and compared the scores


mnb = MultinomialNB()
mnb.fit(X_train, y_train)
scoreForMNB = mnb.score(X_test, y_test)
print("MultinomialNB: ",scoreForMNB)

clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
scoreForSVCrbf = clf.score(X_test, y_test)
print("Rbf SVC: ",scoreForSVCrbf)

clf2 = svm.SVC(kernel='sigmoid')
clf2.fit(X_train, y_train)
scoreForSVCsigmoid = clf2.score(X_test, y_test)
print("Sigmoid SVC: ",scoreForSVCsigmoid)


bnb = BernoulliNB()
bnb.fit(X_train, y_train)
scoreForBNB = bnb.score(X_test, y_test)
print("BernoulliNB: ",scoreForBNB)

rfClf = RandomForestClassifier(n_estimators=100)
rfClf = rfClf.fit(X_train, y_train)
scoreForRf = rfClf.score(X_test, y_test)
print("Random Forest: ",scoreForRf)

model = tree.DecisionTreeClassifier()
model.fit(X_train,y_train)
scoreForDT = model.score(X_test, y_test)
print("Decision Tree: ",scoreForDT)





dataframe = pd.read_csv("Dataset-sentment analysis.csv")
X = np.array(dataframe.drop(['Stress Level'], 1))
y = np.array(dataframe['Stress Level'])
seed = 7
num_trees = 30
kfold = model_selection.KFold(n_splits=10, random_state=seed)
# create the sub models
estimators = []
model1 =  RandomForestClassifier(n_estimators=100)
estimators.append(('RandomForestClassifier', model1))
model2 = RandomForestClassifier(n_estimators=100)
estimators.append(('RandomForestClassifier', model2))
model3 =  RandomForestClassifier(n_estimators=100)
estimators.append(('RandomForestClassifier', model3))
# create the ensemble model
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, X, y, cv=kfold)
print("Ensemble Classifier: ",results.mean())


kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, y, cv=kfold)
print("Adaboost: ",results.mean())


num_trees = 20
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, y, cv=kfold)
print("SGD: ",results.mean())