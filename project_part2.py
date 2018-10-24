import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import math
import numpy as np
import pandas as pd
import nltk
import random
import pickle
import matplotlib
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
# from brew.base import Ensemble
# from brew.base import EnsembleClassifier
# from brew.combination.combiner import Combiner

rcParams['figure.figsize'] = 9, 8
factors=["BMI","Calories Burned","Steps","Distance","Floors","Minutes Sedentary","Minutes Lightly Active",	"Minutes Fairly Active"	,"Minutes Very Active","Activity Calories","Working Hours","Stress Level","Minutes Asleep","Minutes Awake","Number of Awakenings","Time in Bed"	,"calories"	,"avg_bpm","max","min","minutes","resting heart rate","activity","Other","Fatburn","Cardio","Peak"]
df= pd.read_csv("dataset-sentment analysis.csv")
X = np.array(df.drop(['Stress Level'], 1))
y = np.array(df['Stress Level'])


# we find the variable (label) importance value and then display them in the reverse order of decreasing importance
# and then plotted a graph of label vs importance

rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
rnd_clf.fit(X, y)
i=0
gg=[]
label=[]
imp=[]
for importance in zip(rnd_clf.feature_importances_):
	gg.append((factors[i], importance[0]))
	label.append(i)
	imp.append(importance[0])
	i=i+1

gg=sorted(gg,reverse=True,key=lambda x:x[1])


import csv

with open("output.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(gg)

for i in gg:
	print i
# plt.figure(figsize=(20, 3)) 
matplotlib.rcParams.update({'font.size': 8})
# plt.bar(label[:13],imp[:13],color='b',width=0.5,align='center')

# plt.xlabel('Label')
# plt.xticks(label[:13], factors[:13])
# plt.ylabel('Importance')
# plt.legend()
# plt.show()

# plt.bar(label[13:],imp[13:],color='b',width=0.5, align='center')

# plt.xlabel('Label')
# plt.xticks(label[13:], factors[13:])
# plt.ylabel('Importance')
# plt.legend()
# plt.show()