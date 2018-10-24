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
import itertools
from itertools import chain, combinations
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.naive_bayes import *
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
rcParams['figure.figsize'] = 9, 8

df = pd.read_csv('final_dataset-sentment analysis.csv')
# print(len(df))
ndf = df.as_matrix();
from scipy.spatial.distance import cdist
clusters=range(1,26)
meandist=[]
x=np.array(df)

final1=[]
df = pd.read_csv('dataset-sentment analysis.csv')
comp=df['Stress Level']

labels=[]
labels1=[]
listx=[]
final=0
number=0
answer=0
l=[]
values=[0]*26
count=[0]*26
ultra_max=0
heyo=0
ansss=[]

theoutput=[]

head=[0]*26

factors=["BMI","Calories Burned","Steps","Distance","Floors","Minutes Sedentary","Minutes Lightly Active",	"Minutes Fairly Active"	,"Minutes Very Active","Activity Calories","Working Hours","Minutes Asleep","Minutes Awake","Number of Awakenings","Time in Bed"	,"calories"	,"avg_bpm","max","min","minutes","resting heart rate","activity","Other","Fatburn","Cardio","Peak"]

# the innovation we used was to see if the list changed after an iteration, if it did then repeat else go on to next cluster

for op in range (2,27):
	i=[]
	for u in range (0,op):
		i.append(u)
		flag=0
	while i != ansss:
		if flag==1:
			i=ansss
		max1=0
		index=0
		labels=i
		k=len(i)
		values=[0]*26
		count=[0]*26
		ultra_max=0
		ansss=[]
# for each label, only assign if it does not belong to the head of the cluster by using the pearson coefficient 
		for alpha in range(0,26):
			max1=0
			if alpha not in labels:
				for j in range(0,k):
					# print labels[j]
					if max1<abs(pearsonr(ndf[:,alpha],ndf[:,labels[j]])[0]):
						max1=abs(pearsonr(ndf[:,alpha],ndf[:,labels[j]])[0])
						index=j
				values[alpha]=max1
				count[index]=count[index]+1
				head[alpha]=labels[index]
			else:
				head[alpha]=alpha
# after creating the clusters , we choose one label out of each class by comparing the correalion coefficient
# with the target class
		
		to_be_taken=0
		for j in range(0,k):
			max1=0
			for alpha in range(0,26):
				if head[alpha]==labels[j]:
					if max1<abs(pearsonr(ndf[:,alpha],comp)[0]):
						max1=abs(pearsonr(ndf[:,alpha],comp)[0])
						to_be_taken=alpha
			ultra_max=ultra_max+max1
			ansss.append(to_be_taken)
		if final < ultra_max:
			final=ultra_max
			if k==5:
				theoutput=ansss
		if heyo!=op:
			heyo=op
		flag=1


for i in range(0,len(theoutput)):
	theoutput[i]=factors[i]


print theoutput