
Conversation opened. 4 messages. All messages read.

Skip to content
Using BITS Pilani University Mail with screen readers
Search



Mail
COMPOSE
Labels
Inbox (34)
Starred
Sent Mail
Drafts (28)
More 
Hangouts

 
 
  More 
3 of 309  
 
Collapse all Print all In new window
IR 
Inbox
x 

KUNAL GOPAL RIJHWANI <f20150076@goa.bits-pilani.ac.in>
Attachments1:49 PM (8 hours ago)

to me, KUNAL, ANSHUL 
ive sent all codes and csv, you have graphs. report is your work now. last.py ko add as innovation, as instead of taking all powersets , i did that repition vaala method, if the list changes or not.

final_dataset :- without the target class label to do clustering
dataset :- with the preprocessed data
Dataset :- onlly the 5 class labels that we use for predicting


output.csv = the class labels and their importance.

readme explains alll codes
9 Attachments 
 

SIDDHARTH J SHAILENDRA <f20150059@goa.bits-pilani.ac.in>
Attachments2:09 PM (8 hours ago)

to KUNAL 
Send me all the graphs and tables that he asked for.  I can't do it by running these codes on my laptop.

9 Attachments 
 

KUNAL GOPAL RIJHWANI <f20150076@goa.bits-pilani.ac.in>
Attachments3:15 PM (7 hours ago)

to me 
all final values are in csv as mentioned in sir ka mail, ek hi table hai , baaki sir ke paper se utha, clustering and all

4 Attachments 
 

KUNAL GOPAL RIJHWANI <f20150076@goa.bits-pilani.ac.in>
Attachments6:25 PM (4 hours ago)

to me 
the previous mail had a compilation error in last.py file.
this is the updated one

Attachments area
	
Click here to Reply or Forward
Using 3.73 GB
Program Policies
Powered by Google
Last account activity: 5 minutes ago
Details


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

# to generate powerset i.e all possible combinations as we have to check all possible subsets as heads of the clusters
def powerset_generator(i):
    for subset in chain.from_iterable(combinations(i, r) for r in range(len(i)+1)):
        yield list(subset)
df = pd.read_csv('dataset-sentment analysis.csv')
comp=df['Stress Level']

# array to create subsets of all the features 
arr=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
final1=[]


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

head=[0]*26
# so for all these subsets , we assign the rest of the features to either one of the clusters by calculating the 
# correlation coefficient and assigning it to that head with which it has max coefficient

for i in powerset_generator(arr):
	if i!=[]:
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
					flag=0
					if max1<abs(pearsonr(ndf[:,alpha],ndf[:,labels[j]])[0]):
						max1=abs(pearsonr(ndf[:,alpha],ndf[:,labels[j]])[0])
						index=j
						flag=1
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
		# ultra_max=ultra_max/k
		if final < ultra_max:
			final=ultra_max
			print (final, ansss, k)
		if heyo!=k:
			print k
			heyo=k
# print number,final,count,ansss

try.py
Open with
Displaying try.py.