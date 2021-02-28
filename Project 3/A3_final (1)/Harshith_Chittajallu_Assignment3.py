# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:42:00 2020

@author: harsh
"""
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import math
from collections import Counter
df = pd.read_csv (r'C:\Users\harsh\Downloads\CGMData.csv')
df2 = pd.read_csv (r'C:\Users\harsh\Downloads\InsulinData.csv')


df['Sensor Glucose (mg/dL)'] = df['Sensor Glucose (mg/dL)'].interpolate(method='polynomial', order=2)


df2.loc[:, "Datetime"] = pd.to_datetime(df2.loc[:, "Date"]+" "+ df2.loc[:, "Time"])

df.loc[:, "Datetime"] = pd.to_datetime(df.loc[:, "Date"]+" "+ df.loc[:, "Time"])

t=df2[df2['BWZ Carb Input (grams)'].notnull()]
t0 = t[t['BWZ Carb Input (grams)'] != 0]

# start A3 from here
max_carb = t0['BWZ Carb Input (grams)'].max()
min_carb = t0['BWZ Carb Input (grams)'].min()

bin_no = math.ceil((t0['BWZ Carb Input (grams)'].max()-t0['BWZ Carb Input (grams)'].min())/20)
# got 7 bins now these will be the labels

pre_bin_label=[] 
indexe_values = [] #store index values of these bin labels?
for k in range(0,len(t0)):
    for l in range(0,bin_no):
        if l==0:        
            if t0.iloc[k,24] <   t0['BWZ Carb Input (grams)'].min() + 20:  #initial case 
                pre_bin_label.append(0)
        elif (t0['BWZ Carb Input (grams)'].min() + l*20) <= t0.iloc[k,24] <   (t0['BWZ Carb Input (grams)'].min() + (l+1)*20): 
            pre_bin_label.append(l)
#bin_label is a success.


timestamps_insulin = t0.iloc[:,47]  #havent considred subject 2
ti= timestamps_insulin.to_list()



timestamps_CGM = df.iloc[:,47]
ti_cgm=timestamps_CGM.to_list()



i = np.argmin(np.abs(df.iloc[:,47] - t0.iloc[746,47])) #code for finding the equivalent CGM index value for carb input

#threshold: closest datapoint should not be more than 10 mins = 600 sec
e=df.iloc[i,47]-t0.iloc[746,47]

#now test code for valid CGM meal start times begins:
cgm_index=[] #store index values of cgm meal times
# create a separate list for labelling these index values according to their bins
pre_bin_label2=[] #stores the bin value for corresponding cgm index
for j in range(len(t0)):
    i = np.argmin(np.abs(df.iloc[:,47] - t0.iloc[j,47]))
    e=df.iloc[i,47]-t0.iloc[j,47]
    if e.total_seconds() <= 400:
        cgm_index.append(i)
        pre_bin_label2.append(pre_bin_label[j])
    j-=1


    
#cgm meal data timestamps
cgm_meal_timestamps= df.iloc[cgm_index[:],47]


#start megaloop for meal data extraction
pre_bin_label3=[]
meal_data = [] 
label=[]
meal_index=len(cgm_index)-1
temp=meal_index
c=len(df)-1
while c>0:       # update the bin_loop k value based on the meal index position since not every meal is valid 
    meal_sample=[]
    if meal_index >=0:
        if df.iloc[cgm_index[meal_index],47] < df.iloc[c,47] + pd.Timedelta(hours=2):
            pre_bin_label3.append(pre_bin_label2[meal_index])#meal present
            if df.iloc[cgm_index[meal_index-1],47] < df.iloc[cgm_index[meal_index],47]+ pd.Timedelta(hours=2): #meal case2 ignore current meal index and enter next meal index
                mask_df = (df['Datetime'] > df.iloc[cgm_index[meal_index-1],47]-pd.Timedelta(hours=0.5) ) & (df['Datetime'] <= df.iloc[cgm_index[meal_index-1],47]+ pd.Timedelta(hours=2))
                seg=df.loc[mask_df]
                meal_sample= seg['Sensor Glucose (mg/dL)'].tolist()
                meal_sample.reverse()
                meal_data.append(meal_sample)
                label.append(1)
                c= np.argmin(np.abs(df.iloc[:,47] - (df.iloc[cgm_index[meal_index-1],47]+pd.Timedelta(hours=2)))) #get nearest value to next meal's no meal start
                meal_index=meal_index-2 # go to next next meal index
            
            elif df.iloc[cgm_index[meal_index-1],47] == df.iloc[cgm_index[meal_index],47]+ pd.Timedelta(hours=2): #meal case 3
                mask_df = (df['Datetime'] > df.iloc[cgm_index[meal_index],47]+pd.Timedelta(hours=1.5) ) & (df['Datetime'] <= df.iloc[cgm_index[meal_index],47]+ pd.Timedelta(hours=4))
                seg=df.loc[mask_df]
                meal_sample= seg['Sensor Glucose (mg/dL)'].tolist()
                meal_sample.reverse()
                meal_data.append(meal_sample)
                label.append(1)

                c= np.argmin(np.abs(df.iloc[:,47] - (df.iloc[cgm_index[meal_index-1],47]+pd.Timedelta(hours=2))))
                meal_index=meal_index-2
            else:
                mask_df = (df['Datetime'] > df.iloc[cgm_index[meal_index],47]-pd.Timedelta(hours=0.5) ) & (df['Datetime'] <= df.iloc[cgm_index[meal_index],47]+ pd.Timedelta(hours=2))
                seg=df.loc[mask_df]
                meal_sample= seg['Sensor Glucose (mg/dL)'].tolist()
                meal_sample.reverse()
                meal_data.append(meal_sample)
                label.append(1)
                meal_index=meal_index-1
                c= np.argmin(np.abs(df.iloc[:,47] - (df.iloc[cgm_index[meal_index],47]+pd.Timedelta(hours=2))))
           
        else:    
           c= np.argmin(np.abs(df.iloc[:,47] - (df.iloc[c,47]+pd.Timedelta(hours=2)))) 
    else:
        c=c-24
    
#megaloop for combined meal-no meal extraction

# warning pre_bin_label3 and the meal data is reversed because of the loop
premealpre= meal_data
nomealpre=[]
mealpre=[]
label_final = [] 



for z in range(0,len(premealpre)): #filtering out inefficient meal data
    if len(premealpre[z]) >=29:
        mealpre.append(premealpre[z])
        label_final.append(pre_bin_label3[z])  # filtering corresponding bins




from scipy.stats import kurtosis,skew
import statistics as st
import scipy.fftpack as fft

# l1 norm of a vector
from numpy import array
from numpy.linalg import norm
from numpy import inf


def meanabs(data):
    dfx = pd.DataFrame(data) 
  
# Absolute mean deviation 
    a=dfx.mad()
    return a
def cgm_velocity(data):
    initial = min(data)  
    final= data[len(data)-1]
    velocity = (final-initial)/len(data)
    
    return velocity




def fastfourier1(data):
    fourier = fft.rfft(data)
    fourier = abs(fourier)
    peak_1 = max(fourier)
    new_fourier=[]
    for j in range(len(fourier)):
        if fourier[j] != peak_1:
            new_fourier.append(fourier[j])
    peak_2=max(new_fourier)
    return peak_2

def fastfourier2(data):
    fourier = fft.rfft(data)
    fourier = abs(fourier)
    peak_1 = max(fourier)
    new_fourier=[]
    for j in range(len(fourier)):
        if fourier[j] != peak_1:
            new_fourier.append(fourier[j])
    peak_2=max(new_fourier)
    new_fourier2=[]
    for p in range(len(new_fourier)):
        if new_fourier[p] != peak_1:
            new_fourier2.append(new_fourier[p])
    peak_3=max(new_fourier2)
    return peak_3



    
def maxdev(data):
    l = max([max(data)-st.mean(data), st.mean(data)-min(data)])
    return l

f1=[]
f2=[]
f3=[]
f4=[]
f5=[]

f7=[]
f8=[]
f9=[]

for w in range(0, len(mealpre)):
    f1.append(norm(mealpre[w]))
    f2.append(norm(mealpre[w], inf))
    f3.append(maxdev(mealpre[w]))
    f4.append(cgm_velocity(mealpre[w]))
    f5.append(kurtosis(mealpre[w]))
    
    f7.append(skew(mealpre[w]))
    f8.append(fastfourier1(mealpre[w]))
    f9.append(np.std(mealpre[w]))
    

f1=np.array(f1)
f2=np.array(f2)
f3=np.array(f3)
f4=np.array(f4)
f5=np.array(f5)

f7=np.array(f7)
f8=np.array(f8)
f9=np.array(f9)    


f1.reshape(len(f1),1)
f2.reshape(len(mealpre),1)
f3.reshape(len(mealpre),1)
f4.reshape(len(mealpre),1)
f5.reshape(len(mealpre),1)

f7.reshape(len(mealpre),1)
f8.reshape(len(mealpre),1)
f9.reshape(len(mealpre),1)
    
    



label_final2=label_final
label_final2=np.array(label_final2)
label_final2.reshape(-1,1)




training_data=np.column_stack((f1,f2,f3,f4,f5,f7,f8,f9,label_final2))
X=training_data
y=label_final2

 #each index of ground truth represents the no of points that should be per cluster ideally
ground_truth = dict(Counter(label_final2))



from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.svm import SVC
import pickle
import sklearn
from sklearn import cluster
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,recall_score, precision_score, f1_score, classification_report
from sklearn.ensemble import AdaBoostClassifier
import pickle
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import DBSCAN
from numpy import unique
from numpy import where
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler 


untrain_data= pd.DataFrame(data=X,columns=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "class"])
base_data = untrain_data.sample(frac = 1).reset_index(drop=True)

test_sample_length= len(base_data)*0.3 # 70-30 split
test_sample_length = int(test_sample_length)


test_data = base_data.iloc[:test_sample_length,:] 
train_data = base_data.iloc[test_sample_length+1:,:] 

x_train= train_data.iloc[:,[0,1,2,3,4,5,6,7]]
x_test= test_data.iloc[:,[0,1,2,3,4,5,6,7]]
y_train= train_data.iloc[:,[8]]
y_test= test_data.iloc[:,[8]]

sample_x = untrain_data.iloc[:,[0,1,2,3,4,5,6,7]]
mat2 = sample_x.values
scaler= MinMaxScaler(feature_range=(0,1))
mat2= scaler.fit_transform(mat2)
#accuracy=SupportVectorMachine(x_train, x_test, y_train.values.ravel(), y_test.values.ravel())
#accuracy
# Convert DataFrame to matrix
mat = sample_x.values
# Using sklearn
km = sklearn.cluster.KMeans(n_clusters=bin_no)
km.fit(mat)
# Get cluster assignment labels
labels = km.labels_
# Format results as a DataFrame
results = pd.DataFrame([sample_x.index,labels]).T
kmeans_result = dict(Counter(results.iloc[:,1]))

db = DBSCAN(eps=74, min_samples=3).fit(sample_x)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels2 = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels2)) - (1 if -1 in labels2 else 0)
n_noise_ = list(labels2).count(-1)

#def calculate_kn_distance(X,k):

   # kn_distance = []
    #for i in range(len(X)):
     #   eucl_dist = []
      #  for j in range(len(X)):
       #     eucl_dist.append(
        #        math.sqrt(((X[i,0] - X[j,0]) ** 2) + ((X[i,1] - X[j,1]) ** 2)+ ((X[i,2] - X[j,2]) ** 2)+ ((X[i,3] - X[j,3]) ** 2)+((X[i,4] - X[j,4]) ** 2)+
         #           ((X[i,5] - X[j,5]) ** 2)+
          #          ((X[i,6] - X[j,6]) ** 2)+
           #         ((X[i,7] - X[j,7]) ** 2)))

  #      eucl_dist.sort()
   #     kn_distance.append(eucl_dist[k])

    #return kn_distance

#eps_dist = calculate_kn_distance(mat,3)
#plt.hist(eps_dist,bins=30)
#plt.ylabel('n');
#plt.xlabel('Epsilon distance');
#plt.show()

#scam dbscan bro

#Start calculating SSE:
# find centroids for knn
cluster_index= []
centroids=[]  
for f in range(bin_no):
    clustered=[]      
    for e in range(len(labels)):
        if labels[e] ==f:
            clustered.append(mat2[e])
    clustered = np.array(clustered)    
    centroid = [sum(clustered[:,0])/len(clustered),sum(clustered[:,1])/len(clustered),sum(clustered[:,2])/len(clustered),sum(clustered[:,3])/len(clustered),sum(clustered[:,4])/len(clustered),sum(clustered[:,5])/len(clustered),sum(clustered[:,6])/len(clustered),sum(clustered[:,7])/len(clustered)]
    centroids.append(centroid)
centroids = np.array(centroids)    
    
cluster_index2= []
centroids2=[]  
for f in range(max(labels2)):
    clustered=[]      
    for e in range(len(labels2)):
        if labels2[e] ==f:
            clustered.append(mat2[e])
    clustered = np.array(clustered)    
    centroid = [sum(clustered[:,0])/len(clustered),sum(clustered[:,1])/len(clustered),sum(clustered[:,2])/len(clustered),sum(clustered[:,3])/len(clustered),sum(clustered[:,4])/len(clustered),sum(clustered[:,5])/len(clustered),sum(clustered[:,6])/len(clustered),sum(clustered[:,7])/len(clustered)]
    centroids2.append(centroid)
centroids2 = np.array(centroids2)    
    
#SSE starts here
SSE1=0
actual_SSE=[]
for i in range(len(centroids)):
    SSE=[]
    
    for k in range(len(labels)):
        if i==labels[k]:
            SSE.append(sum((mat2[k]-centroids[i])**2))         
    actual_SSE.append(sum(SSE))
    SSE1=SSE1 + sum(SSE)            
    
SSE2=0
actual_SSE2=[]
for i in range(len(centroids2)):
    SSE=[]
    
    for k in range(len(labels2)):
        if i==labels[k]:
            SSE.append(sum((mat2[k]-centroids2[i])**2))         
    actual_SSE2.append(sum(SSE))
    SSE2=SSE2 + sum(SSE)     
    
    
#start Entropy
# find the cluster matrix
def clusterfinder(data):
    
    clust=[]
    c=Counter(c0=0,c1=0,c2=0,c3=0,c4=0,c5=0,c6=0) #needs to be manually updated if changing bins
    for s in range(len(label_final2)):
        if labels[s] == data: 
            if label_final2[s] == 0:
                c.update({'c0':1})
        
            if label_final2[s] == 1:
                c.update({'c1':1})
        
            if label_final2[s] == 2:
                c.update({'c2':1})
       
            if label_final2[s] == 3:
                c.update({'c3':1})
            if label_final2[s] == 4:
                c.update({'c4':1})
        
            if label_final2[s] == 5:
                c.update({'c5':1})
        
            if label_final2[s] == 6:
                c.update({'c6':1})
    clust.append(c['c0'])
    clust.append(c['c1'])
    clust.append(c['c2'])
    clust.append(c['c3'])
    clust.append(c['c4'])
    clust.append(c['c5'])
    clust.append(c['c6'])
    return clust

P=[]
for o in range(bin_no):
    P.append(clusterfinder(o))

P=np.array(P)
E=0
Entropy=[]
Purity=[]
for i in range(bin_no):
    for j in range(bin_no):
      if P[i,j] !=0:
        E=E+ -((P[i,j])/sum(P[i])) * np.log10(P[i,j]/sum(P[i]))/math.log10(2)
    Entropy.append(E)
    Purity.append(max(P[i,:])/sum(P[i,:]))

Entropy=np.array(Entropy)
Purity= np.array(Purity)

TotalP = sum(P) # add all elements row wise then add all elements of resulting row….which is essesntially the whole set
WholeEntropy = 0
WholePurity = 0
for x in range(len(Entropy)):
    WholeEntropy = WholeEntropy + ((sum(P[x,:]))/(sum(TotalP)))*Entropy[x]
for i in range(bin_no):
    
    
    WholePurity = WholePurity + ((sum(P[i,:]))/(sum(TotalP)))*Purity[i]

def clusterfinder(data):
    
    clust=[]
    c=Counter(c0=0,c1=0,c2=0,c3=0,c4=0,c5=0,c6=0) #needs to be manually updated if changing bins
    for s in range(len(label_final2)):
        if labels2[s] == data: 
            if label_final2[s] == 0:
                c.update({'c0':1})
        
            if label_final2[s] == 1:
                c.update({'c1':1})
        
            if label_final2[s] == 2:
                c.update({'c2':1})
       
            if label_final2[s] == 3:
                c.update({'c3':1})
            if label_final2[s] == 4:
                c.update({'c4':1})
        
            if label_final2[s] == 5:
                c.update({'c5':1})
        
            if label_final2[s] == 6:
                c.update({'c6':1})
    clust.append(c['c0'])
    clust.append(c['c1'])
    clust.append(c['c2'])
    clust.append(c['c3'])
    clust.append(c['c4'])
    clust.append(c['c5'])
    clust.append(c['c6'])
    return clust

P2=[]
for o in range(bin_no):
    P2.append(clusterfinder(o))

P2=np.array(P2)
E2=0
Entropy2=[]
Purity2=[]
for i in range(bin_no):
    for j in range(bin_no):
      if P2[i,j] !=0:
        E2=E2+ -((P2[i,j])/sum(P2[i])) * np.log10(P2[i,j]/sum(P2[i]))/math.log10(2)
    Entropy2.append(E2)
    Purity2.append(max(P2[i,:])/sum(P2[i,:]))

Entropy2=np.array(Entropy2)
Purity2= np.array(Purity2)

TotalP2 = sum(P2) # add all elements row wise then add all elements of resulting row….which is essesntially the whole set
WholeEntropy2 = 0
WholePurity2 = 0
for x in range(len(Entropy2)):
    WholeEntropy2 = WholeEntropy2 + ((sum(P2[x,:]))/(sum(TotalP2)))*Entropy2[x]
for i in range(bin_no):
    
    
    WholePurity2 = WholePurity2 + ((sum(P2[i,:]))/(sum(TotalP2)))*Purity2[i]

import csv
with open('Results3.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["SSE for K Means","SSE for DBScan","Entropy for K Means","Entropy for DBscan","Purity for K Means","Purity for DBscan"])
    writer.writerow([SSE1,SSE2,WholeEntropy,WholeEntropy2,WholePurity,WholePurity2])
    