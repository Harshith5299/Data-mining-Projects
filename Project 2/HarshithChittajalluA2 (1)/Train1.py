# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd


df = pd.read_csv (r'C:\Users\harsh\Downloads\CGMData.csv')
df2 = pd.read_csv (r'C:\Users\harsh\Downloads\InsulinData.csv')
df3 = pd.read_csv (r'C:\Users\harsh\Downloads\CGMData670GPatient3.csv')
df4 = pd.read_csv (r'C:\Users\harsh\Downloads\InsulinAndMealIntake670GPatient3.csv')

df['Sensor Glucose (mg/dL)'] = df['Sensor Glucose (mg/dL)'].interpolate(method='polynomial', order=2)
df3['Sensor Glucose (mg/dL)'] = df3['Sensor Glucose (mg/dL)'].interpolate(method='polynomial', order=2)

df2.loc[:, "Datetime"] = pd.to_datetime(df2.loc[:, "Date"]+" "+ df2.loc[:, "Time"])
df3.loc[:, "Datetime"] = pd.to_datetime(df3.loc[:, "Date"]+" "+ df3.loc[:, "Time"])
df4.loc[:, "Datetime"] = pd.to_datetime(df4.loc[:, "Date"]+" "+ df4.loc[:, "Time"])
df.loc[:, "Datetime"] = pd.to_datetime(df.loc[:, "Date"]+" "+ df.loc[:, "Time"])

t=df2[df2['BWZ Carb Input (grams)'].notnull()]
t0 = t[t['BWZ Carb Input (grams)'] != 0]

t1=df4[df4['BWZ Carb Input (grams)'].notnull()]
t02 = t1[t1['BWZ Carb Input (grams)'] != 0]

timestamps_insulin = t0.iloc[:,47]  #havent considred subject 2
ti= timestamps_insulin.to_list()

timestamps_insulin2 = t02.iloc[:,47]  #subject 2
ti2= timestamps_insulin2.to_list()

timestamps_CGM = df.iloc[:,47]
ti_cgm=timestamps_CGM.to_list()

timestamps_CGM2 = df3.iloc[:,47]
ti_cgm2=timestamps_CGM2.to_list()

i = np.argmin(np.abs(df.iloc[:,47] - t0.iloc[746,47]))
i2 = np.argmin(np.abs(df3.iloc[:,47] - t02.iloc[423,47]))
#threshold: closest datapoint should not be more than 10 mins = 600 sec
e=df.iloc[i,47]-t0.iloc[746,47]
e2=df3.iloc[i2,47]-t02.iloc[423,47]
#now test code for valid CGM meal start times begins:
cgm_index=[] #store index values of cgm meal times
for j in range(len(t0)):
    i = np.argmin(np.abs(df.iloc[:,47] - t0.iloc[j,47]))
    e=df.iloc[i,47]-t0.iloc[j,47]
    if e.total_seconds() <= 400:
        cgm_index.append(i)
    j-=1

cgm_index2=[] #store index values of cgm meal times
for j2 in range(len(t02)):
    i2 = np.argmin(np.abs(df3.iloc[:,47] - t02.iloc[j2,47]))
    e2=df3.iloc[i2,47]-t02.iloc[j2,47]
    if e2.total_seconds() <= 600:
        cgm_index2.append(i2)
    j2-=1
    
#cgm meal data timestamps
cgm_meal_timestamps= df.iloc[cgm_index[:],47]

cgm_meal_timestamps2=df3.iloc[cgm_index2[:],47]

#start megaloop for meal data extraction
meal_data = [] 
label=[]
meal_index=len(cgm_index)-1
temp=meal_index
c=len(df)-1
while c>0:
    meal_sample=[]
    if meal_index >=0:
        if df.iloc[cgm_index[meal_index],47] < df.iloc[c,47] + pd.Timedelta(hours=2):  #meal present
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
            df.iloc[cgm_index[meal_index],47] > df.iloc[c,47] + pd.Timedelta(hours=2) #ideal no meal case 
            mask_df = (df['Datetime'] > df.iloc[c,47]) & (df['Datetime'] <= df.iloc[c,47]+ pd.Timedelta(hours=2))
            seg=df.loc[mask_df]
            meal_sample= seg['Sensor Glucose (mg/dL)'].tolist()
            meal_sample.reverse()
            meal_data.append(meal_sample)
            label.append(0)
            c= np.argmin(np.abs(df.iloc[:,47] - (df.iloc[c,47]+pd.Timedelta(hours=2))))
    else:
        mask_df = (df['Datetime'] > df.iloc[c,47]) & (df['Datetime'] <= df.iloc[c,47]+ pd.Timedelta(hours=2))
        seg=df.loc[mask_df]
        meal_sample= seg['Sensor Glucose (mg/dL)'].tolist()
        meal_sample.reverse()
        meal_data.append(meal_sample)
        label.append(0)
        c=c-24
#megaloop for combined meal-no meal extraction
meal_data2 = [] 
label2=[]
meal_index2=len(cgm_index2)-1

c2=len(df3)-1
while c2>0:
    
    meal_sample2=[]
    if meal_index2 >=0:
        if df3.iloc[cgm_index2[meal_index2],47] < df3.iloc[c2,47] + pd.Timedelta(hours=2):  #meal present
            if df3.iloc[cgm_index2[meal_index2-1],47] < df3.iloc[cgm_index2[meal_index2],47]+ pd.Timedelta(hours=2): #meal case2 ignore current meal index and enter next meal index
                mask_df2 = (df3['Datetime'] > df3.iloc[cgm_index2[meal_index2-1],47]-pd.Timedelta(hours=0.5) ) & (df3['Datetime'] <= df3.iloc[cgm_index2[meal_index2-1],47]+ pd.Timedelta(hours=2))
                seg2=df3.loc[mask_df2]
                meal_sample2= seg2['Sensor Glucose (mg/dL)'].tolist()
                meal_sample2.reverse()
                meal_data2.append(meal_sample2) # add the cgm value of the meal intake set ~ 24 values
                label2.append(1)
                c2= cgm_index2[meal_index2-1]-24
                meal_index2=meal_index2-2 # go to next next meal index
            
            elif df3.iloc[cgm_index2[meal_index2-1],47] == df3.iloc[cgm_index2[meal_index2],47]+ pd.Timedelta(hours=2): #meal case 3
                mask_df2 = (df3['Datetime'] > df3.iloc[cgm_index2[meal_index2],47]+pd.Timedelta(hours=1.5) ) & (df3['Datetime'] <= df3.iloc[cgm_index2[meal_index2],47]+ pd.Timedelta(hours=4))
                seg2=df3.loc[mask_df2]
                meal_sample2= seg['Sensor Glucose (mg/dL)'].tolist()
                meal_sample2.reverse()
                meal_data2.append(meal_sample2) 
                label2.append(1)
                
                c2= cgm_index2[meal_index2-1]-24
                meal_index2=meal_index2-2
            else:
                mask_df2 = (df3['Datetime'] > df3.iloc[cgm_index2[meal_index2],47]-pd.Timedelta(hours=0.5) ) & (df3['Datetime'] <= df3.iloc[cgm_index2[meal_index2],47]+ pd.Timedelta(hours=2))
                seg2=df3.loc[mask_df2]
                meal_sample2= seg2['Sensor Glucose (mg/dL)'].tolist()
                meal_sample2.reverse()
                meal_data2.append(meal_sample2)
                label2.append(1)
                meal_index2=meal_index2-1
                c2= cgm_index2[meal_index2]-24
            
        else: 
            df3.iloc[cgm_index2[meal_index2],47] > df3.iloc[c2,47] + pd.Timedelta(hours=2) #ideal no meal case 
            mask_df2 = (df3['Datetime'] > df3.iloc[c2,47]) & (df3['Datetime'] <= df3.iloc[c2,47]+ pd.Timedelta(hours=2))
            seg2=df3.loc[mask_df2]
            meal_sample2= seg2['Sensor Glucose (mg/dL)'].tolist()
            meal_sample2.reverse()
            meal_data2.append(meal_sample2)
            label2.append(0)
            c2= np.argmin(np.abs(df3.iloc[:,47] - (df3.iloc[c2,47]+pd.Timedelta(hours=2))))
    else:
        mask_df2 = (df3['Datetime'] > df3.iloc[c2,47]) & (df3['Datetime'] <= df3.iloc[c2,47]+ pd.Timedelta(hours=2))
        seg2=df3.loc[mask_df2]
        meal_sample2= seg2['Sensor Glucose (mg/dL)'].tolist()
        meal_sample2.reverse()
        meal_data2.append(meal_sample2)
        label2.append(0)
        c2=c2-24


premealpre=[]
nomealpre=[]
mealpre=[]
label_final = label + label2
for x in range(0, len(meal_data)):
    if label[x]==1:
        premealpre.append(meal_data[x])
    if label[x]==0:
        nomealpre.append(meal_data[x])
for y in range(0, len(meal_data2)):
    if label2[y]==1:
        premealpre.append(meal_data2[y])
    if label2[y]==0:
        nomealpre.append(meal_data2[y])

for z in range(0,len(premealpre)): #filtering out inefficient meal data
    if len(premealpre[z]) >=29:
        mealpre.append(premealpre[z])

# Saving the result in CSV file
import csv
with open('Results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Result"])
    for val in label_final:
        writer.writerow([val])

with open("Sample.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(meal_data)


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
    f3.append(np.var(mealpre[w]))
    f4.append(np.std(mealpre[w]))
    f5.append(kurtosis(mealpre[w]))
    
    f7.append(skew(mealpre[w]))
    f8.append(meanabs(mealpre[w]))
    f9.append(maxdev(mealpre[w]))
    

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
    
    
q1=[]
q2=[]
q3=[]
q4=[]
q5=[]

q7=[]
q8=[]
q9=[]

for x in range(0, len(nomealpre)):
    q1.append(norm(nomealpre[x]))
    q2.append(norm(nomealpre[x], inf))
    q3.append(np.var(nomealpre[x]))
    q4.append(np.std(nomealpre[x]))
    q5.append(kurtosis(nomealpre[x]))
    
    q7.append(skew(nomealpre[x]))
    q8.append(meanabs(nomealpre[x]))
    q9.append(maxdev(nomealpre[x]))
    

q1=np.array(q1)
q2=np.array(q2)
q3=np.array(q3)
q4=np.array(q4)
q5=np.array(q5)

q7=np.array(q7)
q8=np.array(q8)
q9=np.array(q9)    


q1.reshape(len(nomealpre),1)
q2.reshape(len(nomealpre),1)
q3.reshape(len(nomealpre),1)
q4.reshape(len(nomealpre),1)
q5.reshape(len(nomealpre),1)

q7.reshape(len(nomealpre),1)
q8.reshape(len(nomealpre),1)
q9.reshape(len(nomealpre),1)


label1=[1]*len(mealpre)
label0=[0]*len(nomealpre)


label_final2=label1+label0
label_final2=np.array(label_final2)
label_final2.reshape(-1,1)

d1= np.concatenate((f1,q1))
d2= np.concatenate((f2,q2))
d3= np.concatenate((f3,q3))
d4= np.concatenate((f4,q4))
d5= np.concatenate((f5,q5))

d7= np.concatenate((f7,q7))
d8= np.concatenate((f8,q8))
d9= np.concatenate((f9,q9))

label_final=label_final2.ravel()
training_data=np.column_stack((d1,d2,d3,d4,d5,d7,d8,d9,label_final2))
X=training_data
y=label_final2

from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.svm import SVC
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,recall_score, precision_score, f1_score, classification_report
from sklearn.ensemble import AdaBoostClassifier

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

def SupportVectorMachine(x_train,x_test,y_train,y_test):
    print("Support Vector Machine")
    
    svc=SVC(kernel='linear')
    svc.fit(x_train,y_train)
    y_pred = svc.predict(x_test)
    
    filename = open("svm.pkl", 'wb')
    pickle.dump(svc, filename)
    filename.close()
    print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))
    print ("Accuracy : ",accuracy_score(y_test,y_pred)*100)
    print("Report : ",classification_report(y_test, y_pred))
    
    return accuracy_score(y_test,y_pred)*100

#accuracy=SupportVectorMachine(x_train, x_test, y_train.values.ravel(), y_test.values.ravel())
#accuracy


def AdaBoost(x_train,x_test,y_train,y_test):
    abc = AdaBoostClassifier()
    abc.fit(x_train , y_train)
    y_pred = abc.predict(x_test)
    
    filename = open("adaboost.pkl", 'wb')
    pickle.dump(abc, filename)
    filename.close()
    
    print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))
    print ("Accuracy : ",accuracy_score(y_test,y_pred)*100)
    print("Report : ",classification_report(y_test, y_pred))
    
accuracy=AdaBoost(x_train,x_test,y_train.values.ravel(),y_test.values.ravel())
accuracy    


