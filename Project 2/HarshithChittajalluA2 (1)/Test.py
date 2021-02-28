# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 00:36:01 2020

@author: harsh
"""
import pandas as pd
import numpy as np
from csv import reader
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.svm import SVC
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,recall_score, precision_score, f1_score, classification_report
from sklearn.ensemble import AdaBoostClassifier
from itertools import zip_longest
import csv



with open('Sample.csv', newline='') as f:
    reader = csv.reader(f)
    result = list(reader)

meal_data = []


# Converting the given raw meal data in CSV format with multiple columns and missing values into a floating list of lists
for sublist in result:
    float_sublist = []
    for x in sublist:
        float_sublist.append(float(x))
    meal_data.append(float_sublist)






# read csv file as a list of lists
#with open('Sample.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
   # csv_reader = reader(read_obj)
    # Pass reader object to list() to get a list of lists, let each list be a row of meal data in the csv file
   # meal_data = list(csv_reader)

#meal_data =np.array(meal_data).astype(np.float)
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

for w in range(0, len(meal_data)):
    f1.append(norm(meal_data[w]))
    f2.append(norm(meal_data[w], inf))
    f3.append(np.var(meal_data[w]))
    f4.append(np.std(meal_data[w]))
    f5.append(kurtosis(meal_data[w]))
    
    f7.append(skew(meal_data[w]))
    f8.append(meanabs(meal_data[w]))
    f9.append(maxdev(meal_data[w]))
    

f1=np.array(f1)
f2=np.array(f2)
f3=np.array(f3)
f4=np.array(f4)
f5=np.array(f5)

f7=np.array(f7)
f8=np.array(f8)
f9=np.array(f9)    


f1.reshape(len(f1),1)
f2.reshape(len(meal_data),1)
f3.reshape(len(meal_data),1)
f4.reshape(len(meal_data),1)
f5.reshape(len(meal_data),1)

f7.reshape(len(meal_data),1)
f8.reshape(len(meal_data),1)
f9.reshape(len(meal_data),1)
    
    
q1=[]
q2=[]
q3=[]
q4=[]
q5=[]

q7=[]
q8=[]
q9=[]

for x in range(0, len(meal_data)):
    q1.append(norm(meal_data[x]))
    q2.append(norm(meal_data[x], inf))
    q3.append(np.var(meal_data[x]))
    q4.append(np.std(meal_data[x]))
    q5.append(kurtosis(meal_data[x]))
    
    q7.append(skew(meal_data[x]))
    q8.append(meanabs(meal_data[x]))
    q9.append(maxdev(meal_data[x]))
    

q1=np.array(q1)
q2=np.array(q2)
q3=np.array(q3)
q4=np.array(q4)
q5=np.array(q5)

q7=np.array(q7)
q8=np.array(q8)
q9=np.array(q9)    


q1.reshape(len(meal_data),1)
q2.reshape(len(meal_data),1)
q3.reshape(len(meal_data),1)
q4.reshape(len(meal_data),1)
q5.reshape(len(meal_data),1)

q7.reshape(len(meal_data),1)
q8.reshape(len(meal_data),1)
q9.reshape(len(meal_data),1)




d1= np.concatenate((f1,q1))
d2= np.concatenate((f2,q2))
d3= np.concatenate((f3,q3))
d4= np.concatenate((f4,q4))
d5= np.concatenate((f5,q5))

d7= np.concatenate((f7,q7))
d8= np.concatenate((f8,q8))
d9= np.concatenate((f9,q9))


testing_data=np.column_stack((d1,d2,d3,d4,d5,d7,d8,d9))


adaboost_model =pickle.load(open("adaboost.pkl","rb"))

y_pred = adaboost_model.predict(testing_data)
# Save the predicted labels in 
pd.DataFrame(y_pred).to_csv("Prediction.csv",header=None, index=None)