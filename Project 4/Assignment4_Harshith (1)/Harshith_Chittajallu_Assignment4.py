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
df = pd.read_csv (r'CGMData.csv') # change the directory to run code
df2 = pd.read_csv (r'InsulinData.csv')


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
# bin_label is a success.


timestamps_insulin = t0.iloc[:,47]  #havent considred subject 2
ti= timestamps_insulin.to_list()



timestamps_CGM = df.iloc[:,47]
ti_cgm=timestamps_CGM.to_list()



i = np.argmin(np.abs(df.iloc[:,47] - t0.iloc[746,47])) #code for finding the equivalent CGM index value for carb input

#threshold: closest datapoint should not be more than 10 mins = 600 sec
e=df.iloc[i,47]-t0.iloc[746,47]
bolus=[]
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
        bolus.append(t0.iloc[j,19])
    j-=1


    
#cgm meal data timestamps
cgm_meal_timestamps= df.iloc[cgm_index[:],47]

valid_bolus =[]
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
                valid_bolus.append(bolus[meal_index])
                c= np.argmin(np.abs(df.iloc[:,47] - (df.iloc[cgm_index[meal_index-1],47]+pd.Timedelta(hours=2)))) #get nearest value to next meal's no meal start
                meal_index=meal_index-2 # go to next next meal index
            
            elif df.iloc[cgm_index[meal_index-1],47] == df.iloc[cgm_index[meal_index],47]+ pd.Timedelta(hours=2): #meal case 3
                mask_df = (df['Datetime'] > df.iloc[cgm_index[meal_index],47]+pd.Timedelta(hours=1.5) ) & (df['Datetime'] <= df.iloc[cgm_index[meal_index],47]+ pd.Timedelta(hours=4))
                seg=df.loc[mask_df]
                meal_sample= seg['Sensor Glucose (mg/dL)'].tolist()
                meal_sample.reverse()
                meal_data.append(meal_sample)
                label.append(1)
                valid_bolus.append(bolus[meal_index])
                c= np.argmin(np.abs(df.iloc[:,47] - (df.iloc[cgm_index[meal_index-1],47]+pd.Timedelta(hours=2))))
                meal_index=meal_index-2
            else:
                mask_df = (df['Datetime'] > df.iloc[cgm_index[meal_index],47]-pd.Timedelta(hours=0.5) ) & (df['Datetime'] <= df.iloc[cgm_index[meal_index],47]+ pd.Timedelta(hours=2))
                seg=df.loc[mask_df]
                meal_sample= seg['Sensor Glucose (mg/dL)'].tolist()
                meal_sample.reverse()
                meal_data.append(meal_sample)
                label.append(1)
                valid_bolus.append(bolus[meal_index])
                meal_index=meal_index-1
                c= np.argmin(np.abs(df.iloc[:,47] - (df.iloc[cgm_index[meal_index],47]+pd.Timedelta(hours=2))))
           
        else:    
           c= np.argmin(np.abs(df.iloc[:,47] - (df.iloc[c,47]+pd.Timedelta(hours=2)))) 
    else:
        c=c-24
    


# warning pre_bin_label3 and the meal data is reversed because of the loop
premealpre= meal_data
nomealpre=[]
mealpre=[]
label_final = [] 


final_bolus=[]
for z in range(0,len(premealpre)): #filtering out inefficient meal data
    if len(premealpre[z]) ==30:
        mealpre.append(premealpre[z]) # 'mealpre' is the finalized meal cgm data set
        label_final.append(pre_bin_label3[z])
        final_bolus.append(valid_bolus[z])# extracting respective final bolus values



final_bolus=np.array(final_bolus)
final_bolus=np.round(final_bolus, decimals=0)
final_bolus=final_bolus.astype(int) # finding final bolus

mealpre=np.array(mealpre)
B_max=mealpre.max(1) #finding B_Max

B_meal=[]
for i in range(len(mealpre)):
    B_meal.append(mealpre[i,6])
B_meal=np.array(B_meal) # finding B_meal, this is the 7th column of the meal data matrix (as we start 30 mins prior to meal intake)
    



extractor=np.array(mealpre)
positive = np.abs(extractor)
cgm_max= np.max(extractor)
cgm_min=np.min(positive)

bin_no2 = math.ceil((cgm_max-cgm_min)/20)
# for finding no of bins

pre_bin_label2=[] 

for k in range(len(B_max)):
    for l in range(bin_no2):
        if B_max[k] >= (cgm_min + l*20) and B_max[k] <   (cgm_min + (l+1)*20): 
            pre_bin_label2.append(l)
           # putting B_max in bins


pre_bin_label3=[] 

for k in range(0,len(B_meal)):
    for l in range(0,bin_no2):
        if l==0:        
            if int(B_meal[k]) <   int((cgm_min + 20)):  #initial case 
                pre_bin_label3.append(0)
        elif (int(cgm_min) + l*20) <= int(B_meal[k]) <   int((cgm_min) + (l+1)*20): 
            pre_bin_label3.append(l) # putting B_meal in bins

pre_bin_label2=np.array(pre_bin_label2)
pre_bin_label3=np.array(pre_bin_label3)
matrix4 =np.column_stack((pre_bin_label2,pre_bin_label3,final_bolus)) # resulting rule matrix 

unique=[]
sub_unique=[]
for i in range(len(matrix4)):
    sub_unique.append(matrix4[i,0])
    sub_unique.append(matrix4[i,1]) 
    sub_unique.append(matrix4[i,2])         
    unique.append(sub_unique)
    sub_unique=[]   # finding all the unique rules

xyunique=[]
xysub_unique=[]
for i in range(len(matrix4)):
    xysub_unique.append(matrix4[i,0])
    xysub_unique.append(matrix4[i,1]) 
             
    xyunique.append(xysub_unique)
    xysub_unique=[]  # finding all the unique support values

output2 = []
for x in xyunique:
    if x not in output2:
        output2.append(x) #gives all the unique support values
        
freq_label2=[]
frequency_count2=0
for i in range(len(output2)):
    for k in range(len(xyunique)):
        if output2[i] == xyunique[k]:
            frequency_count2= frequency_count2+1
    freq_label2.append(frequency_count2)
    frequency_count2=0 # frequency of that particular combination of B_max and B_min (frequency of support values)

output = []
for x in unique:
    if x not in output:
        output.append(x) # gives all the unique rules

freq_label=[]
frequency_count=0
for i in range(len(output)):
    for k in range(len(unique)):
        if output[i] == unique[k]:
            frequency_count= frequency_count+1
    freq_label.append(frequency_count)
    frequency_count=0 # original rule with 3 elements, this gives each of their frequencies


most_freq_item=[]
most_freq_item_value=[]
for i in range(len(output)):
    if freq_label[i] >=2:
        most_freq_item.append(output[i])        
        most_freq_item_value.append(freq_label[i])    

most_frequent_matrix= np.column_stack((most_freq_item,most_freq_item_value)) #sort this

matrix5 =np.column_stack((output,freq_label)) # rule frequency matrix with for finding confidence
matrix6 =np.column_stack((output2,freq_label2)) # support frequency for finding confidence

#megaloop for finding confidence

temp1=[]
temp2=[]
confidence=[]
for i in range(len(matrix5)):
    for j in range(len(matrix6)):
        if matrix5[i,0] == matrix6[j,0] and matrix5[i,1] == matrix6[j,1]:
            confidence.append((matrix5[i,3]/matrix6[j,2]))

final_matrix =np.column_stack((output,confidence,freq_label))

# sort matrix 5 to get outputs according to descending order of frequency and final_matrix to get outputs according to decsending order of confidence
matrix5=matrix5[matrix5[:,3].argsort()[::-1]]
final_matrix=final_matrix[final_matrix[:,3].argsort()[::-1]]

# anomalous matrix
anomalous=[]
for i in range(len(final_matrix)):
    if final_matrix[i,3] < 0.15:
        anomalous.append(final_matrix[i])

anomalous=np.array(anomalous)




import csv
with open('Apriori1.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["B_Max","B_Meal","Bolus Values"])
    for i in range(len(matrix5)):
        writer.writerow([matrix5[i,0],matrix5[i,1],matrix5[i,2]])
with open('Apriori2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["B_Max","B_Meal","Bolus Values","Confidence","Frequency"])
    for i in range(len(final_matrix)):
        writer.writerow([final_matrix[i,0],final_matrix[i,1],final_matrix[i,2],final_matrix[i,3],final_matrix[i,4]])
with open('Apriori3.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["B_Max","B_Meal","Bolus Values","Confidence"])
    for i in range(len(anomalous)):
        writer.writerow([anomalous[i,0],anomalous[i,1],anomalous[i,2],anomalous[i,3]])       
        
    