#!/usr/bin/env python
# coding: utf-8

# In[382]:


import pandas as pd
import copy

df = pd.read_csv (r'C:\Users\Harsh\Downloads\CGMData.csv')
df2 = pd.read_csv (r'C:\Users\Harsh\Downloads\InsulinData.csv')


# In[383]:


df.iloc[40,2]


# In[384]:


k = df2.Alarm[df2.Alarm == 'AUTO MODE ACTIVE PLGM OFF'].index.tolist()
i = df2.index.get_loc(k[1])


# In[385]:


po = df2.iloc[k[1]]
print(po)
date_value = po["Date"]
time_value = po["Time"]


# In[386]:


# seg= df2[["Date","Time"]]
# print(seg)


# In[387]:


# seg2=seg[["Date"=="8/9/2017"]]


# In[388]:


# df2.iloc[k[0]]
# df["datetime"] = pd.to_datetime(df["Date"]+" "+df["Time"])


# In[389]:


# df["datetime"]


# In[390]:


seg2= df.Date[df.Date == date_value].index.tolist()  # Making a list of all the rows that occured on the day auto mode started
print(seg2)


# In[391]:


kk=df.iloc[seg2] # making a dataframe from all the rows in which day the auto mode started
print(kk)


# In[392]:


i2 = kk.Time[kk.Time >= time_value].index.tolist()          #Making a list of all the rows that were greater than or equal to the the time where auto mode occurs
print(i2)
i_use = kk.index.get_loc(i2[-1])          #the closest time value in CGM to the auto mode activation day   


# In[393]:


kk.iloc[i_use]


# In[394]:


k3 = df2.Alarm[df2.Alarm == 'AUTO MODE ACTIVE PLGM OFF'].index.tolist()
i3 = df2.index.get_loc(k[0])


# In[395]:


i3


# In[396]:


po2 = df2.iloc[k[0]]
print(po2)
date_value = po2["Date"]
time_value = po2["Time"]
seg= df.Date[df.Date == date_value].index.tolist()
kk2=df.iloc[seg]
i4 = kk2.Time[kk2.Time >= time_value].index.tolist()
print(i4)
i_use2 = kk2.index.get_loc(i4[-1])


# In[397]:


kk2.iloc[i_use2]


# In[398]:


fill= df.xs('Sensor Glucose (mg/dL)',axis = 1 )


# In[399]:


fill


# In[400]:


s=pd.Series(fill)
s


# In[401]:


s.interpolate() #linear interpolation


# In[402]:


p=s.interpolate(method='polynomial', order=2) # polynomial interpolation
p


# In[403]:


df.replace(fill,p)


# In[404]:


df


# In[405]:


df.xs('Sensor Glucose (mg/dL)',axis = 1 )


# In[406]:


df['Sensor Glucose (mg/dL)'] = df['Sensor Glucose (mg/dL)'].interpolate(method='polynomial', order=2)


# In[407]:


index=df.index
no_of_rows=len(index)
i=no_of_rows
average = 288

CGMmore180 =0
CGMmore250=0
CGMbw70to180=0
CGMbw70to150=0
CGMless70=0
CGMless54=0
AutoCGMmore180 =0
AutoCGMmore250=0
AutoCGMbw70to180=0
AutoCGMbw70to150=0
AutoCGMless70=0
AutoCGMless54=0


# In[ ]:





# In[408]:


df['Sensor Glucose (mg/dL)'][0]


# In[409]:


timedf=(pd.to_timedelta(df['Time'].str.strip()))


# In[410]:


df_lol = copy.deepcopy(df)
df["Time"] = pd.to_datetime(df["Time"])
time2=(df['Time'].dt.hour >= 0) &             (df['Time'].dt.hour <= 5)


# In[411]:


df_lol


# In[412]:


df


# In[413]:


overnight=df[time2]


# In[414]:


df.iloc[40,1]


# In[415]:


df['Date'] = pd.to_datetime(df['Date'])    
uni=df.Date.dt.strftime('%#m/%#d/%Y').unique()
uni2= uni.tolist()    

     


# In[416]:


autoloc=df.loc[df['Index'] ==  kk.iloc[i_use][0]]
autoloc.iloc[0][0]


# In[417]:


Auto = df_lol.iloc[:autoloc.iloc[0][0],:] 
manual = df_lol.iloc[autoloc.iloc[0][0]+1:,:]


# In[418]:


Auto["Time"] = pd.to_datetime(Auto["Time"])                   
Aovernight=(Auto['Time'].dt.hour >= 0) &             (Auto['Time'].dt.hour <= 5)
manual["Time"] = pd.to_datetime(manual["Time"])               #creating a sub dataframe with timestamps between 00:00:00 and 06:00:00
Movernight=(manual['Time'].dt.hour >= 0) &             (manual['Time'].dt.hour <= 5)

Anight=Auto[Aovernight]
Mnight=manual[Movernight]


# In[419]:


Auto["Time"] = pd.to_datetime(Auto["Time"])                   
Amorning=(Auto['Time'].dt.hour >= 6) &             (Auto['Time'].dt.hour <= 23)
manual["Time"] = pd.to_datetime(manual["Time"])               #creating a sub dataframe with timestamps between 06:00:00 and 00:00:00          

Mmorning=(manual['Time'].dt.hour >= 6) &             (manual['Time'].dt.hour <= 23)
Aday=Auto[Amorning]
Mday=manual[Mmorning]


# In[420]:


Mday


# In[421]:


j=len(uni2)-1                  # oVERNIGHT MANUAL
count=0
no_of_days=1
percentage1=0
percentage2=0
percentage3=0
percentage4=0
percentage5=0
percentage6=0
iterations=0
CGMmore180 =0
CGMmore250=0
CGMbw70to180=0
CGMbw70to150=0
CGMless70=0
CGMless54=0
for i in range(len(Mnight)-1,0,-1):
    
    
    if Mnight.iloc[i,1]==uni2[j]:
        
        count=count+1
        if count>72:
            continue
        if Mnight.iloc[i,30]>180:
                CGMmore180 =CGMmore180 + 1
        if Mnight.iloc[i,30]>250:
                CGMmore250 = CGMmore250 + 1
        if Mnight.iloc[i,30]>70 and Mnight.iloc[i,30] < 180:
                CGMbw70to180 = CGMbw70to180 + 1
        if Mnight.iloc[i,30]>70 and Mnight.iloc[i,30] < 150:
                CGMbw70to150=CGMbw70to150+1
        if Mnight.iloc[i,30]<70:
                CGMless70=CGMless70+1
        if Mnight.iloc[i,30]<54:
                CGMless54=CGMless54 + 1
        
    
        
    
    else:
            
            
                if count<52:
                    j=j-1
                    count=0
                    CGMmore180 =0
                    CGMmore250=0
                    CGMbw70to180=0
                    CGMbw70to150=0
                    CGMless70=0
                    CGMless54=0
                    continue
                percentage1=percentage1+CGMmore180/288
                percentage2=percentage2+CGMmore250/288
                percentage3=percentage3+CGMbw70to180/288
                percentage4=percentage4+CGMbw70to150/288
                percentage5=percentage5+CGMless70/288
                percentage6=percentage6+CGMless54/288
                no_of_days=no_of_days+1
                count = 0
                j=j-1
                CGMmore180 =0
                CGMmore250=0
                CGMbw70to180=0
                CGMbw70to150=0
                CGMless70=0
                CGMless54=0
    
    
    
fm1=(percentage1/no_of_days)*100
fm2=(percentage2/no_of_days)*100
fm3=(percentage3/no_of_days)*100
fm4=(percentage4/no_of_days)*100
fm5=(percentage5/no_of_days)*100
fm6=(percentage6/no_of_days)*100
print(fm1,fm2,fm3,fm4,fm5,fm6)


# In[ ]:





# In[422]:


j=len(uni2)-1                  # oVERNIGHT aUTO
count=0
no_of_days=1
percentage1=0
percentage2=0
percentage3=0
percentage4=0
percentage5=0
percentage6=0
iterations=0
AutoCGMmore180 =0
AutoCGMmore250=0
AutoCGMbw70to180=0
AutoCGMbw70to150=0
AutoCGMless70=0
AutoCGMless54=0
for i in range(len(Anight)-1,0,-1):
    
    
    if Anight.iloc[i,1]==uni2[j]:
        
        count=count+1
        if count>72:
            continue
        if Anight.iloc[i,30]>180:
                AutoCGMmore180 =AutoCGMmore180 + 1
        if Anight.iloc[i,30]>250:
                AutoCGMmore250 = AutoCGMmore250 + 1
        if Anight.iloc[i,30]>70 and Anight.iloc[i,30] < 180:
                AutoCGMbw70to180 = AutoCGMbw70to180 + 1
        if Anight.iloc[i,30]>70 and Anight.iloc[i,30] < 150:
                AutoCGMbw70to150=AutoCGMbw70to150+1
        if Anight.iloc[i,30]<70:
                AutoCGMless70=AutoCGMless70+1
        if Anight.iloc[i,30]<54:
                AutoCGMless54=AutoCGMless54 + 1
        
    
        
    
    else:
            
            
                if count<52:
                    j=j-1
                    count=0
                    AutoCGMmore180 =0
                    AutoCGMmore250=0
                    AutoCGMbw70to180=0
                    AutoCGMbw70to150=0
                    AutoCGMless70=0
                    AutoCGMless54=0
                    continue
                percentage1=percentage1+AutoCGMmore180/288
                percentage2=percentage2+AutoCGMmore250/288
                percentage3=percentage3+AutoCGMbw70to180/288
                percentage4=percentage4+AutoCGMbw70to150/288
                percentage5=percentage5+AutoCGMless70/288
                percentage6=percentage6+AutoCGMless54/288
                no_of_days=no_of_days+1
                count = 0
                j=j-1
                AutoCGMmore180 =0
                AutoCGMmore250=0
                AutoCGMbw70to180=0
                AutoCGMbw70to150=0
                AutoCGMless70=0
                AutoCGMless54=0
    
    
    
fa1=(percentage1/no_of_days)*100
fa2=(percentage2/no_of_days)*100
fa3=(percentage3/no_of_days)*100
fa4=(percentage4/no_of_days)*100
fa5=(percentage5/no_of_days)*100
fa6=(percentage6/no_of_days)*100
print(fa1,fa2,fa3,fa4,fa5,fa6)


# In[423]:


len(uni2)


# In[424]:


kk.iloc[i_use][1]


# In[425]:


overnight.iloc[12850,1]


# In[426]:


fm2


# In[427]:


fm3


# In[428]:


fm4


# In[429]:


j=len(uni2)-1                          #MORNING AUTO
count=0
no_of_days=1
percentage1=0
percentage2=0
percentage3=0
percentage4=0
percentage5=0
percentage6=0
iterations=0
AutoCGMmore180 =0
AutoCGMmore250=0
AutoCGMbw70to180=0
AutoCGMbw70to150=0
AutoCGMless70=0
AutoCGMless54=0
for i in range(len(Aday)-1,0,-1):
    
    
    if Aday.iloc[i,1]==uni2[j]:
        
        count=count+1
        if count>216:
            continue
        if Aday.iloc[i,30]>180:
                AutoCGMmore180 =AutoCGMmore180 + 1
        if Aday.iloc[i,30]>250:
                AutoCGMmore250 = AutoCGMmore250 + 1
        if Aday.iloc[i,30]>70 and Aday.iloc[i,30] < 180:
                AutoCGMbw70to180 = AutoCGMbw70to180 + 1
        if Aday.iloc[i,30]>70 and Aday.iloc[i,30] < 150:
                AutoCGMbw70to150=AutoCGMbw70to150+1
        if Aday.iloc[i,30]<70:
                AutoCGMless70=AutoCGMless70+1
        if Aday.iloc[i,30]<54:
                AutoCGMless54=AutoCGMless54 + 1
        
    
        
    
    else:
            
            
                if count<186:
                    j=j-1
                    count=0
                    CGMmore180 =0
                    CGMmore250=0
                    CGMbw70to180=0
                    CGMbw70to150=0
                    CGMless70=0
                    CGMless54=0
                    continue
                percentage1=percentage1+AutoCGMmore180/288
                percentage2=percentage2+AutoCGMmore250/288
                percentage3=percentage3+AutoCGMbw70to180/288
                percentage4=percentage4+AutoCGMbw70to150/288
                percentage5=percentage5+AutoCGMless70/288
                percentage6=percentage6+AutoCGMless54/288
                no_of_days=no_of_days+1
                count = 0
                j=j-1
                AutoCGMmore180 =0
                AutoCGMmore250=0
                AutoCGMbw70to180=0
                AutoCGMbw70to150=0
                AutoCGMless70=0
                AutoCGMless54=0
    
    
    
fa1day=(percentage1/no_of_days)*100
fa2day=(percentage2/no_of_days)*100
fa3day=(percentage3/no_of_days)*100
fa4day=(percentage4/no_of_days)*100
fa5day=(percentage5/no_of_days)*100
fa6day=(percentage6/no_of_days)*100
print(fa1day,fa2day,fa3day,fa4day,fa5day,fa6day)


# In[430]:


#fa2


# In[ ]:





# In[431]:


#print(kk.iloc[i_use][1]>overnight.iloc[10000,1])


# In[432]:


#overnight.iloc[10000,1]


# In[433]:


#kk.iloc[i_use][1]


# In[ ]:





# In[434]:


#Autofm5


# In[435]:


#Autofm6


# In[436]:


#percentage1


# In[437]:


#no_of_days


# In[438]:


#df['Date']


# In[439]:


#df['Date'][1]


# In[440]:


j=len(uni2)-1                          #MORNING Manual
count=0
no_of_days=1
percentage1=0
percentage2=0
percentage3=0
percentage4=0
percentage5=0
percentage6=0
iterations=0
CGMmore180 =0
CGMmore250=0
CGMbw70to180=0
CGMbw70to150=0
CGMless70=0
CGMless54=0
for i in range(len(Mday)-1,0,-1):
    
    
    if Mday.iloc[i,1]==uni2[j]:
        
        count=count+1
        if count>216:
            continue
        if Mday.iloc[i,30]>180:
                CGMmore180 =CGMmore180 + 1
        if Mday.iloc[i,30]>250:
                CGMmore250 = CGMmore250 + 1
        if Mday.iloc[i,30]>70 and Mday.iloc[i,30] < 180:
                CGMbw70to180 = CGMbw70to180 + 1
        if Mday.iloc[i,30]>70 and Mday.iloc[i,30] < 150:
                CGMbw70to150=CGMbw70to150+1
        if Mday.iloc[i,30]<70:
                CGMless70=CGMless70+1
        if Mday.iloc[i,30]<54:
                CGMless54=CGMless54 + 1
        
    
        
    
    else:
            
            
                if count<186:
                    j=j-1
                    count=0
                    CGMmore180 =0
                    CGMmore250=0
                    CGMbw70to180=0
                    CGMbw70to150=0
                    CGMless70=0
                    CGMless54=0
                    continue
                percentage1=percentage1+CGMmore180/288
                percentage2=percentage2+CGMmore250/288
                percentage3=percentage3+CGMbw70to180/288
                percentage4=percentage4+CGMbw70to150/288
                percentage5=percentage5+CGMless70/288
                percentage6=percentage6+CGMless54/288
                no_of_days=no_of_days+1
                count = 0
                j=j-1
                CGMmore180 =0
                CGMmore250=0
                CGMbw70to180=0
                CGMbw70to150=0
                CGMless70=0
                CGMless54=0
    
    
    
fm1day=(percentage1/no_of_days)*100
fm2day=(percentage2/no_of_days)*100
fm3day=(percentage3/no_of_days)*100
fm4day=(percentage4/no_of_days)*100
fm5day=(percentage5/no_of_days)*100
fm6day=(percentage6/no_of_days)*100
print(fm1day,fm2day,fm3day,fm4day,fm5day,fm6day)


# In[444]:


j=len(uni2)-1                          # Full day Manual
count=0
no_of_days=1
percentage1=0
percentage2=0
percentage3=0
percentage4=0
percentage5=0
percentage6=0
iterations=0
CGMmore180 =0
CGMmore250=0
CGMbw70to180=0
CGMbw70to150=0
CGMless70=0
CGMless54=0
for i in range(len(manual)-1,0,-1):
    
    
    if manual.iloc[i,1]==uni2[j]:
        
        count=count+1
        if count>288:
            continue
        if manual.iloc[i,30]>180:
                CGMmore180 =CGMmore180 + 1
        if manual.iloc[i,30]>250:
                CGMmore250 = CGMmore250 + 1
        if manual.iloc[i,30]>70 and manual.iloc[i,30] < 180:
                CGMbw70to180 = CGMbw70to180 + 1
        if manual.iloc[i,30]>70 and manual.iloc[i,30] < 150:
                CGMbw70to150=CGMbw70to150+1
        if manual.iloc[i,30]<70:
                CGMless70=CGMless70+1
        if manual.iloc[i,30]<54:
                CGMless54=CGMless54 + 1
        
    
        
    
    else:
            
            
                if count<258:
                    j=j-1
                    count=0
                    CGMmore180 =0
                    CGMmore250=0
                    CGMbw70to180=0
                    CGMbw70to150=0
                    CGMless70=0
                    CGMless54=0
                    continue
                percentage1=percentage1+CGMmore180/288
                percentage2=percentage2+CGMmore250/288
                percentage3=percentage3+CGMbw70to180/288
                percentage4=percentage4+CGMbw70to150/288
                percentage5=percentage5+CGMless70/288
                percentage6=percentage6+CGMless54/288
                no_of_days=no_of_days+1
                count = 0
                j=j-1
                CGMmore180 =0
                CGMmore250=0
                CGMbw70to180=0
                CGMbw70to150=0
                CGMless70=0
                CGMless54=0
    
    
    
fm1full=(percentage1/no_of_days)*100
fm2full=(percentage2/no_of_days)*100
fm3full=(percentage3/no_of_days)*100
fm4full=(percentage4/no_of_days)*100
fm5full=(percentage5/no_of_days)*100
fm6full=(percentage6/no_of_days)*100
print(fm1full,fm2full,fm3full,fm4full,fm5full,fm6full)


# In[445]:


j=len(uni2)-1                          # full day auto 
count=0
no_of_days=1
percentage1=0
percentage2=0
percentage3=0
percentage4=0
percentage5=0
percentage6=0
iterations=0
AutoCGMmore180 =0
AutoCGMmore250=0
AutoCGMbw70to180=0
AutoCGMbw70to150=0
AutoCGMless70=0
AutoCGMless54=0
for i in range(len(Auto)-1,0,-1):
    
    
    if Auto.iloc[i,1]==uni2[j]:
        
        count=count+1
        if count>288:
            continue
        if Auto.iloc[i,30]>180:
                AutoCGMmore180 =AutoCGMmore180 + 1
        if Auto.iloc[i,30]>250:
                AutoCGMmore250 = AutoCGMmore250 + 1
        if Auto.iloc[i,30]>70 and Auto.iloc[i,30] < 180:
                AutoCGMbw70to180 = AutoCGMbw70to180 + 1
        if Auto.iloc[i,30]>70 and Auto.iloc[i,30] < 150:
                AutoCGMbw70to150=AutoCGMbw70to150+1
        if Auto.iloc[i,30]<70:
                AutoCGMless70=AutoCGMless70+1
        if Auto.iloc[i,30]<54:
                AutoCGMless54=AutoCGMless54 + 1
        
    
        
    
    else:
            
            
                if count<258:
                    j=j-1
                    count=0
                    AutoCGMmore180 =0
                    AutoCGMmore250=0
                    AutoCGMbw70to180=0
                    AutoCGMbw70to150=0
                    AutoCGMless70=0
                    AutoCGMless54=0
                    continue
                percentage1=percentage1+AutoCGMmore180/288
                percentage2=percentage2+AutoCGMmore250/288
                percentage3=percentage3+AutoCGMbw70to180/288
                percentage4=percentage4+AutoCGMbw70to150/288
                percentage5=percentage5+AutoCGMless70/288
                percentage6=percentage6+AutoCGMless54/288
                no_of_days=no_of_days+1
                count = 0
                j=j-1
                AutoCGMmore180 =0
                AutoCGMmore250=0
                AutoCGMbw70to180=0
                AutoCGMbw70to150=0
                AutoCGMless70=0
                AutoCGMless54=0
    
    
    
fa1full=(percentage1/no_of_days)*100
fa2full=(percentage2/no_of_days)*100
fa3full=(percentage3/no_of_days)*100
fa4full=(percentage4/no_of_days)*100
fa5full=(percentage5/no_of_days)*100
fa6full=(percentage6/no_of_days)*100
print(fa1full,fa2full,fa3full,fa4full,fa5full,fa6full)


# In[446]:


import csv
with open('Chittajallu_Results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Measure", "Daytime Percentage time in hyperglycemia (CGM > 180 mg/dL)", "Daytime percentage of time in hyperglycemia critical (CGM > 250 mg/dL)","Daytime percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)","Daytime percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)","Daytime percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)","Daytime percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)","Overnight Percentage time in hyperglycemia (CGM > 180 mg/dL)", "Overnight percentage of time in hyperglycemia critical (CGM > 250 mg/dL)","Overnight percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)","Overnight percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)","Overnight percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)","Overnight percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)","Whole Day Percentage time in hyperglycemia (CGM > 180 mg/dL)", "Whole Day percentage of time in hyperglycemia critical (CGM > 250 mg/dL)","Whole Day percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)","Whole Day percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)","Whole Day percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)","Whole Day percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)"])
    writer.writerow(["Auto",fa1day,fa2day,fa3day,fa4day,fa5day,fa6day,fa1,fa2,fa3,fa4,fa5,fa6,fa1full,fa2full,fa3full,fa4full,fa5full,fa6full])
    writer.writerow(["Manual",fm1day,fm2day,fm3day,fm4day,fm5day,fm6day,fm1,fm2,fm3,fm4,fm5,fm6,fm1full,fm2full,fm3full,fm4full,fm5full,fm6full])


# In[ ]:




