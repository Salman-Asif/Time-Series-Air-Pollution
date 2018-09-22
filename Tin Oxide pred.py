# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 21:37:17 2018

@author: User
"""

import pandas as pd          
import numpy as np          # For mathematical calculations
import matplotlib.pyplot as plt  # For plotting graphs
from datetime import datetime    # To access datetime
from pandas import Series        # To work on series



dataframe = pd.read_csv("AirQualityUCI_new.csv",parse_dates=[['Date', 'Time']])

dataframe_original = dataframe.copy()

train = dataframe[['Date_Time','PT08.S1(CO)']]


#print(train.Date_Time)
train['year'] = train.Date_Time.dt.year
train['year']=train.Date_Time.dt.year 
train['month']=train.Date_Time.dt.month 
train['day']=train.Date_Time.dt.day
train['Hour']=train.Date_Time.dt.hour 


train.index = train['Date_Time']

ctrain = train.copy()

ctrain = ctrain[:][6:]

ctrain = ctrain.ix[:'2005-04-03 23:00:00']

for i in range(1,len(ctrain)):    
    if(ctrain['PT08.S1(CO)'][i] == -200):        
        ctrain['PT08.S1(CO)'][i]=ctrain['PT08.S1(CO)'][i-1]


plt.figure(figsize=(16,8))
plt.plot(ctrain['PT08.S1(CO)'], label='Tin Oxide Level')
plt.title('Time Series(hourly)')
plt.xlabel("Time")
plt.ylabel("Tin Oxide")
plt.legend(loc='best')



ctrain['day of week']=ctrain['Date_Time'].dt.dayofweek
temp = ctrain['Date_Time']



#WHAT IS ROW???------------WEEKEND/WEEKDAY FUNCTION
def applyer(row):
    if row.dayofweek == 5 or row.dayofweek == 6:
        return 1
    else:
        return 0
    
#UNDERSTAND APPLY FUNCTION IN DATETIME
temp2 = ctrain['Date_Time'].apply(applyer)
ctrain['weekend']=temp2


#-----ALL PLOTS START HERE-----
#by year
ctrain.groupby('year')['PT08.S1(CO)'].mean().plot.bar() #not very good as 2004 entries are much more than 2005

#by month but diffrent years
ctrain.groupby('month')['PT08.S1(CO)'].mean().plot.bar(color='#3075ec')

#BOTH YEAR AND MONTH MEAN
temp=ctrain.groupby(['year', 'month'])['PT08.S1(CO)'].mean()
#SOLVE THE X-AXIS INDEX PROBLEM
temp.plot(figsize=(20,10), title= 'Year Month', fontsize=14)

ctrain.groupby('day')['PT08.S1(CO)'].mean().plot.bar()

ctrain.groupby('Hour')['PT08.S1(CO)'].mean().plot.bar()

ctrain.groupby('weekend')['PT08.S1(CO)'].mean().plot.bar()

ctrain.groupby('day of week')['PT08.S1(CO)'].mean().plot.bar()
#--------ENDS HERE---------------------


#LOOK INTO RESAMPLE FUNTION-------------
# Hourly time series
hourly = ctrain.resample('H').mean()

# Converting to daily mean
daily = ctrain.resample('D').mean()

# Converting to weekly mean
weekly = ctrain.resample('W').mean()

# Converting to monthly mean
monthly = ctrain.resample('M').mean()

#TIME SERIES OF DIFFRENT TIME INTERVAL
fig, axs = plt.subplots(4,1)   #WHAT IS fig,axs ??????

hourly['PT08.S1(CO)'].plot(figsize=(15,5), title= 'Hourly', fontsize=14, ax=axs[0])
daily['PT08.S1(CO)'].plot(figsize=(15,5), title= 'Daily', fontsize=14, ax=axs[1])
weekly['PT08.S1(CO)'].plot(figsize=(15,5), title= 'Weekly', fontsize=14, ax=axs[2])
monthly['PT08.S1(CO)'].plot(figsize=(15,5), title= 'Monthly', fontsize=14, ax=axs[3])

plt.show()


ctrain = ctrain.resample('H').mean()

#SPLITTING INTO TRAIN AND VALIDATION
Train=ctrain.ix[:7450]  #index is date therefore split also should 
valid=ctrain.ix[7450:]  #be based on date


Train['PT08.S1(CO)'].plot(figsize=(15,8), title= 'Tin Oxide Level', fontsize=14, label='train')
valid['PT08.S1(CO)'].plot(figsize=(15,8), title= 'Tin Oxide Level', fontsize=14, label='valid')
plt.xlabel("Datetime")
plt.ylabel("Tin Oxide")
plt.legend(loc='best')
plt.show()


from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from sklearn.metrics import mean_squared_error
from math import sqrt

import statsmodels.api as sm


sm.tsa.seasonal_decompose(Train['PT08.S1(CO)']).plot()
result = sm.tsa.stattools.adfuller(ctrain['PT08.S1(CO)'])
plt.show()


y_hat_avg = valid.copy()
fit1 = ExponentialSmoothing(np.asarray(Train['PT08.S1(CO)']),seasonal_periods=48,seasonal='add').fit()
y_hat_avg['Holt_Winter'] = fit1.forecast(len(valid))
plt.figure(figsize=(20,12))
plt.plot(Train['PT08.S1(CO)'], label='Train')
plt.plot(valid['PT08.S1(CO)'], label='Valid')
plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')
plt.show()


rms = sqrt(mean_squared_error(valid['PT08.S1(CO)'], y_hat_avg['Holt_Winter']))
print(rms)


import statsmodels.api as sm
y_hat_avg = valid.copy()
fit1 = sm.tsa.statespace.SARIMAX(Train['PT08.S1(CO)']).fit()
y_hat_avg['SARIMA'] = fit1.predict(start="2005-01-15", end="2005-04-03", dynamic=True)
plt.figure(figsize=(16,8))
plt.plot(Train['PT08.S1(CO)'], label='Train')
plt.plot(valid['PT08.S1(CO)'], label='Valid')
plt.plot(y_hat_avg['SARIMA'], label='SARIMA')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(valid['PT08.S1(CO)'], y_hat_avg.SARIMA))
print(rms)


