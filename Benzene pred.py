# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 19:48:25 2018

@author: User
"""
import pandas as pd          
import numpy as np          # For mathematical calculations
import matplotlib.pyplot as plt  # For plotting graphs
from datetime import datetime    # To access datetime
from pandas import Series        # To work on series



dataframe = pd.read_csv("AirQualityUCI_new.csv",parse_dates=[['Date', 'Time']])

dataframe_original = dataframe.copy()

train = dataframe[['Date_Time','C6H6(GT)']]

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
    if(ctrain['C6H6(GT)'][i] == -200):        
        ctrain['C6H6(GT)'][i]=ctrain['C6H6(GT)'][i-1]

#print(ctrain['C6H6(GT)'][523])

plt.figure(figsize=(16,8))
plt.plot(ctrain['C6H6(GT)'], label='Benzene Level')
plt.title('Time Series(hourly)')
plt.xlabel("Time")
plt.ylabel("BENZENE")
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
ctrain.groupby('year')['C6H6(GT)'].mean().plot.bar() #not very good as 2004 entries are much more than 2005

#by month but diffrent years
ctrain.groupby('month')['C6H6(GT)'].mean().plot.bar(color='#3075ec')

#BOTH YEAR AND MONTH MEAN
temp=ctrain.groupby(['year', 'month'])['C6H6(GT)'].mean()
#SOLVE THE X-AXIS INDEX PROBLEM
temp.plot(figsize=(20,10), title= 'Year Month', fontsize=14)

ctrain.groupby('day')['C6H6(GT)'].mean().plot.bar()

ctrain.groupby('Hour')['C6H6(GT)'].mean().plot.bar()

ctrain.groupby('weekend')['C6H6(GT)'].mean().plot.bar()

ctrain.groupby('day of week')['C6H6(GT)'].mean().plot.bar()
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



monthly['C6H6(GT)']

#TIME SERIES OF DIFFRENT TIME INTERVAL
fig, axs = plt.subplots(4,1)   #WHAT IS fig,axs ??????

hourly['C6H6(GT)'].plot(figsize=(15,5), title= 'Hourly', fontsize=14, ax=axs[0])
daily['C6H6(GT)'].plot(figsize=(15,5), title= 'Daily', fontsize=14, ax=axs[1])
weekly['C6H6(GT)'].plot(figsize=(15,5), title= 'Weekly', fontsize=14, ax=axs[2])
monthly['C6H6(GT)'].plot(figsize=(15,5), title= 'Monthly', fontsize=14, ax=axs[3])

plt.show()


ctrain = ctrain.resample('H').mean()

#SPLITTING INTO TRAIN AND VALIDATION
Train=ctrain.ix[:7470]  #index is date therefore split also should 
valid=ctrain.ix[7471:]  #be based on date

DTrain = daily.ix[:293]
Dvalid = daily.ix[292:]

Train['C6H6(GT)'].plot(figsize=(15,8), title= 'Benzene Level', fontsize=14, label='train')
valid['C6H6(GT)'].plot(figsize=(15,8), title= 'Benzene Level', fontsize=14, label='valid')
plt.xlabel("Datetime")
plt.ylabel("Benzene microg/m^3")
plt.legend(loc='best')
plt.show()


from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from sklearn.metrics import mean_squared_error
from math import sqrt

import statsmodels.api as sm

sm.tsa.seasonal_decompose(Train['C6H6(GT)']).plot()
result = sm.tsa.stattools.adfuller(ctrain['C6H6(GT)'])
plt.show()

d = np.asarray(Dvalid['C6H6(GT)'])
print(d.mean())

y_hat_avg = valid.copy()
fit1 = ExponentialSmoothing(np.asarray(Train['C6H6(GT)']),seasonal_periods=168,trend='add',seasonal='add').fit()
y_hat_avg['Holt_Winter'] = fit1.forecast(len(valid))
plt.figure(figsize=(20,12))
plt.plot(Train['C6H6(GT)'], label='Train')
plt.plot(valid['C6H6(GT)'], label='Valid')
plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(valid['C6H6(GT)'], y_hat_avg.Holt_Winter))
print(rms)

#6-SARIMAX
import statsmodels.api as sm
y_hat_avg = Dvalid.copy()
fit1 = sm.tsa.statespace.SARIMAX(DTrain['C6H6(GT)'], order=(2, 1, 4),seasonal_order=(0,1,1,7)).fit()
y_hat_avg['SARIMA'] = fit1.predict(start="2004-12-28", end="2005-04-03", dynamic=True)
plt.figure(figsize=(16,8))
plt.plot(DTrain['C6H6(GT)'], label='Train')
plt.plot(Dvalid['C6H6(GT)'], label='Valid')
plt.plot(y_hat_avg['SARIMA'], label='SARIMA')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(Dvalid['C6H6(GT)'], y_hat_avg.SARIMA))
print(rms)

