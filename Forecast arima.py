#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[33]:


df=pd.read_csv('perrin-freres-monthly-champagne-.csv')


# In[34]:


df.head()


# In[35]:


df.tail()


# In[36]:


## Cleaning up the data
df.columns=["Month","Sales"]
df.head()


# In[37]:


## Drop last 2 rows
df.drop(106,axis=0,inplace=True)


# In[38]:


df.tail()


# In[40]:


df.drop(105,axis=0,inplace=True)


# In[41]:


df.tail()


# In[42]:


# Convert Month into Datetime
df['Month']=pd.to_datetime(df['Month'])

df.head()


# In[43]:


df.set_index('Month',inplace=True)

df.head()


# In[44]:


df.describe()


# In[45]:


df.plot()


# In[46]:


### Testing For Stationarity

from statsmodels.tsa.stattools import adfuller


# In[47]:


test_result=adfuller(df['Sales'])


# In[48]:


#Ho: It is non stationary
#H1: It is stationary

def adfuller_test(sales):
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")


# In[49]:


adfuller_test(df['Sales'])


# In[50]:


df['Sales First Difference'] = df['Sales'] - df['Sales'].shift(1)


# In[51]:


df['Sales'].shift(1)


# In[52]:


df['Seasonal First Difference']=df['Sales']-df['Sales'].shift(12)


# In[53]:


df.head(14)


# In[54]:


## Again test dickey fuller test
adfuller_test(df['Seasonal First Difference'].dropna())


# In[55]:


df['Seasonal First Difference'].plot()


# In[59]:


from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df['Sales'])
plt.show()


# In[68]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm


# In[69]:


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['Seasonal First Difference'].iloc[13:],lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Seasonal First Difference'].iloc[13:],lags=40,ax=ax2)


# In[62]:


# For non-seasonal data
#p=1, d=1, q=0 or 1
from statsmodels.tsa.arima_model import ARIMA


# In[65]:


model=ARIMA(df['Sales'],order=(1,1,1))
model_fit=model.fit()


# In[66]:


model_fit.summary()


# In[67]:


df['forecast']=model_fit.predict(start=90,end=103,dynamic=True)
df[['Sales','forecast']].plot(figsize=(12,8))


# In[70]:


model=sm.tsa.statespace.SARIMAX(df['Sales'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
results=model.fit()


# In[71]:


df['forecast']=results.predict(start=90,end=103,dynamic=True)
df[['Sales','forecast']].plot(figsize=(12,8))


# In[72]:


from pandas.tseries.offsets import DateOffset
future_dates=[df.index[-1]+ DateOffset(months=x)for x in range(0,24)]


# In[73]:


future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df.columns)

future_datest_df.tail()


# In[74]:


future_df=pd.concat([df,future_datest_df])


# In[75]:


future_df['forecast'] = results.predict(start = 104, end = 120, dynamic= True)  
future_df[['Sales', 'forecast']].plot(figsize=(12, 8)) 


# In[ ]:




