#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[47]:


print(np.__version__)


# In[48]:


print(pd.__version__)


# In[49]:


print(tf.__version__)


# In[50]:


print(sns.__version__)


# In[51]:


print(matplotlib.__version__)


# In[52]:


dataset = pd.read_csv("https://raw.githubusercontent.com/Rupesh37/3.-Police-Data/main/sendtorupeshagain.csv")
dataset.head()


# In[53]:


dataset.drop(["SNo","Name","Symbol","Date","Percentage_Change"],axis=1,inplace=True)
dataset


# In[54]:


dataset.info()


# In[55]:


dataset.describe()


# In[56]:


dataset.isnull().value_counts()


# In[57]:


print(len(dataset))


# In[58]:


split_row = len(dataset) - int(0.2 * len(dataset))
train_data = dataset.iloc[:split_row]
test_data = dataset.iloc[split_row:]


# In[59]:


fig, ax = plt.subplots(1, figsize=(13, 7))
ax.plot(train_data['High'], label='Train', linewidth=2,color='r')
ax.plot(test_data['High'], label='Test', linewidth=2,color='b')
ax.set_ylabel('Price_in_USD', fontsize=15)
ax.set_title('Bitcoin_Price_High', fontsize=16)
ax.legend(loc='best', fontsize=14)


# In[60]:


fig, ax = plt.subplots(1, figsize=(13, 7))
ax.plot(train_data['Low'], label='Train', linewidth=2,color='r')
ax.plot(test_data['Low'], label='Test', linewidth=2,color='b')
ax.set_ylabel('Price_in_USD', fontsize=15)
ax.set_title('Bitcoin_Price_Low', fontsize=16)
ax.legend(loc='best', fontsize=14)


# In[61]:


fig, ax = plt.subplots(1, figsize=(13, 7))
ax.plot(train_data['Open'], label='Train', linewidth=2,color='r')
ax.plot(test_data['Open'], label='Test', linewidth=2,color='b')
ax.set_ylabel('Price_in_USD', fontsize=15)
ax.set_title('Bitcoin_Price_Open', fontsize=16)
ax.legend(loc='best', fontsize=14)


# In[62]:


fig, ax = plt.subplots(1, figsize=(13, 7))
ax.plot(train_data['Close'], label='Train', linewidth=2,color='r')
ax.plot(test_data['Close'], label='Test', linewidth=2,color='b')
ax.set_ylabel('Price_in_USD', fontsize=15)
ax.set_title('Bitcoin_Price_Close', fontsize=16)
ax.legend(loc='best', fontsize=14)


# In[63]:


fig, ax = plt.subplots(1, figsize=(13, 7))
ax.plot(train_data['Volume'], label='Train', linewidth=2,color='r')
ax.plot(test_data['Volume'], label='Test', linewidth=2,color='b')
ax.set_ylabel('Volume', fontsize=15)
ax.set_title('Bitcoin_Volume', fontsize=16)
ax.legend(loc='best', fontsize=14)


# In[64]:


fig, ax = plt.subplots(1, figsize=(13, 7))
ax.plot(train_data['Marketcap'], label='Train', linewidth=2,color='r')
ax.plot(test_data['Marketcap'], label='Test', linewidth=2,color='b')
ax.set_ylabel('Marketcap_in_USD', fontsize=15)
ax.set_title('Bitcoin_Market_Capitalization', fontsize=16)
ax.legend(loc='best', fontsize=14)


# In[65]:


fig, ax = plt.subplots(1, figsize=(13, 7))
ax.plot(train_data['US_inflation_rate'], label='Train', linewidth=2,color='r')
ax.plot(test_data['US_inflation_rate'], label='Test', linewidth=2,color='b')
ax.set_ylabel('US_inflation_rate', fontsize=15)
ax.set_title('Inflation_Rate', fontsize=16)
ax.legend(loc='best', fontsize=14)


# In[66]:


X  = train_data.drop(['Close'],axis=1)
y = train_data['Close']


# In[67]:


X = np.asarray(X).astype(np.float32)
y = np.asarray(y).astype(np.float32)


# In[68]:


from sklearn.model_selection import train_test_split

X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.15, random_state=101)


# In[69]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_cv = scaler.fit_transform(X_cv)


# In[70]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


# In[71]:


model = Sequential()

model.add(Dense(24,activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(24,activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(24,activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(24,activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(24,activation='relu'))




model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')


# In[72]:


from keras.callbacks import EarlyStopping
clb = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)


# In[73]:


model.fit(X_train, y_train,validation_data=(X_cv,y_cv),batch_size=64,epochs=100,verbose=1,callbacks=[clb],shuffle=True)


# In[74]:


loss = pd.DataFrame(model.history.history)
plt.figure(figsize=(30,10))
loss.plot()


# In[75]:


model.summary()


# In[76]:


X_test = test_data.drop(['Close'],axis=1)
y_test = test_data['Close']


# In[77]:


X_test = np.asarray(X).astype(np.float32)
y_test = np.asarray(y).astype(np.float32)


# In[78]:


X_test = scaler.fit_transform(X_test)


# In[79]:


predicted_price = model.predict(X_test)
real_price = y_test


# In[80]:


fig, ax = plt.subplots(1, figsize=(13, 7))
ax.plot(real_price, label='Real_Price', linewidth=1)
ax.plot(predicted_price, label='Predicted_Price', linewidth=1)
ax.set_ylabel('Price_in_USD', fontsize=15)
ax.set_title('Actual_versus_Predicted_Price', fontsize=16)
ax.legend(loc='best', fontsize=14)


# In[81]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# In[82]:


mean_abs_error = mean_absolute_error(real_price,predicted_price)
mean_abs_error


# In[83]:


mean_sq_error = mean_squared_error(real_price,predicted_price)
mean_sq_error


# In[84]:


real_price = np.array(real_price)
predicted_price = np.array(predicted_price)

mape = np.mean(np.abs((real_price - predicted_price)/real_price)*100)
mape


# In[85]:


abs_percent_error = []


# In[86]:


for i in range (len(real_price)):
    per_error = (real_price[i] - predicted_price[i])/real_price[i]
    per_error = abs(per_error)
    abs_percent_error.append(per_error)
    
mape = sum(abs_percent_error)/len(abs_percent_error)    


# In[87]:


mape


# In[88]:


percent_mape = mape*100
print(percent_mape)


# In[ ]:





# In[ ]:




