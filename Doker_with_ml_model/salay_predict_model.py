#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


dataset=pd.read_csv("SalaryData.csv")


# In[3]:


dataset.columns


# In[4]:


x=dataset["YearsExperience"].values.reshape(-1,1)


# In[5]:


y=dataset["Salary"]


# In[6]:


y.ndim


# In[7]:


x.shape


# In[8]:


x.ndim


# In[9]:


from sklearn.linear_model import LinearRegression


# 

# In[10]:


model=LinearRegression()


# model

# In[11]:


model


# In[12]:


model.fit(x,y)


# In[13]:


model.coef_


# In[14]:


model.predict([[1.1]])


# In[15]:


import joblib


# In[18]:


joblib.dump(model,"salaray_model.pk1")


# In[19]:


model2=joblib.load("salaray_model.pk1")


# 

# In[20]:


model2.predict([[1.1]])


# In[ ]:




