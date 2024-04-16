#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np


# In[17]:


df = pd.read_csv("C:/Users/omarz/Documents/4th year/2nd semester/Pattern/Assignments/pattern/adult.csv")
df.head()


# In[18]:


df.shape


# In[19]:


df.describe()


# In[20]:


df.info()


# In[21]:


df['income'].value_counts()


# In[22]:


df['sex'].value_counts()


# In[23]:


df['native.country'].value_counts()


# In[24]:


df['workclass'].value_counts()


# In[25]:


df['occupation'].value_counts()


# In[26]:


df = df.drop(['education', 'fnlwgt'], axis = 1)
df.head(1)


# In[27]:


df.replace('?', np.NaN,inplace = True)
df.head()


# In[28]:


df.fillna(method = 'ffill', inplace = True)


# In[29]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['workclass'] = le.fit_transform(df['workclass'])
df['marital.status'] = le.fit_transform(df['marital.status'])
df['occupation'] = le.fit_transform(df['occupation'])
df['relationship'] = le.fit_transform(df['relationship'])
df['race'] = le.fit_transform(df['race'])
df['sex'] = le.fit_transform(df['sex'])
df['native.country'] = le.fit_transform(df['native.country'])
df['income'] = le.fit_transform(df['income'])

df.head()


# In[30]:


x = df.drop(['income'], axis = 1)
y = df['income']


# In[31]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3) 


# In[34]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
gb = GaussianNB()
gb.fit(x_train,y_train)


# In[36]:


y_pred = gb.predict(x_test)


# In[37]:


tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()


# In[42]:


sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
posterior_probability = gb.predict_proba(x_test)[:, 1]  # Probability of being in the positive class (>50K)


# In[44]:


print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
print("Posterior Probability:", posterior_probability)


# In[ ]:




