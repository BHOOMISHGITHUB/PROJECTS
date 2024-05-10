#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# # DATA COLLECTION AND ANALYSING

# In[2]:


df=pd.read_csv(r"C:\Users\BHOOMISH\Downloads\diabetes.csv")
df


# In[3]:


df.head()


# In[4]:


df.shape 


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df.describe()


# In[8]:


df["Outcome"].value_counts() #value_count which uses to find the how many 0 and 1 are present 


# In[9]:


df.groupby("Outcome").mean()


# In[10]:


#separating data and labels
x=df.iloc[:,:-1]
y=df.iloc[:,-1]


# # data standardization

# In[11]:


scaler=StandardScaler()


# In[12]:


scaler


# In[13]:


scaler.fit(x)


# In[14]:


standardized_data=scaler.transform(x)
standardized_data


# In[15]:


x = standardized_data
y=df.iloc[:,-1]


# In[ ]:





# In[16]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2) 


# In[17]:


x_test


# In[18]:


x_train.shape


# In[19]:


x.shape


# In[20]:


x_test.shape


# In[21]:


#training the model
classifier= svm.SVC(kernel="linear")


# In[22]:


#training the support vector machine classifier
classifier.fit(x_train,y_train)


# In[23]:


#evaluate the model
#accuracy score
#accuracy score on the training data
x_train_prediction=classifier.predict(x_train)


# In[24]:


#finding the training data
training_data_accuracy=accuracy_score(x_train_prediction,y_train)


# In[25]:


print("accuracy_score : ",training_data_accuracy)


# In[26]:


#accuracy score on the test day 
x_test_prediction=classifier.predict(x_test)
training_data_accuracy=accuracy_score(x_test_prediction,y_test)
print("Accuracy_score_test_data : ",training_data_accuracy)


# In[27]:


#making a predictive system 
input_data=(5,166,72,19,175,25.8,0.587,51)

#changing the input data into numpy array
input_data_np_aaray=np.asarray(input_data)

#we should reshape the array for one instance
input_data_reshape=input_data_np_aaray.reshape(1,-1)


# In[28]:


#standardize the input data
std_data=scaler.transform(input_data_reshape)


# In[29]:


std_data


# In[30]:


prediction=classifier.predict(std_data)


# In[31]:


prediction


# In[38]:


if prediction==0:
    print("Diabetes free person")
else:
    print("Diabetes")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




