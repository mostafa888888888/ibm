#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_csv('D:\\kc_house_data.csv') 
data.head()


# In[3]:


data.info()


# In[4]:


data.drop(['id'],axis=1,inplace=True)


# In[5]:


data.describe()


# In[6]:


data['floors'].value_counts().to_frame()


# In[7]:


sns.boxplot(x="waterfront",y="price",data=data)


# In[8]:


x = data["sqft_above"]
y = data["price"]
plt.scatter(x, y)
plt.xlabel("sqft_above")
plt.ylabel("price")
plt.show()


# In[62]:


y=data[["price"]]
x=data[['bedrooms']]


# In[63]:


from sklearn.model_selection import train_test_split


# In[64]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.30,random_state=0)                                                    


# In[65]:


from sklearn.linear_model import LinearRegression


# In[66]:


regressor=LinearRegression()
regressor.fit(x_train,y_train)


# In[67]:



y_pred=regressor.predict(x_test)
regressor.score(x_train,y_train)


# In[68]:


regressor.score(x_test,y_test)


# In[69]:


plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.xlabel('floors')
plt.ylabel('the price')
plt.show()


# In[70]:


regressor.predict([[13000]])


# In[77]:


from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=44)
x_poly=poly_reg.fit_transform(x)


# In[78]:


line2_reg=LinearRegression()
line2_reg.fit(x_poly,y)


# In[79]:


plt.scatter(x,y,color='red')
plt.plot(x,line2_reg.predict(x_poly),color='blue')

plt.xlabel('the level')
plt.ylabel('salary')
plt.show()


# In[85]:


features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
x = data[features ]
y = data['price']


# In[92]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=1)


# In[97]:


x_train_pr=pr.fit_transform(x_train[['floors','waterfront','lat','bedrooms','sqft_basement','view' ,'bathrooms','sqft_living15','sqft_above','grade','sqft_living']])


# In[98]:


from sklearn.preprocessing import PolynomialFeatures


# In[99]:


pr=PolynomialFeatures(degree=2)
pr


# In[100]:


x_train_pr=pr.fit_transform(x_train[['floors','waterfront','lat','bedrooms','sqft_basement','view' ,'bathrooms','sqft_living15','sqft_above','grade','sqft_living']])


# In[101]:


from sklearn.linear_model import Ridge


# In[102]:



pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[['floors', 'waterfront','lat' ,'bedrooms' ,'sqft_basement' ,'view' ,'bathrooms','sqft_living15','sqft_above','grade','sqft_living']])
x_test_pr=pr.fit_transform(x_test[['floors', 'waterfront','lat' ,'bedrooms' ,'sqft_basement' ,'view' ,'bathrooms','sqft_living15','sqft_above','grade','sqft_living']])


# In[103]:



RidgeModel=Ridge(alpha=0.1)

RidgeModel.fit(x_train_pr, y_train)


# In[104]:


RidgeModel.score(x_train_pr, y_train)


# In[106]:


from sklearn.preprocessing import PolynomialFeatures


# In[107]:



pr=PolynomialFeatures(degree=2)
pr


# In[108]:


x_train_pr=pr.fit_transform(x_train[['floors', 'waterfront','lat' ,'bedrooms' ,'sqft_basement' ,'view' ,'bathrooms','sqft_living15','sqft_above','grade','sqft_living']])


# In[109]:


x_polly=pr.fit_transform(x_train[['floors', 'waterfront','lat' ,'bedrooms' ,'sqft_basement' ,'view' ,'bathrooms','sqft_living15','sqft_above','grade','sqft_living']])


# In[110]:



RidgeModel=Ridge(alpha=0.1)

RidgeModel.fit(x_train_pr, y_train)

RidgeModel.score(x_train_pr, y_train)


# In[ ]:




