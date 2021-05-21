#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install imblearn


# In[ ]:





# In[ ]:





# In[3]:


import pandas as pd
import numpy as np


# In[4]:


df=pd.read_csv('wine_quality.csv')
df


# In[6]:


df.info()


# It is vsisble that no null values are present among all the columns ans no str dtype is present making it a fairly numerical dataset.
# 

# In[7]:


df.shape


# The daatset contains 1599 rows and 12 columns. 

# In[8]:


df.isnull().sum()


# In[9]:


df.dtypes


# The dataset contains all numeric values.

# In[10]:


df.describe()


# Observing the std of the residual sugar ,free sulphur di oxide , total sufur di oxide it can be understood that there are ouliers present in the data.

# In[11]:


cor_mat=df.corr()
cor_mat


# In[34]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[13]:


plt.figure(figsize=(10,7))
sns.heatmap(cor_mat.corr(), annot= True)
plt.show


# It is visible from the above heatmap that there is a significant high positive correlation between -'citric acid', 'density' , 'free sulfur dioxide' , 'alcohol' and 'quality'. 
# Whereas 'pH' , 'volatile acidity' show mostly negative corelation.

# In[14]:


sns.countplot(df['quality'])


# The countplot shows the varrying quality of wine with highest quality being 8 and the lowest being 3. 

# In[15]:


sns.pairplot(df)
plt.show()


# In[ ]:





# In[16]:


plt.figure(figsize=[10,8])
plt.bar(df['quality'], df['alcohol'])
plt.xlabel('quality')
plt.ylabel('alcohol')
plt.show()


# In[ ]:





# In[8]:


df['b_quality']=[1 if x>=7 else 0 for x in df['quality']]
df


# By making a new column 'b_quality' the multiple outputs in the data can be sujected to only 2 values "0" and "1". This will making the data prediction easier. Therefore, the values with 1 can be considered as good quality wine and 0 will be considered bad quality.

# In[18]:


sns.countplot(df['b_quality'])


# Since the df['b_quality '] col will be serving as the y variable so it is interesting to observe that there is a class imbalance problem. and hence it must be corrected .

# In[9]:


from scipy.stats import zscore
z=np.abs(zscore(df))
print(z)
treshhold=3 
print (np.where(z<3))
df_new=df[(z<3).all(axis=1)]
print(df_new)


# After removing the outliers it can be observed that 148 rows have been removed almost 9% data have been removed and the data can now bw proceeded for further analysis.

# In[ ]:





# In[ ]:





# In[ ]:





# In[1]:





# In[11]:


df_new.items 


# In[ ]:





# In[12]:



y=df_new['b_quality']
y.value_counts()


# In[15]:


x=df_new.drop(['quality', 'b_quality'], axis=1)
x


# y values are imbalanced so it needs to be balanced by oversampling using SMOTE().

# In[17]:


from imblearn.over_sampling import SMOTE
SM=SMOTE()
X,Y= SM.fit_resample(x,y)


# In[18]:


Y.value_counts()


# now that the dataset is balanced scaling can be done by ranging the values between 0and 1 through normalisation via train_test_split

# In[19]:



from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.4,random_state=42)



# In[20]:


X_train


# In[41]:


from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix , classification_report
import warnings
warnings.filterwarnings('ignore')


# In[42]:


from sklearn.linear_model import LogisticRegression

lg=LogisticRegression()
lg.fit(X_train,Y_train)
Y_pred=lg.predict(X_test)
print(Y_pred)
print(accuracy_score(Y_test,Y_pred))
print(confusion_matrix(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))


# Logistic Regression shows an accuracy of 84%

# In[43]:


from sklearn.svm import LinearSVC

clf = LinearSVC(random_state=0, tol=1e-5)

clf.fit(X_train, Y_train.ravel()) 
clf.score(X_train,Y_train)
pred_clf=clf.predict(X_test)
print(accuracy_score(Y_test,pred_clf))
print(confusion_matrix(Y_test,pred_clf))
print(classification_report(Y_test, pred_clf))


# SVC has an accuracy of 81%

# In[25]:


dtc=DecisionTreeClassifier()
dtc.fit(X_train,Y_train)
dtc.score(X_train,Y_train)
pred_dtc=dtc.predict(X_test)
print(accuracy_score(Y_test,pred_dtc))
print(confusion_matrix(Y_test,pred_dtc))
print(classification_report(Y_test, pred_dtc))


# Decision tree shows an accuracy of 90% which is higher.

# In[27]:


knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,Y_train)
knn.score(X_train,Y_train)
pred_knn=knn.predict(X_test)
print(accuracy_score(Y_test,pred_knn))
print(confusion_matrix(Y_test,pred_knn))
print(classification_report(Y_test, pred_knn))


# Kneighbors has a acore of 86% 

# In[21]:


from sklearn.preprocessing import MinMaxScaler
MM=MinMaxScaler()
MM_fit=MM.fit(X_train)
MM_train=MM_fit.transform(X_train)
MM_test=MM_fit.transform(X_test)
print(MM_train)
print(MM_test)


# The values have undergone the process of normalisation and has been converted to np array.

# In[30]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc


# In[37]:


false_positive_rate , true_positive_rate ,threshold = roc_curve(Y_test,Y_pred)
roc_auc= auc(false_positive_rate , true_positive_rate)


# In[32]:


tpr=true_positive_rate
fpr=false_positive_rate
print(tpr,fpr,threshold)


# In[35]:


plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label='Logistic Regression')
plt.xlabel('false_positive_rate')
plt.ylabel('true_positive_rate')
plt.title('Logistic Regression')
plt.show()


# In[36]:


auc_score=roc_auc_score(Y_test,lg.predict(X_test))
print(auc_score)


# In[38]:


rf= RandomForestClassifier()
rf.fit(X_train,Y_train)
train_score=rf.score(X_train,Y_train)
test_score=rf.score(X_test, Y_test)
MSE= mean_squared_error(Y_test,Y_pred)
RMSE= np.sqrt(MSE)
print('train_score:' , train_score)
print('mean_squared_error :' , MSE)
print('root_mean_squared_error :' , RMSE)
print('test_score:' , test_score)


# In[40]:


X_predict = list(rf.predict(X_test))
df_new = {'predicted':X_predict,'orignal':Y_test}
pd.DataFrame(df_new).head(10)


# It can be seen that the random forest scores the highest of all the algorithm with the lowest errors provided and as  a result it is the best suited model in this case.

# In[ ]:




