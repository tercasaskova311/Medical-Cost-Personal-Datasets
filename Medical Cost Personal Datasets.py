#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import scipy.stats
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


# In[7]:


# Read the data from the file
# Data from https://www.kaggle.com/mirichoi0218/insurance 
df = pd.read_csv('/Users/terezasaskova/Desktop/Data_P2_Theory.csv')


# In[8]:


# Check the first few lines to see the column names and type of content
df.head()


# In[9]:


# Let us check the types of the different columns
df.dtypes


# In[10]:


# Let us convert sex, smoker and region to categorical
df['sex'] = df['sex'].astype('category')
df['smoker'] = df['smoker'].astype('category')
df['region'] = df['region'].astype('category')


# In[11]:


#DATA EXPLORATION
# Lets see the summary of the numerical columns
df.describe()


# In[12]:


# Histogram of charges
df.hist(column="charges");


# In[13]:


# Check if there are differences by sex
df['charges'].hist(by=df['sex'])


# In[14]:


# Check the number of data in the different categories
df['sex'].value_counts()


# In[15]:


df['smoker'].value_counts()


# In[16]:


df['region'].value_counts()


# In[17]:


# Check now some combinations
contingency=pd.crosstab(df["sex"],df["smoker"])
print(contingency)


# In[19]:


# Perform the chi-squared test
c, p, dof, expected = scipy.stats.chi2_contingency(contingency)

# Print the p-value
print('p-value=%f' % p)

# Check if sex and smoking are independent
if p < 0.05:
    print("Sex and smoking are not independent (p < 0.05)")
else:
    print("Sex and smoking are independent (p >= 0.05)")


# In[20]:


#CLASSIFICATION

df['upperQuarter']=df['charges']>np.quantile(df['charges'],0.75)
df.head()


# In[22]:


# Check if there is any relationship between being in the upper quarter and the␣ ↪rest of variables
c, p, dof, expected = scipy.stats.chi2_contingency(pd.
crosstab(df['upperQuarter'],df['sex']))

print('p-value upperQuarter vs sex=%f'%p)


# In[24]:


c, p, dof, expected = scipy.stats.chi2_contingency(pd.
    crosstab(df['upperQuarter'],df['smoker']))
print('p-value upperQuarter vs smoker=%f'%p)


# In[26]:


c, p, dof, expected = scipy.stats.chi2_contingency(pd.
    crosstab(df['upperQuarter'],df['region']))


# In[27]:


print('p-value upperQuarter vs region=%f'%p)


# In[28]:


#Next steep - the numerical variables
sns.boxplot(x='upperQuarter', y='age', data=df)


# In[29]:


sns.boxplot(x='upperQuarter', y='bmi', data=df)


# In[31]:


sns.boxplot(x='upperQuarter', y='children', data=df)


# In[34]:


#NEXT: CLASSIFICATION TREE
from sklearn import tree
clf = tree.DecisionTreeClassifier()
x = df[['age','sex','bmi','children','smoker','region']] 
y = df['upperQuarter']


# In[35]:


print ('A classification is a problem of the form y=f(X) where y is categorical and X is whatever')


# In[37]:


x.head()


# In[38]:


y.head()


# In[39]:


x=pd.get_dummies(x)
x.head()


# In[41]:


clf = clf.fit(x,y)
df_feature_names = list(x.columns) 
df_target_names = [str(s) for s in y.unique()]


# In[42]:


#evaluation of the tree
from sklearn import metrics
y_pred = clf.predict(x) 

print("Accuracy:",metrics.accuracy_score(y, y_pred))


# In[43]:


#vizualization

print(tree.export_text(clf, feature_names=df_feature_names))


# In[44]:


#makes it simpliest

print('Current depth of the tree=%d'%clf.tree_.max_depth)


# In[48]:


max_depth = []; acc_gini = []; acc_entropy = []
for i in range(1,17):
    clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=i)
    clf.fit(x, y)
    y_pred = clf.predict(x)
    acc_gini.append(metrics.accuracy_score(y, y_pred))
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=i)
    clf.fit(x, y)
    y_pred = clf.predict(x)
    acc_entropy.append(metrics.accuracy_score(y, y_pred))
    max_depth.append(i)
d = pd.DataFrame({'acc_gini':pd.Series(acc_gini),
 'acc_entropy':pd.Series(acc_entropy),
 'max_depth':pd.Series(max_depth)})

# visualizing changes in parameters
plt.plot('max_depth','acc_gini', data=d, label='gini')
plt.plot('max_depth','acc_entropy', data=d, label='entropy')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.legend()


# In[50]:


clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2)
clf.fit(x, y)
y_pred = clf.predict(x)

print('Accuracy with max_depth=%d is %f'%(2,metrics.accuracy_score(y,y_pred)))


# In[51]:


fig = plt.figure(figsize=(15,10))
_ = tree.plot_tree(clf,
feature_names=df_feature_names, class_names=df_target_names, filled=True)


# In[57]:


from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(clf, x, y)


# In[60]:


#the tree above does not separete anything => extremly simple classifier

clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=1) 
clf.fit(x, y)
y_pred = clf.predict(x)

print('Accuracy with max_depth=%d is %f'%(1,metrics.accuracy_score(y,y_pred)))


# In[59]:


_ = tree.plot_tree(clf,
                   feature_names=df_feature_names,
class_names=df_target_names, filled=True)


# In[61]:


ConfusionMatrixDisplay.from_estimator(clf, x, y)


# In[ ]:




