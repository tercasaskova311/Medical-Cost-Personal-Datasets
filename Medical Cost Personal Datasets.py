#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import scipy.stats
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


# Read the data from the file
# Data from https://www.kaggle.com/mirichoi0218/insurance 
df = pd.read_csv('/Users/Desktop/Data_P2_Theory.csv')

# Check the first few lines to see the column names and type of content
df.head()

# Let us check the types of the different columns
df.dtypes

# Let us convert sex, smoker and region to categorical
df['sex'] = df['sex'].astype('category')
df['smoker'] = df['smoker'].astype('category')
df['region'] = df['region'].astype('category')

#DATA EXPLORATION
# Lets see the summary of the numerical columns
df.describe()

# Histogram of charges
df.hist(column="charges");

# Check if there are differences by sex
df['charges'].hist(by=df['sex'])

# Check the number of data in the different categories
df['sex'].value_counts()

df['smoker'].value_counts()

df['region'].value_counts()

# Check now some combinations
contingency=pd.crosstab(df["sex"],df["smoker"])
print(contingency)


# Perform the chi-squared test
c, p, dof, expected = scipy.stats.chi2_contingency(contingency)

# Print the p-value
print('p-value=%f' % p)

# Check if sex and smoking are independent
if p < 0.05:
    print("Sex and smoking are not independent (p < 0.05)")
else:
    print("Sex and smoking are independent (p >= 0.05)")


#CLASSIFICATION

df['upperQuarter']=df['charges']>np.quantile(df['charges'],0.75)
df.head()


# Check if there is any relationship between being in the upper quarter and the␣ ↪rest of variables
c, p, dof, expected = scipy.stats.chi2_contingency(pd.
crosstab(df['upperQuarter'],df['sex']))

print('p-value upperQuarter vs sex=%f'%p)



c, p, dof, expected = scipy.stats.chi2_contingency(pd.
    crosstab(df['upperQuarter'],df['smoker']))
print('p-value upperQuarter vs smoker=%f'%p)

c, p, dof, expected = scipy.stats.chi2_contingency(pd.
    crosstab(df['upperQuarter'],df['region']))

print('p-value upperQuarter vs region=%f'%p)

#Next steep - the numerical variables
sns.boxplot(x='upperQuarter', y='age', data=df)

sns.boxplot(x='upperQuarter', y='bmi', data=df)


sns.boxplot(x='upperQuarter', y='children', data=df)


#NEXT: CLASSIFICATION TREE
from sklearn import tree
clf = tree.DecisionTreeClassifier()
x = df[['age','sex','bmi','children','smoker','region']] 
y = df['upperQuarter']

print ('A classification is a problem of the form y=f(X) where y is categorical and X is whatever')

x.head()

y.head()

x=pd.get_dummies(x)
x.head()

clf = clf.fit(x,y)
df_feature_names = list(x.columns) 
df_target_names = [str(s) for s in y.unique()]

#evaluation of the tree
from sklearn import metrics
y_pred = clf.predict(x) 

print("Accuracy:",metrics.accuracy_score(y, y_pred))

#vizualization

print(tree.export_text(clf, feature_names=df_feature_names))


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



#the tree above does not separete anything => extremly simple classifier

clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=1) 
clf.fit(x, y)
y_pred = clf.predict(x)

print('Accuracy with max_depth=%d is %f'%(1,metrics.accuracy_score(y,y_pred)))


_ = tree.plot_tree(clf,
                   feature_names=df_feature_names,
class_names=df_target_names, filled=True)


ConfusionMatrixDisplay.from_estimator(clf, x, y)






