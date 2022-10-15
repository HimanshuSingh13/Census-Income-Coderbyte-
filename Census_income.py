#!/usr/bin/env python
# coding: utf-8

# # PROBLEM STATEMENTS

# This data was extracted from the 1994 Census bureau database by Ronny Kohavi and Barry Becker (Data Mining and Visualization, Silicon Graphics). A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) & (AGI>100) & (AFNLWGT>1) && (HRSWK>0)). The prediction task is to determine whether a person makes over $50K a year or less then.

# ## Importing Libraries that will come handy for the projects

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


columns = ['Age', 'Work Class', 'Final Weight', 'Education', 'Education Number', 'Marital Status', 'Occupation',
          'Relationship', 'Race', 'Sex', 'Capital Gain', 'Capital Loss', 'Hours per Week', 'Country', 'Income']
data = pd.read_csv("D:\\data set\\adult.data", names = columns)


# In[3]:


data


# From the above code we can observe that the data set contains 32561 rows × 15 columns.

# ## Exploratory Data Analysis (EDA)

# In[4]:


data.info()


#  From the above we can observe that our data do not contain any missing value and  our target value "Income" has an Object datatype, so before moving further lets convert it into numerical datatype  

# In[5]:


from sklearn.preprocessing import LabelEncoder

labelEncoder = LabelEncoder()
data['Income'] = labelEncoder.fit_transform(data['Income'])


# In[6]:


data.info()


# In[7]:


data.hist(figsize=(20,12))


# From the above graphs we can conclude
# 
# a) Age and Hours per Week column can be group into bins.
# b) In column Capital Gain,Capital Loss and Final Weight the data is left skewed.

# In[8]:


plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),annot = True)


# From the above graph we can observe that there is not very high linear correlation with the target  column("Income") and final wweight column have zero correlation with the target value. so we can drop it.

# ## AGE

# bucketing the age columns into different bins as shown below
# 
# a) 0-25: Young
# 
# b) 25-50: Adult
# 
# c) 50-100: Old

# In[9]:


data['Age'] = pd.cut(data['Age'], bins = [0, 25, 50, 100], labels = ['Young', 'Adult', 'Old'])  


# In[10]:


sns.countplot(x = 'Age', hue = 'Income', data = data)


# From the above graph it can be observed that there is very less young people having Income grated than $50K

# ## Hours per Week

# bucketing the age columns into different bins as shown below
# 
# a) 0-30: Low Hrs
# 
# b) 30-40: Normal Hrs
# 
# c) 40-100:High Hrs

# In[11]:


data['Hours per Week'] = pd.cut(data['Hours per Week'], bins = [0, 30, 40, 100], labels = ['Low Hrs', 'Normal Hrs', 'High Hrs'])  


# In[12]:


sns.countplot(x = 'Hours per Week', hue = 'Income', data = data)


# From the above graph it can be observe that as the working hours increase the no of people receving more than 50k also increase.

# ## Final Weight

# As already discussed we can drop the column as it have zero correlation with the target column.

# In[13]:


data.drop(['Final Weight'], axis = 1, inplace = True)


# In[14]:


data


# ## Capital Gain and Capital loss

# Using the above two column we can come up with new feature.
# Capital_diff= Capital Gain-Capital loss

# In[15]:


data['Capital Diff'] = data['Capital Gain'] - data['Capital Loss']
data.drop(['Capital Gain'], axis = 1, inplace = True)
data.drop(['Capital Loss'], axis = 1, inplace = True)
data['Capital Diff'] = pd.cut(data['Capital Diff'], bins = [-5000, 5000, 100000], labels = ['low', 'High'])
sns.countplot(x = 'Capital Diff', hue = 'Income', data = data)


# we can observe that for both the category low and high people with hight Income (more than 50k)

# ## Work Class

# In[16]:


plt.figure(figsize=(20,10))
sns.countplot(x = 'Work Class', hue = 'Income', data = data)


# From the above graph it can be observed that "without pay" and "Never-worked" column have very less records so it is safe to remove them.
#  we can also observe a category "?" , it is a error and have very low vaalue so we can remove it too.
# 

# In[17]:


data = data.drop(data[data['Work Class'] == ' ?'].index)
data = data.drop(data[data['Work Class'] == ' Without-pay'].index)
data = data.drop(data[data['Work Class'] == ' Never-worked'].index)


# In[18]:


data


# ## Education and Education Number

# In[19]:


plt.figure(figsize=(20,10))
sns.countplot(x = 'Education', hue = 'Income', data = data)


# In[20]:


data['Education'].value_counts()


# ## Marital Status and Relationship

# In[21]:


data['Relationship'].value_counts()


# In[22]:


data['Marital Status'].value_counts()


# ## Occupation

# In[23]:


plt.figure(figsize=(20,10))
plt.xticks(rotation = 45)
sns.countplot(x = 'Occupation', hue = 'Income', data = data)


# From the above grap it can be observe that there is no missing value and there is all unique catagories so we can keep it as it is

# ## Race

# In[24]:


plt.figure(figsize=(20,10))
sns.countplot(x = 'Race', hue = 'Income', data = data)


# From the above graph it can be observed that the maximum information is about white people so we can combine other categories in one as others.

# In[25]:


data['Race'].replace([' Black', ' Asian-Pac-Islander', ' Amer-Indian-Eskimo', ' Other'],' Other', inplace = True)


# In[26]:


plt.figure(figsize=(20,10))
sns.countplot(x = 'Race', hue = 'Income', data = data)


# ## Sex

# In[27]:


plt.figure(figsize=(20,10))
sns.countplot(x = 'Sex', hue = 'Income', data = data)


# From the above it can be observe that there are more Male than compare to Female and there are more male reciving salary more than 50k

# ## Country

# In[28]:


data['Country'].value_counts()


# From the above it can be observed that there is some category with "?" that can be droped and majorty of people are from "US" so we can create two category as " US  and OTHERS"

# In[33]:


dataset = data.drop(data[data['Country'] == ' ?'].index)
countries = np.array(data['Country'].unique())
countries = np.delete(countries, 0)
data['Country'].replace(countries, 'Other', inplace = True)


# In[34]:


plt.figure(figsize=(20,10))
sns.countplot(x = 'Country', hue = 'Income', data = dataset)


# ## Splitting the datasets into features and target value

# In[35]:


data


# In[37]:


y = data['Income']
x= data.drop(['Income'], axis = 1)
x = pd.get_dummies(x)
print("Total features: {}".format(x.shape[1]))


# In[38]:


x


# In[39]:


y


# In[41]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 13)


# ## Machine learning

# ### Importing libraries

# In[42]:


from sklearn.metrics import f1_score, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# In[43]:


classifiers = [GaussianNB(), 
               SVC(kernel = 'rbf', probability = True), 
               DecisionTreeClassifier(random_state = 0), 
               RandomForestClassifier(n_estimators = 100, random_state = 0), 
               GradientBoostingClassifier(random_state = 0)]
classifier_names = ["Gaussian Naive Bayes", 
                    "Support Vector Classifier", 
                    "Decision Tree Classifier", 
                    "Random Forest Classifier", 
                    "Gradient Boosting Classifier"]
accuracies = []


# In[45]:


for i in range(len(classifiers)):
    classifier = classifiers[i]
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    print("{}:".format(classifier_names[i]))
    print("F1 score: {:.2f}".format(f1_score(y_test, y_pred)))
    accuracy = accuracy_score(y_test, y_pred)*100
    accuracies.append(accuracy)


# #### From the above result it can be observe that "Gradient Boosting Classifier: F1 score: 0.66" is performing best

# In[ ]:




