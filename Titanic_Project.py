#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


df_train = pd.read_csv('train.csv')


# In[4]:


df_train['Age'].fillna(df_train['Age'].mean(), inplace=True)
df_train['Embarked'].fillna(df_train['Embarked'].mode()[0], inplace=True)


# In[5]:


df_train['Age']


# In[6]:


def get_category(age):
    if age<18:
        return 'CHILD'
    if age>55:
        return 'OLDER'
    return 'YOUNG'


# In[7]:


df_train['Age_Cat']=df_train['Age'].apply(get_category)


# In[8]:


df_train['Age_Cat'].value_counts()


# In[9]:


import seaborn as sb


# In[10]:


sb.countplot(df_train['Age_Cat'], hue=df_train['Survived'])


# In[11]:


df_train.info()


# In[13]:


df_train['inCabin'] = (~df_train['Cabin'].isna()).astype(int)


# In[14]:


sb.countplot(df_train['inCabin'], hue=df_train['Survived'])


# # OneHotEncoding

# In[15]:


df_train['Embarked']


# In[16]:


temp = pd.get_dummies(df_train['Embarked'], drop_first=True, prefix='Embarked')


# In[17]:


#df_train['Embarked'] = df_train['Embarked'].map({'S':0, 'C':1, 'Q':2})


# In[18]:


df_train.columns


# In[17]:


#df_train['Sex'] = df_train['Sex'].map({'male':0, 'female':1})


# In[19]:


temp1=pd.get_dummies(df_train['Sex'], drop_first=True)


# In[20]:


df_train.info()


# In[21]:


# Label Encoding
df_train['Age_Cat']=df_train['Age_Cat'].map({'CHILD':0, 'YOUNG':1, 'OLDER':2})


# In[22]:


df_train['Age_Cat'].unique()


# In[23]:


df_train=pd.concat([df_train,temp],axis=1)


# In[24]:


df_train=pd.concat([df_train,temp1],axis=1)


# In[25]:


df_train.columns


# In[25]:


#'Braund, Mr. Owen Harris'.split(' ')[1]


# In[26]:


#df_train['Title']=df_train['Name'].apply(lambda X:X.split(' ')[1])


# In[27]:


#df_train['Title'].value_counts()


# In[26]:


df_train.info()


# In[27]:


df_train.head()


# In[29]:


df_corr=df_train.corr()['Survived']


# In[31]:


df_corr[df_corr.apply(lambda x:x>=-0.1 and x<=0.1)]


# In[31]:


#Select columns to train model
selected_columns = list(df_train.columns)
selected_columns.remove('Survived')
selected_columns.remove('PassengerId')
selected_columns.remove('Name')
selected_columns.remove('Ticket')
selected_columns.remove('Cabin')
selected_columns.remove('Age')
selected_columns.remove('Embarked')
selected_columns.remove('Sex')
selected_columns.remove('Embarked_Q')


#selected_columns.remove('Fare')
#selected_columns.remove('Title')
selected_columns


# # Building a ML Model

# In[32]:


X = df_train[selected_columns]
y = df_train['Survived']


# In[33]:


from sklearn.model_selection import train_test_split


# In[34]:


X_train, X_val, y_train, y_val= train_test_split(X,y,test_size=0.2, random_state=13)


# In[35]:


from sklearn.ensemble import RandomForestClassifier


# In[36]:


df_train[selected_columns]


# In[37]:


from sklearn.ensemble import AdaBoostClassifier


# In[38]:


model = AdaBoostClassifier()
model.fit(X_train,y_train)
model.score(X_train,y_train), model.score(X_val,y_val)


# In[39]:


from sklearn.neighbors import KNeighborsClassifier


# In[49]:


model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train,y_train)
model_knn.score(X_train,y_train), model_knn.score(X_val,y_val)


# In[41]:


from sklearn.svm import SVC


# In[42]:


model_poly = SVC(kernel='poly')
model_poly.fit(X_train,y_train)
model_poly.score(X_train,y_train), model_poly.score(X_val,y_val)


# In[43]:


model_lin = SVC(kernel='linear')
model_lin.fit(X_train,y_train)
model_lin.score(X_train,y_train), model_lin.score(X_val,y_val)


# In[44]:


model_rbf = SVC(kernel='rbf')
model_rbf.fit(X_train,y_train)
model_rbf.score(X_train,y_train), model_rbf.score(X_val,y_val)


# In[45]:


from sklearn.tree import DecisionTreeClassifier


# In[53]:


model_dt = DecisionTreeClassifier(random_state=13)
model_dt.fit(X_train,y_train)
model_dt.score(X_train,y_train), model_dt.score(X_val,y_val)


# In[47]:


from sklearn.linear_model import LogisticRegression


# In[56]:


model = AdaBoostClassifier(base_estimator=LogisticRegression(random_state=13,n_jobs=-1))
model.fit(X_train,y_train)
model.score(X_train,y_train), model.score(X_val,y_val)


# # Hyperparameter tuning

# In[59]:


params={'n_estimators':[10,25,50,75,100],
       'min_samples_leaf':[2,3,4,5,6],
        'max_depth':[2,3,4,5,6],
        'min_samples_split':[3,4,5,6,7]}


# In[60]:


from sklearn.model_selection import GridSearchCV


# In[94]:


grid_cv=GridSearchCV(RandomForestClassifier(),params,cv=5,verbose=3,n_jobs=-1)


# In[95]:


grid_cv.fit(X_train,y_train)


# In[96]:


grid_cv.best_estimator_


# In[97]:


model_rf=RandomForestClassifier(max_depth=5, min_samples_leaf=2, min_samples_split=7, n_estimators=50)
model_rf.fit(X_train,y_train)
model_rf.score(X_train,y_train), model_rf.score(X_val,y_val),model_rf.score(X,y)


# In[65]:


# Hyper parameter tuning using GridSearchCV in DT
dict1 = {'min_samples_leaf':[2,3,4,5,6,7], 
         'max_depth': [2,3,4,5,6,7],
         'min_samples_split': [3,4,5,6,7,8],
         'criterion':['gini','entropy']}


# In[66]:


grid_cv = GridSearchCV(DecisionTreeClassifier(), dict1, cv=5, verbose=3)


# In[67]:


grid_cv.fit(X_train, y_train)


# In[68]:


grid_cv.best_estimator_


# In[71]:


model1_dt=DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_leaf=6,
                       min_samples_split=4)
model1_dt.fit(X_train,y_train)
model1_dt.score(X_train,y_train),model1_dt.score(X_val,y_val),model1_dt.score(X,y)


# # Use final model to make predictions on test data(df_test)

# In[72]:


df_test = pd.read_csv('test.csv')


# In[73]:


df_test.columns


# In[74]:


df_test['Fare'].fillna(df_test['Fare'].mean(), inplace=True)


# In[75]:


df_test['Age'].fillna(df_test['Age'].mean(), inplace=True)


# In[76]:


df_test['Age_Cat']=df_test['Age'].apply(get_category)


# In[77]:


df_test['Age_Cat'].value_counts()


# In[78]:


df_test['inCabin'] = (~df_test['Cabin'].isna()).astype(int)


# In[79]:


df_test.info()


# In[80]:


temp_2=pd.get_dummies(df_test['Sex'], drop_first=True)


# In[81]:


temp_1 = pd.get_dummies(df_test['Embarked'], drop_first=True, prefix='Embarked')


# In[82]:


df_test=pd.concat([df_test,temp_1],axis=1)


# In[83]:


df_test=pd.concat([df_test,temp_2],axis=1)


# In[ ]:


#df_test['Embarked'] = df_test['Embarked'].map({'S':0, 'C':1, 'Q':2})


# In[ ]:


#df_test['Sex'] = df_test['Sex'].map({'male':0, 'female':1})


# In[84]:


df_test.columns


# In[85]:


#Select columns to test model
selected_columns = list(df_test.columns)
#selected_columns.remove('Survived')
selected_columns.remove('PassengerId')
selected_columns.remove('Name')
selected_columns.remove('Ticket')
selected_columns.remove('Cabin')
#selected_columns.remove('Fare')
selected_columns.remove('Age')
selected_columns.remove('Embarked')
selected_columns.remove('Sex')
selected_columns.remove('Embarked_Q')
#selected_columns.remove('Fare')
#selected_columns.remove('Title')
selected_columns


# In[86]:


df_test[selected_columns].info()


# In[87]:


# Label Encoding
df_test['Age_Cat']=df_test['Age_Cat'].map({'CHILD':0, 'YOUNG':1, 'OLDER':2})


# In[88]:


X_test = df_test[selected_columns]


# In[98]:


yp = model_rf.predict(X_test)


# In[99]:


df_test['Survived'] = yp


# In[100]:


df_test[['PassengerId','Survived']].to_csv('first1_rf.csv', index = False)


# In[ ]:


yp = model.predict(X_test)


# In[ ]:


df_test['Survived'] = yp


# In[ ]:


df_test[['PassengerId','Survived']].to_csv('first2_rf.csv', index = False)


# In[ ]:




