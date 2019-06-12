
# coding: utf-8

# In[1]:



###Import packages
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# In[2]:


tr=pd.read_csv('/Users/swapnilgharat/Desktop/KAGGLE PROJECTS/CONSOLE GAMES/titanic/train.csv')


# In[3]:


##Take a Sneak Peak at the Data.
tr.head()


# In[4]:


#check all column names
for i in tr.columns:
    print (i)


# In[5]:


#check for uneveness in the data and the missing values
tr.count()


# In[6]:


tr.Age.max(),tr.Age.min()


# In[7]:


tr['Survived'].value_counts()


# In[8]:


get_ipython().magic('matplotlib inline')


# In[9]:


alpha_color=0.5


# In[10]:


###Visualzing the male and the femal survivors
tr['Sex'].value_counts().plot(kind='bar',color=['b','r'],alpha=alpha_color)


# In[11]:


##Survivors based on the class
tr['Pclass'].value_counts().plot(kind='bar',color=['b','r','g'],alpha=alpha_color)


# In[12]:


tr.plot(kind='scatter',x='Survived',y='Age')


# In[13]:


##bar graph for the age and the survived people(But this graph is cluttered so we need to bin it )
tr[tr['Survived']==1]['Age'].value_counts().sort_index().plot(kind='bar')


# In[14]:


bins=[0,10,20,30,40,50,60,70,80]
tr['AgeBin']= pd.cut(tr['Age'], bins)


# In[15]:


tr[tr['Survived']==1]['AgeBin'].value_counts().sort_index().plot(kind='bar')


# In[16]:


tr[tr['Survived']==1]['Pclass'].value_counts().sort_index().plot(kind='bar')


# In[17]:


tr[tr['Survived']==1]['Embarked'].value_counts().sort_index().plot(kind='bar')


# In[18]:


binse=['PassengerId','Name','Ticket','Fare','Cabin']


# In[19]:


for i in binse:
    tr=tr.drop(i,inplace=False,axis=1)


# In[20]:


tr.drop('AgeBin',1)


# In[21]:


dummies=['Sex','Embarked']


# In[22]:


for i in dummies:
    dum=pd.get_dummies(tr[i],prefix='new', prefix_sep='_', dummy_na=False)
    tr=tr.drop(i,axis=1,inplace=False)
    tr=pd.concat([tr,dum],axis=1)


# In[23]:


tr=tr.drop('AgeBin',1)


# In[24]:


tr=tr.fillna(method='bfill')


# In[25]:


######tr=tr.drop(tr['Age'].isnull())


# In[26]:


ytr=tr['Survived']


# In[27]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tr, ytr, test_size=0.2, random_state=42)


# In[28]:


dbin=[X_train
,X_train
,X_test
,y_train, y_test]


# In[29]:


X_train[X_train['Age'].isnull()].count()


# In[30]:


X_train.head()


# In[31]:


X_train.shape
X_train['Survived'].value_counts()


# In[32]:


#from sklearn.utils import resample
#major = X_train[X_train['Survived']==0]
#minor = X_train[X_train['Survived']==1]
#upsampled = resample(minor,replace=True,n_samples=176,random_state=123)
#newupsampled = pd.concat([major,upsampled])


# In[33]:


X_train=X_train.drop('Survived',1)


# In[34]:


X_train


# In[35]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
XStdtrain = scaler.fit(X_train).transform(X_train)


# In[36]:


XStdtrain.shape


# In[37]:


y_train.shape


# In[38]:


X_test.shape
y_test.shape


# In[39]:


X_test=X_test.drop('Survived',axis=1)


# In[40]:


XStdtest = scaler.transform(X_test)


# In[41]:


from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


# # SVM:

# In[42]:


from sklearn import svm
from sklearn.svm import SVC
modelsvm = SVC()
modelsvm.fit(XStdtrain,y_train)
ypred = modelsvm.predict(XStdtest)
print ("Training Accuracy")
print (metrics.accuracy_score(y_test, ypred)*100, "%")
print (metrics.classification_report(y_test, ypred))


# In[43]:


tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]}]
modsvm2 = GridSearchCV(modelsvm, cv = 6 ,refit = 'true',param_grid = tuned_parameters)
modsvm2.fit(XStdtrain,y_train)
print(modsvm2.best_params_)


# In[44]:


modsvm3 = SVC(kernel = 'rbf', C = 10,gamma=0.001, probability = True,class_weight = 'balanced')


# In[45]:


modsvm3.fit(XStdtrain,y_train)
yprednew =modsvm3.predict(XStdtest)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, yprednew))


# # RANDOM FOREST:

# In[46]:


from sklearn.ensemble import RandomForestClassifier
seedStart = 2357
modelnowRF=RandomForestClassifier(random_state=seedStart)
modelnowRF.fit(XStdtrain,y_train)


# In[47]:


ypredRF = modelnowRF.predict(XStdtest)
print ("Training Accuracy")
print (metrics.accuracy_score(y_test, ypredRF)*100, "%")
print (metrics.classification_report(y_test, ypredRF))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,ypredRF)


# In[48]:


from sklearn.grid_search import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True)
param_grid = {'n_estimators': [200,700], 'max_features': ['auto', 'sqrt', 'log2']}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 7)
CV_rfc.fit(XStdtrain,y_train)


# In[49]:


print(CV_rfc.best_params_)


# In[50]:


rfcT = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=700, oob_score = True)
rfcT.fit(XStdtrain,y_train)


# In[51]:


ypredrfct = rfcT.predict(XStdtest)
print ("Training Accuracy")
print (metrics.accuracy_score(y_test, ypredrfct)*100, "%")
print (metrics.classification_report(y_test, ypredrfct))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,ypredrfct)


# # BUILDING TEST MODEL
# 

# In[52]:


test=pd.read_csv('/Users/swapnilgharat/Desktop/KAGGLE PROJECTS/CONSOLE GAMES/titanic/test.csv')


# In[53]:


for i in test.columns:
    print (i)


# In[54]:


binse=['Name','Ticket','Fare','Cabin']
for i in binse:
    test=test.drop(i,inplace=False,axis=1)


# In[55]:


test


# In[56]:


dummies=['Sex','Embarked']
for i in dummies:
    dum=pd.get_dummies(test[i],prefix='new', prefix_sep='_', dummy_na=False)
    test=test.drop(i,axis=1,inplace=False)
    test=pd.concat([test,dum],axis=1)


# In[57]:


test=test.fillna(method='ffill')


# In[61]:


test[test['Age'].isnull()].count()


# In[62]:



PassengerId=test['PassengerId']


# In[63]:


PassengerId


# In[64]:


test=test.drop('PassengerId',axis=1)


# In[65]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
XStdtesting = scaler.fit(test).transform(test)


# In[66]:


XStdtesting.shape


# In[67]:


ypredicttest = rfcT.predict(XStdtesting)


# In[68]:


print(len(ypredicttest))


# In[79]:


import numpy as np
import pandas as pd
row_number=np.linspace(0,418,418)
yforpredtest_df2 = pd.DataFrame(row_number)


# In[81]:


yforpredtest_df2['Survived'] = ypredicttest


# In[80]:


yforpredtest_df2['PassengerId']=PassengerId


# In[91]:


df_Final = pd.DataFrame()


# In[92]:


df_Final['PassengerId']=PassengerId
df_Final['Survived']= ypredicttest


# In[94]:


PredictionsDF=df_Final


# In[97]:


PredictionsDF.to_csv('/Users/swapnilgharat/Desktop/KAGGLE PROJECTS/CONSOLE GAMES/titanic/TitanicPredictionsF.csv')

