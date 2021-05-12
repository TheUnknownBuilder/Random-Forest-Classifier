#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, learning_curve, StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import itertools
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
  
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def TrainingPhase(a, b, c, d):
    print('\nRandom Forest CLassifier\n')
    print('K-Fold Cross-Validation Accuracy: \n')
    model = RandomForestClassifier()
    resultsAccuracy = []
    model.fit(a, b)
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    accuracy_results = model_selection.cross_val_score(model, a,b, cv=kfold, scoring='accuracy')
    #Training the dataset
    accuracyMessage = "%s- %s: %f %s:(%f)" % ("RF", "Mean of the accuracy results", accuracy_results.mean(), "Standard Deviation",accuracy_results.std())
    print(accuracyMessage) 
    
dict_characters = {0: 'Healthy', 1: 'Diabetes'}
        


# In[17]:


dataset = read_csv('C:/Users/Vani/Downloads/Diabetes.csv')
dataset.head(10) #Prints the number of rows like Limit 20


# In[18]:


def plotHistogram(values,label,feature,title):
    sns.set_style("whitegrid")
    plotOne = sns.FacetGrid(values, hue=label,aspect=2)#Sets the width and the labels
    plotOne.map(sns.distplot,feature,kde=True)
    plotOne.set(xlim=(0, values[feature].max()))#Sets the x-axis limit
    plotOne.add_legend()
    plotOne.set_axis_labels(feature, 'Proportion')
    plotOne.fig.suptitle(title)
    plt.show()
plotHistogram(dataset,"Outcome",'Insulin','Insulin vs Diagnosis (Blue = Healthy; Orange = Diabetes)')
plotHistogram(dataset,"Outcome",'SkinThickness','SkinThickness vs Diagnosis (Blue = Healthy; Orange = Diabetes)')
#Studied but doubt about feature


# In[19]:


dataset2 = dataset.iloc[:, :-1] #COnsiders the columns till the second last- last being the outcome is not needed.
print("Number of Rows, Number of Columns: ",dataset2.shape)
print("\nColumn Name           Total number of Null Values\n")
print((dataset2[:] == 0).sum())
#Gives the sum of the values which are 0.


# In[20]:


print("Number of Rows, Number of Columns: ",dataset2.shape)
print("\nColumn Name              % Null Values\n")
print(((dataset2[:] == 0).sum())/768*100) #Taking the % of the null values


# In[21]:


mask = np.zeros_like(dataset.corr())
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, g = plt.subplots(figsize=(7, 5))
    g = sns.heatmap(dataset.corr(),cmap="YlGnBu",mask=mask,vmax=.3,square=True, annot=True, linewidths=.5)
    #pearson correlation- default method in dataset.corr()


# In[22]:


dataset.corr()


# In[23]:


data = read_csv('C:/Users/Vani/Downloads/Diabetes.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
imputer = SimpleImputer(missing_values=0, strategy='median')#null values getting replaced by median of the column
X_train2 = imputer.fit_transform(X_train)
X_test2 = imputer.transform(X_test)
X_train3 = pd.DataFrame(X_train2)
plotHistogram(X_train3,None,4,'Insulin vs Diagnosis (Blue = Healthy; Orange = Diabetes)')
plotHistogram(X_train3,None,3,'SkinThickness vs Diagnosis (Blue = Healthy; Orange = Diabetes)')


# In[24]:


labels = {0:'Pregnancies',1:'Glucose',2:'BloodPressure',3:'SkinThickness',4:'Insulin',5:'BMI',6:'DiabetesPedigreeFunction',7:'Age'}
print(labels)
print("\nColumn #, # of Zero Values\n")
print((X_train3[:] == 0).sum())
# data[:] = data[:].replace(0, np.NaN)
# print("\nColumn #, # of Null Values\n")
# print(np.isnan(X_train3).sum()) #Checking if there are any 0 values still there


# In[25]:


TrainingPhase(X_train2, y_train, X_test2, y_test)
clf=RandomForestClassifier()
scores = cross_val_score(clf, X_train2, y_train, cv=5)
print("%20s | Accuracy: %0.2f%% (+/- %0.2f%%)" % ("Random Forest Classifier", 100*scores.mean(), 100*scores.std() * 2))


# In[26]:


def runRandomForest(a, b, c, d):
    model = RandomForestClassifier()
    accuracy_scorer = make_scorer(accuracy_score)
    model.fit(a, b)
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    accuracy = model_selection.cross_val_score(model, a, b, cv=kfold, scoring='accuracy')
    mean = accuracy.mean() 
    stdev = accuracy.std()
    prediction = model.predict(c)
    cnf_matrix = confusion_matrix(d, prediction)
    plot_learning_curve(model, 'Learning Curve For RandomForestClassifier', a, b, (0.60,1.1), 10)
    plt.show()
    plot_confusion_matrix(cnf_matrix, classes=dict_characters,title='Confusion matrix')
    plt.show()
    print('RandomForestClassifier - Training set accuracy: %s (%s)' % (mean, stdev))
    return


runRandomForest(X_train2, y_train, X_test2, y_test)
feature_names1 = X.columns.values


# In[28]:


feature_names = X.columns.values
clf1 = RandomForestClassifier(max_depth=3,min_samples_leaf=12)
clf1.fit(X_train2, y_train)
feature_names = X.columns.values
clf2 = RandomForestClassifier(max_depth=3,min_samples_leaf=12)
clf2.fit(X_train2, y_train)
print('Accuracy of RandomForestClassifier: {:.2f}'.format(clf2.score(X_test2, y_test)))
columns = X.columns
coefficients = clf2.feature_importances_.reshape(X.columns.shape[0], 1)
absCoefficients = abs(coefficients)
fullList = pd.concat((pd.DataFrame(columns, columns = ['Variable']), pd.DataFrame(absCoefficients, columns = ['absCoefficient'])), axis = 1).sort_values(by='absCoefficient', ascending = False)
print('RandomForestClassifier - Feature Importance:')
print('\n',fullList,'\n')


# In[29]:


data = read_csv('C:/Users/Vani/Downloads/Diabetes.csv')
data2 = data.drop(['Pregnancies','BloodPressure','DiabetesPedigreeFunction', 'Age','SkinThickness','Insulin'], axis=1)
X2 = data2.iloc[:, :-1]
y2 = data2.iloc[:, -1]
X_train3, X_test3, y_train3, y_test3 = train_test_split(X2, y2, test_size=0.2, random_state=1)
imputer = SimpleImputer(missing_values=0, strategy='mean')
X_train3 = imputer.fit_transform(X_train3)
X_test3 = imputer.transform(X_test3)
clf4 = RandomForestClassifier()
clf4.fit(X_train3, y_train3)
print('Accuracy of RandomForestClassifier in Reduced Feature Space: {:.2f}'.format(clf4.score(X_test3, y_test3)))
columns = X2.columns
coefficients = clf4.feature_importances_.reshape(X2.columns.shape[0], 1)
absCoefficients = abs(coefficients)
fullList = pd.concat((pd.DataFrame(columns, columns = ['Variable']), pd.DataFrame(absCoefficients, columns = ['absCoefficient'])), axis = 1).sort_values(by='absCoefficient', ascending = False)
print('\nRandomForestClassifier - Feature Importance:')
print('\n',fullList,'\n')

clf3 = RandomForestClassifier()
clf3.fit(X_train2, y_train)
print('\n\nAccuracy of RandomForestClassifier in Full Feature Space: {:.2f}'.format(clf3.score(X_test2, y_test)))
columns = X.columns
coefficients = clf3.feature_importances_.reshape(X.columns.shape[0], 1)
absCoefficients = abs(coefficients)
fullList = pd.concat((pd.DataFrame(columns, columns = ['Variable']), pd.DataFrame(absCoefficients, columns = ['absCoefficient'])), axis = 1).sort_values(by='absCoefficient', ascending = False)
print('RandomForestClassifier - Feature Importance:')
print('\n',fullList,'\n')

