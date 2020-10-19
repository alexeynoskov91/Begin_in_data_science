# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import SVG
from graphviz import Source
from IPython.display import display 
from sklearn import tree
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

titanic_data = pd.read_csv('https://storage.googleapis.com/kagglesdsdata/competitions/3136/26502/train.csv?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1603099598&Signature=MGYWpweqaN%2Fjdzw5S0prDXpknNeowKW31vH3Hp%2FVO6FUpAYCzXyIIp854VD7nfTJNglUygew6cpGSTqaAILugGPDa59qqkMS34pwFvYIaTM95b4g1M9MubvgO5UTLu%2FnsxtwjpVzgHMVGAbnQG%2FiUujMnhDjLzAGO4pXMPNKFH5l83%2BvSkZmjFTG0lvwVeJNzv6SHYGB1P7ssMMSsgjKST3Muh0vH1xJMxLaG5BrJpu0Du7uIC52hyL6LN6NyaH7jdCMhhzsgnJIMWsAag9MKwYI5HqpLj4E7pd4xidJ1meil5XMVnQFkbKHk2788KNK%2BERcQot%2B%2B9Q0yuAPxPWmqg%3D%3D&response-content-disposition=attachment%3B+filename%3Dtrain.csv')

titanic_data.isnull().sum()

X = titanic_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
y = titanic_data.Survived


X = pd.get_dummies(X)

X = X.fillna({'Age':X.Age.median()})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf1 = tree.DecisionTreeClassifier()

clf1.fit(X, y)

print(clf1.score(X_train, y_train))
print(clf1.score(X_test, y_test))

clf1.fit(X_train, y_train)
print(clf1.score(X, y))
print(clf1.score(X_train, y_train))
print(clf1.score(X_test, y_test))
print('____________________________')
clf2 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)

clf2.fit(X, y)

print(clf2.score(X_train, y_train))
print(clf2.score(X_test, y_test))

clf2.fit(X_train, y_train)

print(clf2.score(X, y))
print(clf2.score(X_train, y_train))
print(clf2.score(X_test, y_test))

max_depth_values = range(1, 100)

scores_data1 = pd.DataFrame()

for max_depth in max_depth_values:
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    temp_score = clf.score(X_test, y_test)
    temp_score_data =  pd.DataFrame({'max_depth':[max_depth], 
                                     'train_score':[train_score],
                                     'test_score':[test_score]})
    scores_data1 = scores_data1.append(temp_score_data)                                    
    
    print(clf.score(X_train, y_train))
    print(clf.score(X_test, y_test))    
    print('__________________________________')

print(scores_data1)

# pd.melt(df, id_vars=['A'], value=['B'],
        # var_name='myVarname', value_name='myValname')


scores_data_long1 = pd.melt(scores_data1, id_vars=['max_depth'], 
                           value_vars=['train_score', 'test_score'], 
                           var_name='set_type', value_name='score')
print(scores_data_long1)                           

sns.lineplot(x='max_depth', y='score', hue='set_type', data=scores_data_long1)


clf3 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4)
arr = cross_val_score(clf3, X_train, y_train, cv=5)
print(arr)
print(arr.mean())

scores_data = pd.DataFrame()

for max_depth in max_depth_values:
    clf4 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    clf4.fit(X_train, y_train)
    train_score = clf4.score(X_train, y_train)
    test_score = clf4.score(X_test, y_test)
    temp_score = clf4.score(X_test, y_test)
    
    mean_cross_val_score = cross_val_score(clf4, X_train, y_train, cv=6).mean()
    
    # for i in range(2,100):
        # if cross_val_score(clf4, X_train, y_train, cv=i).mean() > mean_cross_val_score:
            # mean_cross_val_score = cross_val_score(clf4, X_train, y_train, cv=i).mean()
    
    
    
    temp_score_data =  pd.DataFrame({'max_depth':[max_depth], 
                                     'train_score':[train_score],
                                     'test_score':[test_score],
                                     'cross_val_score':[mean_cross_val_score]})
    scores_data = scores_data.append(temp_score_data)                                    
    
    print(clf4.score(X_train, y_train))
    print(clf4.score(X_test, y_test))    
    print('__________________________________')

print(scores_data)

scores_data_long = pd.melt(scores_data, id_vars=['max_depth'], 
                           value_vars=['train_score', 'test_score', 'cross_val_score'], 
                           var_name='set_type', value_name='score')
print(scores_data_long)  

sns.lineplot(x='max_depth', y='score', hue='set_type', data=scores_data_long)


model = LogisticRegression()

