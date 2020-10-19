import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

train_iris_data = pd.read_csv('https://stepik.org/media/attachments/course/4852/train_iris.csv')
test_iris_data = pd.read_csv('https://stepik.org/media/attachments/course/4852/test_iris.csv')

print(train_iris_data) 
print(test_iris_data) 

train_iris_data.isnull().sum()

X_train = train_iris_data.drop(['species', 'Unnamed: 0'], axis=1)
y_train = train_iris_data.species

X_test = test_iris_data.drop(['species', 'Unnamed: 0'], axis=1)
y_test = test_iris_data.species


print(X_train)
print(X_test)
print(y_train)
print(y_test)    
    
max_depth_values = range(1, 100)

scores_data = pd.DataFrame()
np.random.seed(0)
for max_depth in max_depth_values:

    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    temp_score_data =  pd.DataFrame({'max_depth':[max_depth], 
                                     'train_score':[train_score],
                                     'test_score':[test_score]})
    scores_data = scores_data.append(temp_score_data)

print(scores_data)
print(scores_data.head(15))

scores_data_long = pd.melt(scores_data, id_vars=['max_depth'], 
                           value_vars=['train_score', 'test_score'], 
                           var_name='set_type', value_name='score')                       

sns.lineplot(x='max_depth', y='score', hue='set_type', data=scores_data_long)






