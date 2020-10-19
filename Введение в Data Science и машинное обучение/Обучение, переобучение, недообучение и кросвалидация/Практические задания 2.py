import os
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
 
# path1 = os.path.dirname(r'C:\Users\Носков Алексей\Desktop\Программинг\Stepik\Введение в Data Science и машинное обучение\Обучение, переобучение, недообучение и кросвалидация/dataset_209691_15(8).txt')
path = r'C:\Users\Носков Алексей\Desktop\Программинг\Stepik\Введение в Data Science и машинное обучение\Обучение, переобучение, недообучение и кросвалидация/dataset_209691_15.txt'
dogs_n_cats_data = pd.read_csv('https://stepik.org/media/attachments/course/4852/dogs_n_cats.csv')
test_dogs_n_cats_data = pd.read_json(path)

print(dogs_n_cats_data.isnull().sum())

print(dogs_n_cats_data) 

# print(dogs_n_cats_data.eq(Вид == 'собачка'.sum())

print(dogs_n_cats_data.eq(pd.Series(['собачка'], index=['Вид'])).sum())
print(dogs_n_cats_data.eq(pd.Series(['котик'], index=['Вид'])).sum())
print(dogs_n_cats_data.eq(pd.Series([0], index=['Лазает по деревьям'])).sum())
print(dogs_n_cats_data.eq(pd.Series([1], index=['Лазает по деревьям'])).sum())

X_train = dogs_n_cats_data.drop(['Вид'], axis=1)
y_train = dogs_n_cats_data.Вид
X_test = test_dogs_n_cats_data
y_test = test_dogs_n_cats_data

print(X_train)
# print(X_test)
print(y_train)
# print(y_test)

max_depth_values = range(1, 100)

scores_data = pd.DataFrame()
np.random.seed(0)
for max_depth in max_depth_values:

    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    # test_score = clf.score(X_test, y_test)
    temp_score_data =  pd.DataFrame({'max_depth':[max_depth], 
                                     'train_score':[train_score]
                                     # , 'test_score':[test_score]
                                     })
    scores_data = scores_data.append(temp_score_data)

print(scores_data)
print(scores_data.head(15))

scores_data_long = pd.melt(scores_data, id_vars=['max_depth'], 
                           value_vars=['train_score'
                           # , 'test_score'
                           ], 
                           var_name='set_type', value_name='score')                       

sns.lineplot(x='max_depth', y='score', hue='set_type', data=scores_data_long)

print(clf.predict(X_test))

print(np.sum(clf.predict(X_test)=='собачка'))
print(np.sum(clf.predict(X_test)=='котик'))

tree.plot_tree(clf)



