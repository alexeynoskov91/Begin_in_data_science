import pandas as pd

# my_stat = pd.read_csv('my_stat.csv')
my_stat = pd.read_csv('https://stepik.org/media/attachments/course/4852/my_stat.csv')

# print(my_stat)

subset_1 = my_stat.iloc[0:10, 0:3].drop(columns=['V2'])
subset_2 = my_stat.drop(index=[0,4], columns=['V1', 'V3'])

print(subset_1)
print(subset_2)

# print(my_stat.describe())

