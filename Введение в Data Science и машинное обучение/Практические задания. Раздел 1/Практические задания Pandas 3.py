import pandas as pd

# my_stat = pd.read_csv('my_stat.csv')
my_stat = pd.read_csv('https://stepik.org/media/attachments/course/4852/my_stat.csv')


subset_1 = my_stat[(my_stat['V1']>0) & (my_stat['V3']=='A')]
subset_2 = my_stat[(my_stat['V2']!=10) | (my_stat['V4']>=1)]


print(subset_1)
print(subset_2)

