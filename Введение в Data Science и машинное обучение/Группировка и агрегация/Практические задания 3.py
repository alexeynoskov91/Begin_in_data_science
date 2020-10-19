import pandas as pd
import numpy as np

my_stat = pd.read_csv('https://stepik.org/media/attachments/course/4852/accountancy.csv')


print(my_stat.groupby(['Type', 'Executor']).aggregate({'Salary': 'mean'}).rename(columns={'Salary':'mean_salary'}))

