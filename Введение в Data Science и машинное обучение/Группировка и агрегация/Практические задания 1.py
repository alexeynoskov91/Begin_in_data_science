import pandas as pd
import numpy as np

my_stat = pd.read_csv('https://stepik.org/media/attachments/course/4852/dota_hero_stats.csv')


sum(my_stat.legs == 8)
sum(my_stat.legs == 6)
sum(my_stat.legs == 4)
sum(my_stat.legs == 2)
sum(my_stat.legs == 0)

print(sum(my_stat.legs == 8))
print(sum(my_stat.legs == 6))
print(sum(my_stat.legs == 4))
print(sum(my_stat.legs == 2))
print(sum(my_stat.legs == 0))