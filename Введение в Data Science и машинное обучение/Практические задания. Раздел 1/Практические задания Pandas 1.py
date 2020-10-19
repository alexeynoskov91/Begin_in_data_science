import pandas as pd

d = {"type":pd.Series(['A', 'A', 'B', 'B'], index=['01', '02', '03', '04']), "value": pd.Series([10, 14, 12, 23], index=['01', '02', '03', '04'])}

my_data = pd.DataFrame(data=d)


print(my_data) 

