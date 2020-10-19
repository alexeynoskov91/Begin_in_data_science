from scipy.stats import entropy
import pandas as pd

cat_stat = pd.read_csv('https://stepik.org/media/attachments/course/4852/cats.csv')

# sklearn.metrics.log_loss(y_true, y_pred, *, eps=1e-15, normalize=True, sample_weight=None, labels=None)

# scipy.stats.entropy(pk, qk=None, base=None, axis=0)

E = entropy([2/5,3/5],base=2)
# bark_entropy1 = entropy([4/5,1/5],base=2)

# climbing_entropy0 = entropy([0/4,4/4],base=2)
# climbing_entropy1 = entropy([6/6,0/6],base=2)

# fur_entropy0 = entropy([1/1,0/1],base=2)
# fur_entropy1 = entropy([5/9,4/9],base=2)

print('E:', E)
# print('bark_entropy1:', bark_entropy1)

# print('climbing entropy0:', climbing_entropy0)
# print('climbing entropy1:', climbing_entropy1)  

# print('fur_entropy0:', fur_entropy0)
# print('fur_entropy1:', fur_entropy1)