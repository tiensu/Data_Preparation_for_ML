# test classification dataset
from sklearn.datasets import make_classification
# define dataset
X, y = make_classification(n_samples=1000, n_features=100, n_informative=60, n_redundant=40, random_state=7, n_classes=10)
# summarize the dataset
print(X.shape, y.shape)