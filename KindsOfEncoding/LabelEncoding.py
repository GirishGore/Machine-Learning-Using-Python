import pandas as pd  

X = pd.read_csv("E:/EDrive/Python/Machine Learning with Python Tutorials/KindsOfEncoding/train.csv")
X.shape
X.columns
X.describe()
X.head

X.Property_Area.describe()
X.Property_Area.unique()



train = X.copy()
print(X.head())


print(X.dtypes)


for ctype in train.columns[X.dtypes == 'object']:
	X[ctype] = X[ctype].factorize()[0];

print(X.columns)
print(X.head)

#print(X['Property_Area'].factorize())
