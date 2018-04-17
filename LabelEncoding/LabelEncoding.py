import pandas as pd  

X = pd.read_csv("./train.csv")
train = X.copy()
print(X.head())


print(X.dtypes)


for ctype in train.columns[X.dtypes == 'object']:
	X[ctype] = X[ctype].factorize()[0];

print(X.columns)
print(X.head)

#print(X['Property_Area'].factorize())
