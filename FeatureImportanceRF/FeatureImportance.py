import pandas as pd  
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np

X = pd.read_csv("./train.csv")
print(X.head())


print(X.dtypes)

y = X.Loan_Status
X = X.drop(['Loan_ID','Loan_Status'], axis=1)
X = X.fillna(-999)

train = X.copy()

for ctype in train.columns[X.dtypes == 'object']:
	if( ctype != 'Loan_ID'):
		X[ctype] = X[ctype].factorize()[0];

print(X.columns)
print(X.head)

rf = RandomForestClassifier()
rf.fit(X,y)

plt.plot(rf.feature_importances_)
plt.xticks(np.arange(X.shape[1]), X.columns.tolist() , rotation=80)
plt.show()


