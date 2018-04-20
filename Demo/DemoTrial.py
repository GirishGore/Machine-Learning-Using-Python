import pandas as pd  
import matplotlib.pyplot as plt
import numpy as np

X = pd.read_csv("./train.csv")

X['TotalIncome'] = X['CoapplicantIncome'] + X['ApplicantIncome']
X['LoanToIncome'] = (X['LoanAmount'] / X['Loan_Amount_Term'])*X['TotalIncome']
#print(X.head())
#print(X.dtypes)

y = X.Loan_Status

### Drop Loan_ID from the import
X = X.drop(['Loan_ID','Loan_Status'], axis=1)

## Fill all NA values with a constant like -999
X = X.fillna(-999)

train = X.copy()

for ctype in train.columns[train.dtypes == 'object']:
		X[ctype] = X[ctype].factorize()[0];

#print(X.head)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X,y)
print(rf)

plt.plot(rf.feature_importances_)
plt.xticks(np.arange(X.shape[1]), X.columns.tolist() , rotation=10)
plt.show()


### Test and Train Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=31)

### Scaling
from sklearn import preprocessing
    
scaler = preprocessing.StandardScaler().fit(X)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print("Now printing transformed input")
#print(X_train)
#print(X_test)

## For Random Forest we dont need any splitting

