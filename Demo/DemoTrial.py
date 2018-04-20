import pandas as pd  
import matplotlib.pyplot as plt
import numpy as np

X = pd.read_csv("./train.csv")


#X['TotalIncome'] = X['CoapplicantIncome'] + X['ApplicantIncome']
#X['LoanPerUnitTerm'] = (X['LoanAmount'] * 100)/ X['Loan_Amount_Term']
#X['LoanAmountByTotalIncome'] = X['LoanAmount'] / X['TotalIncome']
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


### Split into Test and Train
### Test and Train Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=31)

### Scaling
from sklearn import preprocessing
    
scaler = preprocessing.StandardScaler().fit(X)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print("Now printing transformed input")


#print(X.head)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=1000,oob_score=True)
rf.fit(X_train,y_train)
print(rf)

plt.plot(rf.feature_importances_)
plt.xticks(np.arange(X_train.shape[1]), X.columns.tolist() , rotation=10)
plt.show()



from sklearn.metrics import accuracy_score

predicted = rf.predict(X_test)
accuracy = accuracy_score(y_test, predicted)
print(accuracy)
print(f'Out-of-bag score estimate: {rf.oob_score_:.3}')
#print(X_train)
#print(X_test)

## For Random Forest we dont need any splitting

from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = pd.DataFrame(confusion_matrix(y_test, predicted), columns=["Loan_Status_Y", "Loan_Status_N"] , index=["Loan_Status_Y", "Loan_Status_N"])
print(cm)
sns.heatmap(cm, annot=True,cmap='Blues', fmt='g')
plt.show()

# n_estimators : The number of trees in the forest.
# criterion : 'gini' or 'entropy' for information gain
# max_features : number of features to be considered
                        # If int, then consider max_features features at each split.
						# If float, then max_features is a percentage and int(max_features * n_features) features are considered at each split.
						# If “auto”, then max_features=sqrt(n_features).
						# If “sqrt”, then max_features=sqrt(n_features) (same as “auto”).
						# If “log2”, then max_features=log2(n_features).
						# If None, then max_features=n_features
# max_depth : The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.