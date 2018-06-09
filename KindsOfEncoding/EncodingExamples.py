import pandas as pd  

X = pd.read_csv("E:/EDrive/Python/Machine Learning with Python Tutorials/KindsOfEncoding/train.csv")
X.describe(include='all')

X.Property_Area.describe()
X.Property_Area.unique()
X.Property_Area.value_counts()
print(X.dtypes)


num_cols = X.dtypes[(X.dtypes == 'float64') | (X.dtypes == 'int64')]
num_data = X[num_cols.index.values]
num_data.hist()

### Normalized x  =    (X - Xmin) / (Xmax - Xmin)
from sklearn.preprocessing import MinMaxScaler
min_max=MinMaxScaler()
num_data_trans = min_max.fit_transform(num_data)
### Checking in any NaN value exists
num_data.isnull().values.any()
### Imputing NA values with proper numbers
num_data.mean()  ## axis = 0 columns default ; axis = 1 rows
num_data = num_data.fillna(num_data.mean())
### Checking in any NaN value exists
num_data.isnull().values.any()

num_data_trans = pd.DataFrame(min_max.fit_transform(num_data))
num_data_trans.hist()


### Basic Label Encoding
Xcopy = X.copy()
for ctype in Xcopy.columns[Xcopy.dtypes == 'object']:
    Xcopy[ctype] = Xcopy[ctype].factorize()[0]

print(Xcopy.columns)
print(Xcopy.head)

Xcopy.Property_Area.value_counts()

XProp = X.Property_Area
type(XProp)

## One label encoding using dummy
print(pd.get_dummies(XProp))

#### Scikit learn for One Label Encoding

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(XProp)
print(integer_encoded)


from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder()
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
binary_encoded = one_hot_encoder.fit_transform(integer_encoded)
print(binary_encoded)
type(binary_encoded)
binary_encoded.shape
one_hot_encoder.n_values_
one_hot_encoder.feature_indices_

df = pd.DataFrame(binary_encoded.toarray())
df.columns = ['',]
