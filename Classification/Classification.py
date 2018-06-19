# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 21:38:39 2018

@author: Girish
"""

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

mnist
type(mnist)
mnist.DESCR
X , y  = mnist.data , mnist.target

X.shape
y.shape
### 28 by 28 pixel image comes to 784

import matplotlib.pyplot as plt
import matplotlib
plt.imshow(X[28812].reshape(28,28) , matplotlib.cm.binary , interpolation='nearest')
y[28812]

X_train , y_train , X_test , y_test = X[:60000] , y[:60000] , X[60000:] , y[60000:]



from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state=42,solver = 'lbfgs')
#logreg.fit(X_train , y_train)

from sklearn.model_selection import cross_val_score
cross_val_score(logreg , X_train , y_train , cv = 5 , scoring="accuracy")

from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(random_state=42)
# sgd.fit(X_train , y_train)
cross_val_score(sgd , X_train , y_train , cv = 5 , scoring="accuracy")

from sklearn import svm
clf = svm.SVC()
cross_val_score(sgd , X_train , y_train , cv = 5 , scoring="accuracy")

from sklearn.model_selection import cross_val_predict
predictTrain = cross_val_predict(sgd , X_train , y_train , cv = 10)


from sklearn.metrics import confusion_matrix , precision_score , recall_score , f1_score
confusion_matrix(y_train , predictTrain)
precision_score(y_train , predictTrain , average='micro')
precision_score(y_train , predictTrain , average='macro')
recall_score(y_train , predictTrain , average='macro')
f1_score(y_train , predictTrain , average='macro')
### Precision = TP / (TP + FP)
### Recall(Sensitivity)  = TP / (TP + FN)

## F1 score is harmonic mean of Precision and Recall
## F1 favlurs classifiers with similar precision and recall

## Videos favourable for kids :   High Precision , Low Recall
## Alarm for shoplifter :  Low Precision , High Recall




import numpy as np
y_train9 = (y_train == 5)
X_train9 = X_train

y_train9

predictTrainDF = cross_val_predict(logreg , X_train , y_train9 , cv = 5 , method="decision_function")

from sklearn.metrics import precision_recall_curve

prec , recl , thresh = precision_recall_curve(y_train9 , predictTrainDF)
import matplotlib.pyplot as plt

plt.plot(thresh,prec[:-1], "b--",label="Precison")
plt.plot(thresh,recl[:-1], "g-",label="Recall")
plt.xlabel("Threshold")
plt.ylim([0,1])
plt.show()
