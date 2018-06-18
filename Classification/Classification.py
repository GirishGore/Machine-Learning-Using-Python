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

from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(random_state=42)
sgd.fit(X_train , y_train)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train , y_train)

from sklearn.model_selection import cross_val_score
cross_val_score(logreg , X_train , y_train , cv = 3 , scoring="accuracy")
