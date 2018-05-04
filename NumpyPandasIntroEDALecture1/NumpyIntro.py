# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 18:29:27 2018

@author: Girish
"""

##  NumPy : Numerical Python
##  homogeneous multidimensional array of elements belonging to same type
##  indexed by a tuple of positive numbers called axis

import numpy as np

basicArray = np.array(np.arange(1,10,1,dtype=int).reshape(3,3))
print(basicArray)

print(basicArray.ndim)

print(basicArray.shape)
