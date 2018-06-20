# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 17:32:58 2018

@author: Girish
"""

import pandas as pd
winedata_offers = pd.read_excel("E:\\EDrive\\Python\\Machine Learning with Python Tutorials\\Unsupervised Learning\\WineKMC.xlsx" , sheet_name=0)
winedata_transactions = pd.read_excel("E:\\EDrive\\Python\\Machine Learning with Python Tutorials\\Unsupervised Learning\\WineKMC.xlsx" , sheet_name=1)

winedata_offers.head()
winedata_transactions.head()
winedata_offers.columns= ["offer_id", "campaign", "varietal", "min_qty", "discount", "origin", "past_peak"]
winedata_transactions.columns = ["customer_name", "offer_id"]

# join the offers and transactions table
df = pd.merge(winedata_offers, winedata_transactions)

df.head()
df.columns
# create a "pivot table" which will give us the number of times each customer responded to a given offer
matrix = df.pivot_table(index=['customer_name'], columns=['offer_id'], aggfunc='count')

matrix.head()
matrix.columns
# a little tidying up. fill NA values with 0 and make the index into a column

matrix = matrix.fillna(0).reset_index()
matrix.head()
# save a list of the 0/1 columns. we'll use these a bit later
# save a list of the 0/1 columns. we'll use these a bit later
x_cols = matrix.columns[1:]

from sklearn.cluster import KMeans

cluster = KMeans(n_clusters=5)
matrix['clusterid'] = cluster.fit_predict(matrix[matrix.columns[2:]])
matrix.clusterid.value_counts()


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
matrix['x'] = pca.fit_transform(matrix[x_cols])[:,0]
matrix['y'] = pca.fit_transform(matrix[x_cols])[:,1]
matrix = matrix.reset_index()

customer_clusters = matrix[['customer_name', 'clusterid', 'x', 'y']]

from ggplot import *
ggplot(customer_clusters, aes(x='x', y='y', color='clusterid')) + geom_point(size=75) + ggtitle("Customers Grouped by Cluster")
