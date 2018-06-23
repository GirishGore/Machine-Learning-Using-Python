# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 17:32:58 2018

@author: Girish
"""

import pandas as pd
winedata_offers = pd.read_excel("E:\\EDrive\\Python\\Machine Learning with Python Tutorials\\Unsupervised Learning\\WineKMC.xlsx" , sheet_name=0)
winedata_transactions = pd.read_excel("E:\\EDrive\\Python\\Machine Learning with Python Tutorials\\Unsupervised Learning\\WineKMC.xlsx" , sheet_name=1)

winedata_offers.head()
## barietal : made from or belonging to a single specified variety of grape.
winedata_transactions.head()

winedata_offers.columns= ["offer_id", "campaign", "varietal", "min_qty", "discount", "origin", "past_peak"]
winedata_transactions.columns = ["customer_name", "offer_id"]

# join the offers and transactions table
df = pd.merge(winedata_offers, winedata_transactions)

df.head()
df.isnull().sum()
df.columns

winedata_offers.shape
winedata_transactions.shape
df.shape

# create a "pivot table" which will give us the number of times each customer responded to a given offer
matrix = df.pivot_table(index=['customer_name'], columns=['offer_id'], aggfunc='count')

matrix.head()
matrix.columns
matrix.index.name
matrix.index
matrix.shape

print(matrix.columns.value_counts())

winedata_offers.campaign.value_counts()
winedata_transactions.customer_name.value_counts()

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
customer_clusters.head()
customer_clusters.columns

from ggplot import *
ggplot(customer_clusters, aes(x='x', y='y', color='clusterid')) + geom_point(size=75) + ggtitle("Customers Grouped by Cluster")

df = pd.merge(df , customer_clusters , on='customer_name')
df.columns.values
df.columns.values[8] = 'clusterid'
df.columns.values[9] = 'x'
df.columns.values[10] = 'y'
df.rename(columns={'clusterid': 'cluster_id'}, inplace=True)
df.info
df.columns[8]


df['1'] = df.cluster_id== 1
cluster1 = df[df['1'] == True]

cluster1.columns

cluster1.min_qty.mean()
df.min_qty.mean()
