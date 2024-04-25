#!/usr/bin/env python
# coding: utf-8

# # Unsupervised Lab Session

# ## Learning outcomes:
# - Exploratory data analysis and data preparation for model building.
# - PCA for dimensionality reduction.
# - K-means and Agglomerative Clustering

# ## Problem Statement
# Based on the given marketing campigan dataset, segment the similar customers into suitable clusters. Analyze the clusters and provide your insights to help the organization promote their business.

# ## Context:
# - Customer Personality Analysis is a detailed analysis of a company’s ideal customers. It helps a business to better understand its customers and makes it easier for them to modify products according to the specific needs, behaviors and concerns of different types of customers.
# - Customer personality analysis helps a business to modify its product based on its target customers from different types of customer segments. For example, instead of spending money to market a new product to every customer in the company’s database, a company can analyze which customer segment is most likely to buy the product and then market the product only on that particular segment.

# ## About dataset
# - Source: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis?datasetId=1546318&sortBy=voteCount
# 
# ### Attribute Information:
# - ID: Customer's unique identifier
# - Year_Birth: Customer's birth year
# - Education: Customer's education level
# - Marital_Status: Customer's marital status
# - Income: Customer's yearly household income
# - Kidhome: Number of children in customer's household
# - Teenhome: Number of teenagers in customer's household
# - Dt_Customer: Date of customer's enrollment with the company
# - Recency: Number of days since customer's last purchase
# - Complain: 1 if the customer complained in the last 2 years, 0 otherwise
# - MntWines: Amount spent on wine in last 2 years
# - MntFruits: Amount spent on fruits in last 2 years
# - MntMeatProducts: Amount spent on meat in last 2 years
# - MntFishProducts: Amount spent on fish in last 2 years
# - MntSweetProducts: Amount spent on sweets in last 2 years
# - MntGoldProds: Amount spent on gold in last 2 years
# - NumDealsPurchases: Number of purchases made with a discount
# - AcceptedCmp1: 1 if customer accepted the offer in the 1st campaign, 0 otherwise
# - AcceptedCmp2: 1 if customer accepted the offer in the 2nd campaign, 0 otherwise
# - AcceptedCmp3: 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
# - AcceptedCmp4: 1 if customer accepted the offer in the 4th campaign, 0 otherwise
# - AcceptedCmp5: 1 if customer accepted the offer in the 5th campaign, 0 otherwise
# - Response: 1 if customer accepted the offer in the last campaign, 0 otherwise
# - NumWebPurchases: Number of purchases made through the company’s website
# - NumCatalogPurchases: Number of purchases made using a catalogue
# - NumStorePurchases: Number of purchases made directly in stores
# - NumWebVisitsMonth: Number of visits to company’s website in the last month

# ### 1. Import required libraries

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from sklearn.cluster import KMeans
from scipy.stats import zscore
from scipy.spatial import distance
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA


# ### 2. Load the CSV file (i.e marketing.csv) and display the first 5 rows of the dataframe. Check the shape and info of the dataset.

# In[7]:


df = pd.read_csv('C:\\Users\\PURNANGSHU ROY\\OneDrive\\Desktop\\Data\\marketing.csv')
df.head()


# In[158]:


# Basic information of the dataset
df.info()


# In[159]:


# Shape of the dataset (Customers, Columns)
df.shape


# ### 3. Check the percentage of missing values? If there is presence of missing values, treat them accordingly.

# In[160]:


# Missing values
df.isnull().sum()/len(df)*100


# In[161]:


#Treating missing values
df['Income'] = df['Income'].fillna(df['Income'].mean())
#checking changed
df.isnull().sum()


# ### 4. Check if there are any duplicate records in the dataset? If any drop them.

# In[162]:


df.duplicated().sum()


# 0 duplicates

# ### 5. Drop the columns which you think redundant for the analysis 

# In[163]:


df = df.drop(columns = ['ID', 'Dt_Customer'], axis = 1)


# ### 6. Check the unique categories in the column 'Marital_Status'
# - i) Group categories 'Married', 'Together' as 'relationship'
# - ii) Group categories 'Divorced', 'Widow', 'Alone', 'YOLO', and 'Absurd' as 'Single'.

# In[164]:


# Columns and count before grouping
df['Marital_Status'].value_counts()


# In[165]:


# Grouping categories 'Married', 'Together' as 'relationship'.
df['Marital_Status'] = df['Marital_Status'].replace(['Married','Together'],'relationship')

# Grouping categories 'Divorced', 'Widow', 'Alone', 'YOLO', and 'Absurd' as 'Single'.
df['Marital_Status'] = df['Marital_Status'].replace(['Divorced','Widow','Alone', 'YOLO','Absurd'],'Single')


# In[166]:


#After grouping , the categories and the count
df['Marital_Status'].value_counts()


# ### 7. Group the columns 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', and 'MntGoldProds' as 'Total_Expenses'

# In[167]:


#grouping the mentioned columns as Total_Expenses
df['Total_Expenses'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']
df.head(3)


# ### 8. Group the columns 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', and 'NumDealsPurchases' as 'Num_Total_Purchases'

# In[168]:


#grouping the mentioned columns as Num_Total_Purchases
df['Num_Total_Purchases'] = df['NumWebPurchases']+ df['NumCatalogPurchases']+ df['NumStorePurchases']+ df['NumDealsPurchases']
df.head(3)


# ### 9. Group the columns 'Kidhome' and 'Teenhome' as 'Kids'

# In[169]:


#grouping Kidhome and Teenhome as Kids
df['Kids'] = df['Kidhome']+ df['Teenhome']
df.head(3)


# ### 10. Group columns 'AcceptedCmp1 , 2 , 3 , 4, 5' and 'Response' as 'TotalAcceptedCmp'

# In[170]:


#grouping mentioned columns as TotalAcceptedCmp
df['TotalAcceptedCmp'] = df['AcceptedCmp1']+ df['AcceptedCmp2']+ df['AcceptedCmp3']+ df['AcceptedCmp4']+ df['AcceptedCmp5']+ df['Response'] 
df.head(3)


# ### 11. Drop those columns which we have used above for obtaining new features

# In[171]:


# Dropping in use columns
col_del = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumDealsPurchases', 'Kidhome', 'Teenhome', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response' ]
df = df.drop(columns= col_del, axis=1)
df.head(3)


# In[172]:


#After dropping , the new shape of dataset
df.shape


# Initially we had 27 columns(shape- 2240, 27) now we have 11 columns(shape- 2240, 11).

# ### 12. Extract 'age' using the column 'Year_Birth' and then drop the column 'Year_birth'

# In[173]:


# Extracting 'age
df['Age'] = 2024 - df['Year_Birth']


# In[174]:


# dropping column year_birth
df = df.drop('Year_Birth', axis = 1)
df.head()


# In[175]:


# Copy of the dataset
dfc = df.copy()


# ### 13. Encode the categorical variables in the dataset

# In[176]:


# Now Label Encoding the categorical variables
cat_cols = ['Education', 'Marital_Status']
le = LabelEncoder()
for i in cat_cols:
    df[i] = df[[i]].apply(le.fit_transform)

df.head(3)


# ### 14. Standardize the columns, so that values are in a particular range

# In[177]:


df1 = df.copy()
ss = StandardScaler()

scaled_features = ss.fit_transform(df1.values)
scaled_features_df = pd.DataFrame(scaled_features, index=df1.index, columns= df1.columns)

scaled_features_df.head(5)


# ### 15. Apply PCA on the above dataset and determine the number of PCA components to be used so that 90-95% of the variance in data is explained by the same.

# In[178]:


# Covariance Matrix
cov_matrix = np.cov(scaled_features.T)

# Eigen values and Eigen vectors
eig_vals, eig_vectors = np.linalg.eig(cov_matrix)

# Scree Plot
total = sum(eig_vals)
var_exp = [(i/total)*100 for i in sorted(eig_vals, reverse = True)]
cum_var_exp = np.cumsum(var_exp)

# Diagram
plt.bar(range(11), var_exp)
plt.step(range(11), cum_var_exp)
plt.xlabel("Principal components")
plt.ylabel("Explained variance ratio") 


# ### 16. Apply K-means clustering and segment the data (Use PCA transformed data for clustering)

# In[179]:


pca = PCA(n_components = 8)
pca_df = pd.DataFrame(pca.fit_transform(scaled_features_df), columns=['PC1', 'PC2', 'PC3', 'PC4','PC5','PC6','PC7','PC8'])
pca_df.head()


# In[180]:


# Find optimal k using elbow plot
cluster_errors = []
cluster_range = range(2,15)

for num_clusters in cluster_range:
    cluster = KMeans(num_clusters, random_state=100)
    cluster.fit(pca_df)
    cluster_errors.append(cluster.inertia_)

cluster_df = pd.DataFrame({'num_clusters': cluster_range, 'cluster_errors': cluster_errors})

# Elbow plot.
plt.figure(figsize=[10,6])
plt.plot(cluster_df['num_clusters'], cluster_df['cluster_errors'], marker='o')


# In[181]:


# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=100)
kmeans.fit(pca_df)


# In[182]:


# Check the cluster labels
label = pd.DataFrame(kmeans.labels_, columns=['Label'])
kmeans_df = pca_df.join(label)
kmeans_df.head()


# In[183]:


kmeans_df['Label'].value_counts()


# In[184]:


sns.scatterplot(data=kmeans_df, x='PC1', y='PC2', hue='Label', palette='viridis')


# ### 17. Apply Agglomerative clustering and segment the data (Use Original data for clustering), and perform cluster analysis by doing bivariate analysis between the cluster label and different features and write your observations.

# In[185]:


plt.figure(figsize=[18,6])
merg = linkage(scaled_features, method='ward')
dendrogram(merg, leaf_rotation = 90)
plt.xlabel('Datapoints')
plt.ylabel('Euclidean distance')
plt.show()


# In[186]:


from sklearn.metrics import silhouette_score


# In[187]:


for i in range(2,25):
    hier = AgglomerativeClustering(n_clusters=i)
    hier = hier.fit(scaled_features_df)
    labels = hier.fit_predict(scaled_features_df)
    print(i, silhouette_score(scaled_features_df, labels))


# In[188]:


hie_cluster = AgglomerativeClustering(n_clusters= 3, linkage='ward')
hie_cluster_model = hie_cluster.fit(scaled_features_df)


# In[189]:


df_label1 = pd.DataFrame(hie_cluster_model.labels_,columns=['Labels'])
df_label1.head(5)


# In[190]:


# joining the label dataframe
df_hier = dfc.join(df_label1)
df_hier.head()


# ### Visualization and Interpretation of results

# In[191]:


sns.barplot(x=df_hier['Labels'], y=df_hier['Total_Expenses'])
plt.show()


# The Total_Expenses is much higher for cluster 0 compared to the clusters 1 and 2

# In[192]:


sns.barplot(x=df_hier['Labels'], y=df_hier['Income'])
plt.show()


# Where as the Income is also higher for the cluster 0 followed by cluster 2

# In[193]:


sns.countplot(x='Marital_Status', hue='Labels', data=df_hier)
plt.show()


# We can observe that the most of the customers who are in a relationship falls under cluster 0

# In[194]:


sns.barplot(x=df_hier['Labels'], y=df_hier['Num_Total_Purchases'])
plt.show()


# The Total number of purchases is also much higher for customers from cluster 0 compared to cluster 1 and 2

# ## Conclusion:

# 1. Among the different clusters, cluster 0 has customers who spend the most and have the highest income levels, indicating the highest levels of purchasing activity.
# 
# 2. Customers belonging to cluster 1 have the lowest total expenses, minimum account balances, and make the fewest purchases compared to other clusters.
# 
# 3. Customers in cluster 2 have average income levels and exhibit typical purchasing behavior, falling between the extreme behaviors observed in clusters 0 and 1.

# -----
# ## Happy Learning
# -----
