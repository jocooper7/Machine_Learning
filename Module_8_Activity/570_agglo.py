import pandas as pd
import seaborn as sns
import numpy as np

# Import needed sklearn packages
# Your code here
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# Import needed scipy packages
# Your code here
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform, pdist

# Silence warning
import warnings
warnings.filterwarnings('ignore')


healthy = pd.read_csv('healthy_lifestyle.csv')
healthy = healthy.dropna(subset = ['sunshine_hours', 'happiness_levels'])

# Input the number of clusters
number = int(input())

# Define input features
X = healthy[['sunshine_hours', 'happiness_levels']]

# Use StandardScaler() to standardize input features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=['sunshine_hours', 'happiness_levels'])

# Initialize and fit an agglomerative clustering model using ward linkage in scikit-learn, with a user-defined
# number of clusters
# Your code here
aggClus = AgglomerativeClustering(n_clusters = number, linkage = 'ward')
aggClus = aggClus.fit(X)

# Add cluster labels to the healthy dataframe
healthy['labels']= aggClus.labels_.astype(str) # Your code here
print(healthy[['sunshine_hours', 'happiness_levels', 'labels']])

# Perform agglomerative clustering using SciPy

# Calculate the distances between all instances
# Your code here
distances = pdist(X)

# Convert the distance matrix to a square matrix
# Your code here
distances = squareform(distances)

# Define a clustering model with ward linkage
clustersHealthyScipy = linkage(distances, method = 'ward') # Your code here

print('First five rows of the linkage matrix from SciPy:\n', np.round(clustersHealthyScipy[:5, :], 0))