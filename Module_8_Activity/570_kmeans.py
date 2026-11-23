# Import needed packages
# Your code here
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

healthy = pd.read_csv("healthy_lifestyle.csv")

# Input the number of clusters
number = int(input())

# Define input features
X = healthy[["sunshine_hours", "happiness_levels"]]

# Use StandardScaler() to standardize input features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=["sunshine_hours", "happiness_levels"])
X = X.dropna()

# Initialize a k-means clustering model named kmeansHealthy with a user-defined number of clusters, 
# init="random", n_init=10, random_state=123, and algorithm="elkan"
# Your code here
kmeansHealthy = KMeans(n_clusters = number, init = 'random', n_init = 10, random_state = 123, algorithm = 'elkan')

# Fit the algorithm to the input features
# Your code here
kmeansHealthy = kmeansHealthy.fit(X)

# Find and print the cluster centroids
centroid = kmeansHealthy.cluster_centers_ # Your code here
print("Centroids:", np.round(centroid,4))

# Find and print the cluster inertia
inertia = kmeansHealthy.inertia_ # Your code here
print("Inertia:", np.round(inertia,4))