""" Clustering in Python"""

""" Chapter 1 - Introduction to Clustering"""
# A - Intro
# Common unsupervised learning algorithms:
#  - Clustering
#  - Neural networks
#  - Anomaly detection

# Clustering - grouping items with similar characteristics
# Items in groups more similar to each other than in other groups

# B - Basics of cluster analysis
# Clustering algorithms:
#  - Hierarchical clustering
#  - K means clustering
# Other clustering algorithms - DBSCAN, Gaussian Methods

# Hierarchical clustering in SciPy
from scipy.cluster.hierarchy import linkage, fcluster
from matplotlib import pyplot as plt
import seaborn as sns, pandas as pd

x_coordinates = [80.1, 93.1, 86.6, 98.5, 86.4, 9.5, 15.2, 3.4,
                 10.4, 20.3, 44.2, 56.8, 49.2, 62.5, 44.0]
y_coordinates = [87.2, 96.1, 95.6, 92.4, 92.4, 57.7, 49.4,
                 47.3, 59.1, 55.5, 25.6, 2.1, 10.9, 24.1, 10.3]

df = pd.DataFrame({'x_coordinate' : x_coordinates,
                   'y_coordinate' : y_coordinates})

Z = linkage(df, 'ward')  # Computes distances between intermediate clusters
df['cluster_labels'] = fcluster(Z, 3, criterion='maxclust')  # Generates clusters and assigns associated cluster labels

sns.scatterplot(x='x_coordinate', y='y_coordinate', hue='cluster_labels', data=df)
#plt.show()

# K Means
# Based on random cluster centre generated for each cluster
# Distance to cluster centres computed for each point and assigned to closest cluster
# Recompute cluster centre
# Performed a pre-defined number of times

# K-Means clustering in SciPy
from scipy.cluster.vq import kmeans, vq
from matplotlib import pyplot as plt
import seaborn as sns, pandas as pd

import random
random.seed((1000,2000))

centroids,_ = kmeans(df, 3)  # Computes centroids of clusters
df['cluster_labels'], _ = vq(df, centroids)  # Cluster assignments for each point
# Second argument in both methods is distortion - captured in dummy variable

sns.scatterplot(x='x_coordinate', y='y_coordinate',
                hue='cluster_labels', data=df)
#plt.show()

# C: Data preparation for cluster analysis

# Why do we need to prepare data for clustering?
# Variables have incomparable units, e.g. product dimensions and price
# Variables with same units have vastly different scales and variances (expenditures on cereals, travel)
# Data in raw form may lead to bias in clustering
# Cluster may be heavily dependent on one variable
# Solution - normalisation of individual variables

# Normalisation
# Rescale values to a std dev of 1

# x_new = x / std_dev(x)

from scipy.cluster.vq import whiten
data = [5, 1, 3, 3, 2, 3, 3, 8, 1, 2, 2, 3, 5]
# Can be 1D or multi-D

scaled_data = whiten(data)
print(scaled_data)

# Illustration - normalization of data
# Import plotting library
from matplotlib import pyplot as plt

# Initialise original, scaled data
plt.plot(data, label='original')
plt.plot(scaled_data, label='scaled')
plt.legend
plt.show()

# C: Basics of hierarchical clustering
# Creating a distance matrix using linkage
# Computes the distances between clusters
scipy.cluster.linkage(observations,
                      method='single',  # How to calc proximity bet two cluster centres
                      metric='euclidean',  # How to calc distance
                      optimal_ordering='False')  # Changes the order of linkage matrices

# Which method to use?
# Single - based on two closest objects
# Complete - based on two farthest objects
# Average - based on arithmetic mean of all objects
# Centroid - based on the geometric mean of all objects
# Median - based on the median of all objects
# Ward - based on the diff bet sum of squares of the joint clusters, minus the indiv sum squares
#  - Focuses on clusters more concentric towards its centre

# Create cluster labels with fcluster
scipy.cluster.hierarchy.fcluster(distance_matrix,  # Output of linkage() method
                                 num_cluster,  # number of clusters
                                 criterion)  # How to decide thresholds to form clusters - will use max_clust

# In sns - extra label 0 shown, even through no points present
# Can be removed if store cluster labels as strings

# Ward method - clusters more dense towards centre
# Single method - clusters more dispersed
# Complete method - uses two furthest objects - results similar to ward method

# Thoughts on selecting a method
# No right method for all problems
# Need to understand the distribution of data






