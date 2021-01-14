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

""" Chapter 2 - Hierarchical Clustering"""
# A: Basics of hierarchical clustering
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

# B: Visualize clusters
# Helps to make sense of the clusters formed
# Additional step of validation
# Helps spot trends in data

# Visualize using matplotlib
from matplotlib import pyplot as plt

df = pd.DataFrame({'x': [2, 3, 5, 6, 2],
                   'y': [1, 1, 5, 5, 2],
                   'labels': ['A', 'A', 'B', 'B', 'A']})

colors = {'A' : 'red',
          'B' : 'blue'}

df.plot.scatter(x='x,',
                y='y',
                c=df['labels'].apply(lambda x:colors[x]))
plt.show()

# Visualize using seaborn
import seaborn as sns

sns.scatterplot(x='x',
                y='y',
                hue='labels',
                data=df)
plt.show()

# C: How many clusters?
# Introduction to dendrograms
# Dendrograms help in showing progressions as clusters are merged
# A branching diagram that demonstrates how each cluster is composed by branching out into its child nodes

# Create a dendrogram in SciPy
from scipy.cluster.hierarchy import dendrogram
Z = linkage(df[['x_whiten', 'y_whiten']],
            method='ward',
            metric='euclidean')
dn = dendrogram(Z)
plt.show()

# Distance between vertical lines - indicates inter-cluster distance
# Intersections - number of clusters

# D: Limitations of hierarchical clustering
# Measuring speed in hierarchical clustering
# Linkage - most time consuming
# Can use timeit module to time it

# Use of timeit module
from scipy.cluster.hierarchy import linkage
import pandas as pd
import random, timeit

points = 100

df = pd.DataFrame({'x':random.sample(range(0, points), points),
                   'y':random.sample(range(0, points), points)})

%timeit linkage(df[['x','y']], method='ward', metric="euclidean")

# Can perform iterations for an increasing number of points
# Increasing runtimewith datapoints
# Quadratic increase of runtime
# Not feasible for large datasets

""" Chapter 3 - Basics of K-Means clustering"""
# K means runs significantly faster on large datasets

# Step 1 - Generate cluster centres
# kmeans(obs, k_or_guess=, iter=, thresh=, check_finite=)
# obs: standardized observations
# k_or_guess: number of clusters
# iter: number of iterations (default:20)
# thres: threshold (default: 1e-05) - alg terminated if change in distortion between iters is <= threshold
# check_finite: whether to check if observations contain only finite numbers (default: True)
#  - True = Data with NaN or infinite values are not considered for classification
#  - Ensures results are accurate and unbiased

# Returns two objects - cluster centres and distortion
# Considerably fewer operations than hierarchical clustering

# Distortion - sum of sq of distances between datapoints and cluster centres

# Step 2 - Assign cluster labels
# vq(obs, code_book=, check_finite=True)
# obs: standardized observations
# code_book: cluster centres - first output of kmeans method
# check_finite: as above, default=True

# Returns two objects - a list of cluster labels, a list of distortions

# A note on distortions
# kmeans returns a single value of distortions - based on overall data
# vq returns a list of distortions - one for each datapoint
#  - Mean of list should approx equal kmeans value for same obs

# Running k-means
from scipy.cluster.vq import kmeans, vq

# Generate cluster centers and labels
cluster_centres, _ = kmeans(df[['scaled_x', 'scaled_y']], 3)
df['cluster_labels'], _ = vq(df[['scaled_x', 'scaled_y']], cluster_centres)

# Plot clusters
sns.scatterplot(x='scaled_x',
                y='scaled_y',
                hue='cluster_labels',
                data=df)
plt.show()

# B: How many clusters?
# No absolute right method to find the right number of clusters in kmeans
# Elbow method

# Distortion - decreases with an increasing number of clusters
# Becomes zero when the number of clusters = number of obs
# Elbow plot - line plot between cluster centres and corres distortions
# Ideal point - beyond which point distortion increases relatively less

# Declaring variables for use
distortions = []
num_clusters = range(2,7)

# Populating distortions for various clusters
for i in num_clusters:
    centroids, distortion = kmeans(df[['scaled_x', 'scaled_y']], i)
    distortions.append(distortion)

# Plotting elbow plot data
elbow_plot_data = pd.DataFrame({'num_clusters' : num_clusters,
                                'distortions' : distortions})
sns.lineplot(x='num_clusters', y='distortions', data=elbow_plot_data)
plt.show()

# Elbow method only gives an indication of optimal _k_
# Does not always pinpoint optimal k_
# Fails when distortion is evenly distributed
# Other methods: average silhouette and gap statistic

# Limitations of kmeans clustering
# Overcomes runtime challenges
# How to find _k_? Elbow method is one way, but may not always work
# Impact of seeds on clustering
# Biased towards equal sized clusters

# Impact of seeds
# Process of defining initial cluster centres is random, can affect final clusters
# To get consistent results - initialise a seed
from numpy import random
random.seed(12)

# Impact of seeds only to be seen when data is clustered in a fairly uniform way
# If distinct clusters exist - effect of seeds will not result in any changes

# Uniform clusters in kmeans
# kMeans - minimises distortions - results in clusters with similar areas
# Hierarchical - using complete method, clusters formed are intuitive and consistent

# Consider datasize and pattern before deciding on algorithm


""" Chapter 4 - Clustering in the real world"""

# Dominant colors in images





