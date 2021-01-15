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
# Analysing images to find dominant colors
# All images consist of pixels
# Each pixel has three values - Red, Green and Blue
# Pixel color - combination of RBG values
# Perform k-means on standardized RGB values to find cluster centers
# Uses - identifying features in satellite images

# Feature identification in satellite images
# Kmeans can be used to identify surface features

# Tools to find dominant colors
# Convert image to pixels - matplotlib.image.imread - converts jpg into matrix with RGB values
# Display colors of cluster centres - matplotlib.pyplot.imshow

# Convert image to RGB image
import os
#print(os.getcwd())
os.chdir('C:\\Users\\susie\\PycharmProjects\\datacamp')

import matplotlib.image as img
import pandas as pd
image = img.imread('sea.jpg')
image.shape  # Three dimensions

r = []
g = []
b = []

for row in image:
    for pixel in row:
        # A pixel contains RGB values
        temp_r, temp_g, temp_b = pixel
        r.append(temp_r)
        g.append(temp_g)
        b.append(temp_b)

pixels = pd.DataFrame({'red' : r,
                       'blue': b,
                       'green' : g})

print(pixels.head())

# Scale data
from scipy.cluster.vq import whiten

scaled_red = whiten(pixels['red'])
scaled_green = whiten(pixels['green'])
scaled_blue = whiten(pixels['blue'])

pixels_scaled = pd.DataFrame({'scaled_red' : scaled_red,
                              'scaled_green' : scaled_green,
                              'scaled_blue' : scaled_blue})

# Create an elbow plot
distortions = []
num_clusters = range(1,11)

from scipy.cluster.vq import kmeans, vq

# Create a list of distortions from the kmeans method
for i in num_clusters:
    cluster_centres, distortion = kmeans(pixels_scaled[['scaled_red', 'scaled_blue', 'scaled_green']], i)
    distortions.append(distortion)

# Create a data frame with two lists - number of clusters and distortions
elbow_plot = pd.DataFrame({'num_clusters' : num_clusters,
                           'distortions' : distortions})

# Create a line plot of num_clusters and distortions
sns.lineplot(x='num_clusters', y='distortions', data=elbow_plot)
plt.xticks(num_clusters)
plt.show()

# Elbow plot indicates 2 clusters - support initial observation of two colors in the image

# Standardising colors

colors = []

# Find standard deviations
r_std, g_std, b_std = pixels[['red', 'blue', 'green']].std()

# Scale actual RGB values in range of 0-1
for cluster_centre in cluster_centres:
    scaled_r, scaled_g, scaled_b = cluster_centre
    colors.append((
        scaled_r * r_std/255,
        scaled_g * g_std/255,
        scaled_b * b_std/255
    ))

# Display dominant colors

# Dimensions: 2 x 3 (N x 3 matrix)
print(colors)

# Dimensions: 1 x 2 x 3 (1 x N x 3 matrix)
plt.imshow([colors])
plt.show()

# B: Document clustering
# 1. Clean data before processing - remove items including punctuation, emoticons and other words such as 'the'
# 2. Determine importance of terms in doc - TF-IDF matrix
# 3. Display top terms in each cluster

# Clean and tokenize data
# Convert text into smaller parts called tokens, clean data for processing

from nltk.tokenize import word_tokenize
import re

def remove_noise(text, stop_words = []):
    tokens = word_tokenize(text)
    cleaned_tokens = []
    for token in tokens:
        token = re.sub('[^A-Za-z0-9]+','',token)
        if len(token) > 1 and token.lower() not in stop_words:
            # Get lowercase
            cleaned_tokens.append(token.lower())
    return  cleaned_tokens

# Document term matrix and sparse matrices
# Document term matrix formed
# Most elements in matrix are zeros
# Sparse matrix is created - more efficient storage

# TF-IDF (Term Frequency - Inverse Document Frequency)
# A weighted measure - evaluate how important a word is to a document in a collection
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=50,  # Max and min fraction of documents (20-80% docs)
                                   # Keep top 50 terms
                                   min_df=0.2, tokenizer=remove_noise)  # Use custom function as tokenizer

tfidf_matrix = tfidf_vectorizer.fit_transform(data)

# Clustering with sparse matrix
# kmeans() in SciPy does not support sparse matrices
# Use .todense() to convert to a matrix

cluster_centres, distortion = kmeans(tfidf_matrix.todense(), num_clusters)
# Don't use elbow plot - takes an erratic form due to the high number of variables

# Top terms per cluster
# Cluster centres - lists with a size equal to the number of terms
# Each value in the cluster center is its importance
# Create a dictionary and print top terms

terms = tfidf_vectorizer.get_feature_names()  # Create a list of all terms

for i in range(num_clusters):
    center_terms = dict(zip(terms, list(cluster_centres[i])))  # Create a dict with cluster centres and terms
    sorted_terms = sorted(center_terms, key=center_terms.get, reverse=True)
    print(sorted_terms[:3])

# More considerations
# Can modify remove_noise function to work with hyperlinks, emoticons etc.
# Normalize words to base form (run, ran, running -> run)
# .todense() may not work with large datasets

# C: Clustering with multiple features
# Basic checks

# Prints average category values by cluster
print(fifa.groupby('cluster_labels')[['scaled_heading_accuracy', 'scaled_volleys', 'scaled_finishing']].mean())

# Prints count of cluster labels
print(fifa.groupby('cluster_labels')['ID'].count())

# If one cluster smaller than others - may want to check if cluster centre similar to other clusters
# If yes, may want to reduce number of clusters

# Visualizations
# Visualise cluster centres
# Visualise other variables for each cluster

# Plot cluster centers
fifa.groupby('cluster_labels')[scaled_features].mean().plot(kind='bar')
plt.show()

# Top items in clusters
for cluster in fifa['cluster_labels'].unique()
    print(cluster, fifa[fifa['cluster_labels'] == cluster]['name'].values[:5])

# Feature reduction
# Useful when dealing with large number of features
# Factor analysis, multi-dimensional scaling - pre-cursor to clustering

