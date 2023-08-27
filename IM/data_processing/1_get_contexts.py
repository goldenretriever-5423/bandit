import numpy as np
from sklearn import cluster
from gensim import downloader

context_vectors_file = "1_10context_vectors.out"
num_clusters = 10

# use glove dataset to build semantic context
model = downloader.load("glove-twitter-200")
kmeans_training_data = model[model.wv.vocab]
kmeans = cluster.KMeans(n_clusters=num_clusters)
kmeans.fit(kmeans_training_data)

# build the kernels
#labels = kmeans.labels_
centroids = kmeans.cluster_centers_
np.savetxt(context_vectors_file, centroids) 

