import streamlit as st
import pickle
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt

# Load the preprocessed and scaled data (assumed to be pre-saved in a file)
with open('preprocessed_scaled_data.pkl', 'rb') as f:
    X_scaled = pickle.load(f)

# Streamlit Sidebar for UMAP Parameters
st.sidebar.header("UMAP Parameters")
n_neighbors = st.sidebar.slider('n_neighbors', min_value=5, max_value=50, value=15, step=1)
min_dist = st.sidebar.slider('min_dist', min_value=0.0, max_value=1.0, value=0.1, step=0.01)
n_components = st.sidebar.slider('n_components', min_value=2, max_value=10, value=2, step=1)

# Streamlit Sidebar for Clustering Parameters
st.sidebar.header("Clustering Parameters")
n_clusters = st.sidebar.slider('n_clusters', min_value=2, max_value=10, value=3, step=1)
linkage = st.sidebar.selectbox('linkage', options=['single', 'complete', 'average', 'ward'], index=1)

# Step 1: Apply UMAP
st.write("### UMAP Dimensionality Reduction")
umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=42)
umap_transformed_data = umap_model.fit_transform(X_scaled)

st.write("UMAP completed with the following parameters:")
st.write(f"n_neighbors: {n_neighbors}, min_dist: {min_dist}, n_components: {n_components}")

# Step 2: Apply Agglomerative Clustering
st.write("### Agglomerative Clustering")
clustering_model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
cluster_labels = clustering_model.fit_predict(umap_transformed_data)

# Step 3: Calculate Metrics
silhouette_avg = silhouette_score(umap_transformed_data, cluster_labels)
calinski_harabasz_avg = calinski_harabasz_score(umap_transformed_data, cluster_labels)
davies_bouldin_avg = davies_bouldin_score(umap_transformed_data, cluster_labels)

# Step 4: Display Metrics
st.write(f"**Silhouette Score**: {silhouette_avg:.2f}")
st.write(f"**Calinski-Harabasz Index**: {calinski_harabasz_avg:.2f}")
st.write(f"**Davies-Bouldin Index**: {davies_bouldin_avg:.2f}")

# Step 5: Plot the Clustering Results
st.write("### Clustering Results Plot")
fig, ax = plt.subplots()
scatter = ax.scatter(umap_transformed_data[:, 0], umap_transformed_data[:, 1], c=cluster_labels, cmap='viridis', s=50)
plt.title(f"Agglomerative Clustering with {n_clusters} Clusters\n"
          f"Silhouette Score = {silhouette_avg:.2f}")
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.grid(True)

st.pyplot(fig)
