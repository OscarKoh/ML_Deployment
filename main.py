import streamlit as st
import pickle
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt

# Title and Description
st.title("Agglomerative Clustering with Pre-Processed Data")
st.write("This app allows you to apply UMAP for dimensionality reduction and Agglomerative Clustering on pre-processed data.")

# Step 1: Load Preprocessed Data
st.write("Loading Preprocessed and Scaled Data...")
try:
    with open('preprocessed_scaled_data.pkl', 'rb') as f:
        X_scaled, numeric_cols = pickle.load(f)
    st.write("Data successfully loaded!")
except FileNotFoundError:
    st.error("Preprocessed data file not found. Please ensure the preprocessed_scaled_data.pkl file is available.")
    st.stop()

# Step 2: UMAP Parameters Input
st.sidebar.header("UMAP Parameters")
best_n_neighbors = st.sidebar.slider('n_neighbors', min_value=5, max_value=50, value=10, step=1)
best_min_dist = st.sidebar.slider('min_dist', min_value=0.0, max_value=1.0, value=0.1, step=0.01)
n_components = st.sidebar.slider('n_components', min_value=2, max_value=10, value=2, step=1)

# Step 3: UMAP Dimensionality Reduction
st.write("Applying UMAP for Dimensionality Reduction...")
umap_model = umap.UMAP(n_neighbors=best_n_neighbors, min_dist=best_min_dist, n_components=n_components, random_state=42)
umap_transformed_data = umap_model.fit_transform(X_scaled)
st.write("UMAP Dimensionality Reduction Done.")

# Step 4: Agglomerative Clustering Parameter Grid and Fine-Tuning
param_grid = {
    'n_clusters': range(2, 11),  # Test a range of cluster numbers
    'linkage': ['single', 'complete', 'average', 'ward']  # Test different linkage methods
}

results = []

# Step 5: Apply Agglomerative Clustering and Evaluate Metrics
for params in ParameterGrid(param_grid):
    hierarchical = AgglomerativeClustering(n_clusters=params['n_clusters'], linkage=params['linkage'])
    hierarchical_labels = hierarchical.fit_predict(umap_transformed_data)
    
    silhouette_avg = silhouette_score(umap_transformed_data, hierarchical_labels)
    calinski_harabasz_avg = calinski_harabasz_score(umap_transformed_data, hierarchical_labels)
    davies_bouldin_avg = davies_bouldin_score(umap_transformed_data, hierarchical_labels)
    
    results.append({
        'params': params,
        'silhouette_score': silhouette_avg,
        'calinski_harabasz_score': calinski_harabasz_avg,
        'davies_bouldin_score': davies_bouldin_avg
    })

# Step 6: Find Best Clustering Parameters
results_df = pd.DataFrame(results)
best_result = results_df.loc[results_df['silhouette_score'].idxmax()]
st.write("Best Parameters Based on Silhouette Score:", best_result)

# Step 7: Apply Best Agglomerative Clustering Model
best_hierarchical = AgglomerativeClustering(n_clusters=best_result['params']['n_clusters'], linkage=best_result['params']['linkage'])
best_hierarchical_labels = best_hierarchical.fit_predict(umap_transformed_data)

# Step 8: Final Evaluation Metrics
st.write(f"Final Silhouette Score: {silhouette_score(umap_transformed_data, best_hierarchical_labels):.2f}")
st.write(f"Final Calinski-Harabasz Index: {calinski_harabasz_score(umap_transformed_data, best_hierarchical_labels):.2f}")
st.write(f"Final Davies-Bouldin Index: {davies_bouldin_score(umap_transformed_data, best_hierarchical_labels):.2f}")

# Step 9: Plot Clustering Results
fig, ax = plt.subplots()
scatter = ax.scatter(umap_transformed_data[:, 0], umap_transformed_data[:, 1], c=best_hierarchical_labels, cmap='viridis', s=50)
plt.title(f"Hierarchical Clustering with {best_result['params']['n_clusters']} Clusters")
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.grid(True)

st.pyplot(fig)
