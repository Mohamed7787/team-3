# Part E - Unsupervised Customer Segmentation (K-Means)
from commen import *
from preproc import X_train_c, num_cols, cat_cols, num_pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Preprocess raw numerical features (Impute NaNs + Scale)
X_cluster_scaled = num_pipeline.fit_transform(X_train_c[num_cols])

# 2. Find optimal K using Elbow Method & Silhouette Scores
inertia = []
silhouette_scores = []
K_range = range(2, 9)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_cluster_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_cluster_scaled, labels))

# Plot Elbow and Silhouette curves
fig, ax1 = plt.subplots(figsize=(8, 4))

ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia (Elbow)', color='tab:blue')
ax1.plot(K_range, inertia, 'bo-', label='Inertia')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Silhouette Score', color='tab:orange')
ax2.plot(K_range, silhouette_scores, 'or--', label='Silhouette Score')
ax2.tick_params(axis='y', labelcolor='tab:orange')

plt.title('Elbow Method & Silhouette Analysis for Optimal K')
fig.tight_layout()
plt.show()

# 3. Train final K-Means with optimal K (e.g., K=3)
optimal_k = 3
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)

# Create a copy to attach cluster labels for profiling
X_train_profile = X_train_c.copy()
X_train_profile['Cluster'] = final_kmeans.fit_predict(X_cluster_scaled)

# 4. Profile Clusters
print(f"=== Customer Profile Summary (K = {optimal_k}) ===")
cluster_profile = X_train_profile.groupby('Cluster')[num_cols].mean()
print(cluster_profile)