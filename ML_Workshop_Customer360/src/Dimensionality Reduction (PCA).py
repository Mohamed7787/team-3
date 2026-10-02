# Part F - Dimensionality Reduction (PCA) & Anomaly Detection
from commen import *
from preproc import x_train_pred_c, y_train_c
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

# 1. Apply PCA on fully preprocessed feature set (X_train_prep has 0 NaNs)
pca = PCA(n_components=2, random_state=42)
X_train_pca = pca.fit_transform(x_train_pred_c)

explained_variance = pca.explained_variance_ratio_
print(f"PCA Variance Explained: Comp 1 = {explained_variance[0]:.2%}, Comp 2 = {explained_variance[1]:.2%}")
print(f"Total Explained Variance: {sum(explained_variance):.2%}")

# Plot 2D PCA projection colored by Churn
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train_c, cmap='coolwarm', alpha=0.6)
plt.colorbar(scatter, label='Churn (0 = No, 1 = Yes)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Customer Data 2D Visualization (PCA Projection)')
plt.show()

# 2. Anomaly Detection using Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
anomalies = iso_forest.fit_predict(x_train_pred_c)

# Convert preprocessed array to a DataFrame or attach to original X_train copy
X_train_anomalies = pd.DataFrame(x_train_pred_c)
X_train_anomalies['Is_Anomaly'] = np.where(anomalies == -1, True, False)

num_anomalies = X_train_anomalies['Is_Anomaly'].sum()
print(f"\n=== Anomaly Detection Results ===")
print(f"Detected Anomalous Customers: {num_anomalies} out of {len(x_train_pred_c)} ({num_anomalies / len(x_train_pred_c):.1%})")

# Visualizing Anomalies on PCA Space
plt.figure(figsize=(8, 6))
plt.scatter(X_train_pca[anomalies == 1, 0], X_train_pca[anomalies == 1, 1], c='blue', alpha=0.3, label='Normal Customer')
plt.scatter(X_train_pca[anomalies == -1, 0], X_train_pca[anomalies == -1, 1], c='red', alpha=0.8, label='Anomalous Customer', marker='x')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Anomaly Detection via Isolation Forest (PCA Reduced)')
plt.legend()
plt.show()