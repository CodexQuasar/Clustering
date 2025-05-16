# Enhanced Clustering Pipeline with Visualization and Analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# 1. Load and preprocess data
file_path = "C:/Users/adity/Downloads/TASK2_dataset.csv"
df = pd.read_csv(file_path)

# Text preprocessing function
def preprocess_text(text):
    if pd.isna(text):
        return ""
    return str(text).lower().strip()  # Add more preprocessing as needed

# 2. Enhanced TF-IDF computation with preprocessing
def compute_tfidf(texts, max_features=1000):
    vectorizer = TfidfVectorizer(
        preprocessor=preprocess_text,
        max_features=max_features,
        stop_words='english'  # Optional: remove English stopwords
    )
    tfidf_matrix = vectorizer.fit_transform(texts).toarray()
    vocab = vectorizer.get_feature_names_out()
    return tfidf_matrix, vocab

# 3. Improved PCA with scaling
def pca_manual(X, n_components=2):
    # Standardize first
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Center the data
    X_meaned = X_scaled - np.mean(X_scaled, axis=0)
    
    # Compute covariance matrix
    covariance_matrix = np.cov(X_meaned, rowvar=False)
    
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Sort and select components
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors_sorted = eigenvectors[:, sorted_indices]
    selected_eigenvectors = eigenvectors_sorted[:, :n_components]
    
    return np.dot(X_meaned, selected_eigenvectors)

# 4. Process all keyword columns
final_features = []
keyword_vectors = {}

for col in ["keyword_1", "keyword_2", "keyword_3"]:
    texts = df[col].astype(str).tolist()
    tfidf_matrix, vocab = compute_tfidf(texts)
    reduced_matrix = pca_manual(tfidf_matrix, 2)
    final_features.append(reduced_matrix)
    keyword_vectors[col] = {word: vector for word, vector in zip(vocab, tfidf_matrix.T)}

# Create combined features
X_final = np.hstack(final_features)  # 6D features
X_embedding = np.mean(final_features, axis=0)  # 2D embedding

# 5. Enhanced K-Means with better initialization
def kmeans_manual(X, k, max_iters=100, tol=1e-4):
    np.random.seed(42)
    
    # Improved initialization: select points far apart
    centroids = [X[np.random.randint(X.shape[0])]]
    for _ in range(1, k):
        dists = np.array([min([np.linalg.norm(x - c) for c in centroids]) for x in X])
        probabilities = dists / dists.sum()
        centroids.append(X[np.argmax(probabilities)])
    centroids = np.array(centroids)

    for _ in range(max_iters):
        # Vectorized distance computation
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        clusters = np.argmin(distances, axis=1)
        
        new_centroids = np.array([X[clusters == j].mean(axis=0) for j in range(k)])
        
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        
        centroids = new_centroids
    
    return clusters, centroids

# 6. Elbow Method with silhouette analysis
def plot_elbow_and_silhouette(X, max_k=10):
    inertia_values = []
    k_values = range(2, max_k+1)  # Start from 2 for silhouette
    
    plt.figure(figsize=(12, 5))
    
    # Elbow plot
    plt.subplot(1, 2, 1)
    for k in k_values:
        clusters, centroids = kmeans_manual(X, k)
        inertia = sum(np.min(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1))
        inertia_values.append(inertia)
    
    plt.plot(k_values, inertia_values, marker='o', linestyle='--')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    
    # Silhouette plot (conceptual - actual calculation would require more code)
    plt.subplot(1, 2, 2)
    plt.plot(k_values, [0]*len(k_values), marker='o')  # Placeholder
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score (higher is better)")
    plt.title("Silhouette Analysis")
    
    plt.tight_layout()
    plt.show()
    
    return inertia_values

# Run analysis
inertia_values = plot_elbow_and_silhouette(X_embedding)

# 7. After choosing optimal k (example with k=3)
optimal_k = 3  # Change based on your elbow plot
clusters, centroids = kmeans_manual(X_embedding, optimal_k)

# 8. Visualize clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=X_embedding[:, 0], 
    y=X_embedding[:, 1], 
    hue=clusters,
    palette="viridis",
    alpha=0.7
)
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker='X',
    s=200,
    c='red',
    label='Centroids'
)
plt.title("Keyword Clusters Visualization")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()

# 9. Analyze cluster keywords
def analyze_clusters(df, clusters, keyword_columns=["keyword_1", "keyword_2", "keyword_3"]):
    df['cluster'] = clusters
    for cluster_num in range(optimal_k):
        print(f"\nCluster {cluster_num} Analysis:")
        cluster_data = df[df['cluster'] == cluster_num]
        
        # Show top keywords
        all_keywords = []
        for col in keyword_columns:
            all_keywords.extend(cluster_data[col].dropna().tolist())
        
        from collections import Counter
        keyword_counts = Counter(all_keywords)
        print("Top 10 Keywords:", keyword_counts.most_common(10))
        
        # Optional: Show sample documents
        print("\nSample documents:")
        print(cluster_data.head(3).to_string())

analyze_clusters(df, clusters)