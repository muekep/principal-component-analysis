import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# Load the Breast Cancer Dataset from the sklearn.datasets
cancer = load_breast_cancer()
#Separate the features and the target variable
X = cancer.data
y = cancer.target
#Create a Pandas DataFrame to make the data more readable and include the feature names.
feature_names = cancer.feature_names

# Create a Pandas DataFrame for better visualization and understanding
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print("Original Dataset Shape:", df.shape)
print(df.head())

# Standardize the Data
#Use StandardScaler to standardize the features by removing the mean and scaling to unit variance. 
#This ensures that all features contribute equally to the PCA process.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert the scaled data back to a DataFrame for easier handling.
df_scaled = pd.DataFrame(X_scaled, columns=feature_names)
print(df_scaled.head())

# Apply Principal Component Analysis
# Initialize PCA with the number of components you want to retain.
# Start by keeping all components to analyze explained variance.
pca = PCA()
pca.fit(X_scaled)

# Explained Variance Ratio
# This tells us the proportion of the dataset's variance explained by each principal component.
explained_variance_ratio = pca.explained_variance_ratio_
print(explained_variance_ratio)

# Cumulative Explained Variance
# This shows the cumulative proportion of variance explained by the top N components.
cumulative_explained_variance = np.cumsum(explained_variance_ratio)
print("\nCumulative Explained Variance:")
print(cumulative_explained_variance)

# Visualize Explained Variance (Scree Plot)
# A scree plot helps in deciding the optimal number of principal components to retain.
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='-')
plt.title('Scree Plot')
plt.xlabel('Principal Component Number')
plt.ylabel('Explained Variance Ratio')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='-')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.grid(True)
plt.show()

# Determine the Number of Essential Components
# Based on the scree plot and cumulative explained variance, we can decide how many
# components capture a significant portion of the variance.

# Let's find the number of components that explain at least 90% of the variance.
n_components_90 = np.argmax(cumulative_explained_variance >= 0.90) + 1
print(f"\nNumber of principal components needed to explain 90% of variance: {n_components_90}")
