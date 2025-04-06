import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the Breast Cancer Dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
feature_names = cancer.feature_names

# Create a Pandas DataFrame for better visualization and understanding
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print("Original Dataset Shape:", df.shape)
print("\nFirst 5 rows of the original data:")
print(df.head())

# Standardize the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA with 2 Components
n_components = 2
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# Create a new DataFrame with the principal components
principal_df = pd.DataFrame(data=X_pca, columns=['principal component 1', 'principal component 2'])

# Concatenate with the target variable
final_df = pd.concat([principal_df, df[['target']]], axis=1)

print("\nShape of the data after PCA with 2 components:", final_df.shape)
print("\nFirst 5 rows of the data with 2 principal components:")
print(final_df.head())

# Analyze Explained Variance Ratio for 2 Components
explained_variance_ratio = pca.explained_variance_ratio_
print("\nExplained Variance Ratio for 2 Principal Components:", explained_variance_ratio)
print("Total Explained Variance (for 2 components):", sum(explained_variance_ratio))

# Visualize the Reduced Data (2D Scatter Plot)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='principal component 1', y='principal component 2', hue='target', data=final_df, palette='viridis')
plt.title('2-Component PCA of Breast Cancer Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

# Analyze the Contribution of Original Features to the 2 Principal Components
components_df = pd.DataFrame(pca.components_, columns=feature_names, index=['principal component 1', 'principal component 2'])
print("\nContribution of Original Features to the 2 Principal Components:")
print(components_df)

# Interpretation of the Components:
# - Examine the signs and magnitudes of the values in each row.
# - For 'principal component 1', features with large positive or negative values have a strong influence on this component. 
# - A positive value means the feature increases along with the component, while a negative value means it decreases.
# - Similarly, analyze 'principal component 2'.
# - This helps understand which original features are most important in distinguishing the data along these two principal component axes.
