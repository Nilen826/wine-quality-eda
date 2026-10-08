import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("winequalityN.csv.csv")
print(df.head())
# Shape
print("Shape:", df.shape)

# Data types
print("\nData Types:")
print(df.dtypes)

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicates
print("\nDuplicate Rows:", df.duplicated().sum())
# --- PLOT 1: Distribution of all numeric columns ---
df.hist(figsize=(14, 10), bins=30, color='steelblue', edgecolor='white')
plt.suptitle('Feature Distributions')
plt.tight_layout()
plt.show()

# --- PLOT 2: Wine type count ---
df['type'].value_counts().plot(kind='bar', color=['steelblue', 'crimson'])
plt.title('Wine Type Count')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# --- PLOT 3: Quality score distribution ---
df['quality'].value_counts().sort_index().plot(kind='bar', color='steelblue')
plt.title('Quality Score Distribution')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# --- PLOT 4: Correlation heatmap ---
plt.figure(figsize=(12, 8))
sns.heatmap(df.select_dtypes(include='number').corr(),
            annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# --- PLOT 5: Boxplots to detect outliers ---
numeric_cols = df.select_dtypes(include='number').columns
df[numeric_cols].plot(kind='box', subplots=True,
                      layout=(3, 4), figsize=(16, 10))
plt.suptitle('Boxplots - Outlier Detection')
plt.tight_layout()
plt.show()
