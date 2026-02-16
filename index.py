
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ==========================================
# STEP 1: LOAD DATA
# ==========================================

# Load customer dataset (replace with actual dataset path)
df = pd.read_csv("customers.csv")

# Display first few rows
print(df.head())

# ==========================================
# STEP 2: DATA CLEANING
# ==========================================

# Remove duplicate records
df.drop_duplicates(inplace=True)

# Fill missing numerical values with median
df.fillna(df.median(numeric_only=True), inplace=True)

# Fill missing categorical values
df.fillna("Unknown", inplace=True)

# ==========================================
# STEP 3: FEATURE ENGINEERING
# ==========================================

# Create Annual Spending feature
df['Annual_Spending'] = df['Avg_Order_Value'] * df['Purchase_Frequency']

# Create Engagement Score using weighted formula
df['Engagement_Score'] = (
    0.4 * df['Website_Visits'] +
    0.3 * df['App_Usage_Score'] +
    0.3 * df['Email_Response_Rate']
)

# ==========================================
# STEP 4: SELECT FEATURES FOR CLUSTERING
# ==========================================

features = [
    'Age',
    'Income',
    'Annual_Spending',
    'Purchase_Frequency',
    'Recency',
    'Engagement_Score',
    'Discount_Usage',
    'Return_Rate'
]

X = df[features]

# ==========================================
# STEP 5: FEATURE SCALING
# ==========================================

# Scaling ensures all features contribute equally
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# STEP 6: FIND OPTIMAL NUMBER OF CLUSTERS
# ==========================================

wcss = []  # Within Cluster Sum of Squares

for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot elbow curve
plt.plot(range(1, 10), wcss)
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method for Optimal Clusters")
plt.show()

# ==========================================
# STEP 7: APPLY K-MEANS CLUSTERING
# ==========================================

# Assume optimal clusters = 4
kmeans = KMeans(n_clusters=4, random_state=42)

# Assign cluster labels to customers
df['Customer_Segment'] = kmeans.fit_predict(X_scaled)

# ==========================================
# STEP 8: SEGMENT PROFILING
# ==========================================

# Calculate average values per segment
segment_profile = df.groupby('Customer_Segment')[features].mean()

print("\nSegment Profile:\n")
print(segment_profile)

# ==========================================
# STEP 9: VISUALIZATION
# ==========================================

# Visualize segments
sns.scatterplot(
    x=df['Annual_Spending'],
    y=df['Purchase_Frequency'],
    hue=df['Customer_Segment'],
    palette='viridis'
)

plt.title("Customer Segments")
plt.xlabel("Annual Spending")
plt.ylabel("Purchase Frequency")
plt.show()

# ==========================================
# STEP 10: SAVE OUTPUT
# ==========================================

# Save segmented dataset
df.to_csv("customer_segments_output.csv", index=False)

print("\nCustomer segmentation completed successfully.")
