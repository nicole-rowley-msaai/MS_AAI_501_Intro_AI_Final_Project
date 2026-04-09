import os
import kagglehub
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, precision_recall_curve, average_precision_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
"""
Dependencies: pip3 install:
    kaggle, kagglehub, pandas, numpy, matplotlib, seaborn, scikit-learn
    *** or install from requirements.txt ***

Generate token key at https://www.kaggle.com/settings under user profile.
Create token file ~/.kaggle/kaggle.json with username and key format: 
{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}

"""
# ---------------Kaggle Authentication---------------

# Kaggle API Authentication
dataset_reference = 'nelgiriyewithana/credit-card-fraud-detection-dataset-2023' 
download_path = './data' # Local directory to save the data

# Create the directory if it doesn't exist
if not os.path.exists(download_path):
    os.makedirs(download_path)

# Use the kaggle API command to download the dataset
# The command is run using '!' or '%' in notebooks, or subprocess in scripts
print(f"\nDownloading dataset {dataset_reference} to {download_path}...")
os.system(f'kaggle datasets download -d {dataset_reference} -p {download_path}')

# Unzip the downloaded file(s)
# The download will be a zip file with the dataset name
zip_files = [f for f in os.listdir(download_path) if f.endswith('.zip')]
for zip_file in zip_files:
    zip_path = os.path.join(download_path, zip_file)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(download_path)
    os.remove(zip_path) # remove the zip file after extraction
    print(f"\nExtracted {zip_file}")

print("\nDownload and extraction complete.")

csv_file_path = os.path.join(download_path, 'creditcard_2023.csv') 
df = pd.read_csv(csv_file_path, sep=None, engine='python') # Use sep=None to auto-detect the separator

# ---------------Data Acquisition & Inspection---------------

print(f"\n{'#'*30}\nStarting Data Acquisition & Inspection\n{'#'*30}")

df_cc_fraud = df.copy() # Create a copy of the original dataframe for processing

# Load into Pandas and Inspect
df_cc_fraud = pd.read_csv('creditcard_2023.csv')  # adjust filename if needed
print(f"\nOriginal Dataset Shape: {df_cc_fraud.shape}")
print(df_cc_fraud.head())

# ---------------Exploratory Data Analysis---------------

print(f"\n{'#'*30}\nStarting Exploratory Data Analysis\n{'#'*30}")

# Summary of df
df_cc_fraud.info()

# Identify missing values
missing_values = df_cc_fraud.isnull().sum().max()
print(f"\nMaximum missing values in any column: {missing_values}")

# Drop non-predective columns
if 'id' in df_cc_fraud.columns:
    df_cc_fraud = df_cc_fraud.drop('id', axis=1)

# Check Class Balance
print(df_cc_fraud['Class'].value_counts(normalize=True))

# Summary Stats for Amount
amount_summary = df_cc_fraud['Amount'].describe()
print("--- Transaction Amount Range Summary ---")
print(f"Minimum: ${amount_summary['min']:.2f}")
print(f"Maximum: ${amount_summary['max']:.2f}")
print(f"Median:  ${amount_summary['50%']:.2f}")
print(f"Mean:    ${amount_summary['mean']:.2f}")
print(f"Range:   ${amount_summary['max'] - amount_summary['min']:.2f}")

# Define custom color mapping
custom_colors = {0: "seagreen", 1: "salmon"}

# Create the side-by-side boxplots for 'Amount' and 'Class'
plt.figure(figsize=(8, 6))
sns.boxplot(
    data=df_cc_fraud,
    x='Class',
    y='Amount',
    hue='Class',      # This links the color to the Class
    palette=custom_colors,
    legend=False      # This prevents an unnecessary extra legend
)

# Adjust formatting for readability
plt.title('Transaction Amounts: Legit (0) vs. Fraud (1)')
plt.xlabel('Transaction Class')
plt.ylabel('Amount ($)')
plt.xticks([0, 1], ['Legit (0)', 'Fraud (1)'])

plt.show()

# Remove Target Variable and Get Features
X = df_cc_fraud.copy()
X.drop(columns=['Class'], inplace=True)
# Get all 29 continuous features
features = [col for col in X.columns]
X.head()

# Assign Target Variable
y = df_cc_fraud['Class']
y.head()

#Combine into one DataFrame for easier analysis
df = X.copy()
df['target'] = y

# Create a 6x5 grid of subplots for Histograms
fig, axes = plt.subplots(6, 5, figsize=(24, 28))
axes = axes.flatten()

# Plot features
for i, col in enumerate(features):
    sns.histplot(
        data=df,
        x=col,
        hue='target',
        kde=True,
        bins=30,
        palette='Set1',
        ax=axes[i],
        alpha=0.5,
    )

    axes[i].set_title(col, fontsize=14)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')


# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()

# Create Heatmap of Features
corr_matrix = df[features].corr()

# Create a mask to hide the upper triangle (prevents duplicate visual information)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Set up the matplotlib figure size to accommodate 30 variables
plt.figure(figsize=(22, 18))

# Draw the heatmap
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
            linewidths=0.5, cbar_kws={"shrink": .5})

plt.title('Correlation Matrix of All 30 Features', fontsize=20)
plt.xticks(rotation=90, fontsize=12)
plt.yticks(rotation=0, fontsize=12)

plt.tight_layout()
plt.show()

# ---------------Data Preprocessing---------------

print(f"\n{'#'*30}\nStarting Data Preprocessing\n{'#'*30}")

# Create unbalanced master dataset
legit_df = df_cc_fraud[df_cc_fraud['Class'] == 0]
fraud_df = df_cc_fraud[df_cc_fraud['Class'] == 1]

# Target 0.2% fraud
num_fraud_to_keep = int((0.002 * len(legit_df)) / 0.998)
fraud_downsampled = fraud_df.sample(n=num_fraud_to_keep, random_state=42)

# Safe-Split (Stratified Ratio Split: splits a single class dataframe into subsets based on ratios)
def stratified_ratio_split(df_class, ratios):
    ratios = np.array(ratios) / sum(ratios)  # Normalize
    df_class = df_class.sample(frac=1, random_state=42)  # Shuffle

    # Calculate the integer split points
    indices = (ratios.cumsum() * len(df_class)).astype(int)
    
    # Use standard list slicing to ensure they remain DataFrames
    parts = []
    start_idx = 0
    for end_idx in indices:
        parts.append(df_class.iloc[start_idx:end_idx])
        start_idx = end_idx
        
    return parts

# Split legit and fraud separately to guarantee proportions
legit_subs = stratified_ratio_split(legit_df, [1, 2, 3])
fraud_subs = stratified_ratio_split(fraud_downsampled, [1, 2, 3])

# Combine them back into 3 subsets
subset_1 = pd.concat([legit_subs[0], fraud_subs[0]]).sample(frac=1, random_state=42)
subset_2 = pd.concat([legit_subs[1], fraud_subs[1]]).sample(frac=1, random_state=42)
subset_3 = pd.concat([legit_subs[2], fraud_subs[2]]).sample(frac=1, random_state=42)

# Define the StratifiedKFold for later use in cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Verification
subsets = [subset_1, subset_2, subset_3]

print(f"\n---Subset Summaries---")
for i, sub in enumerate(subsets, 1):
    fraud_count = sub['Class'].sum()
    print(f"Subset {i} ({['1/6', '2/6', '3/6'][i-1]}): {len(sub)} rows | "
          f"Fraud Count: {fraud_count} | "
          f"Fraud %: {(fraud_count/len(sub))*100:.4f}%")

# ---------------Random Forest Model Training and Testing---------------

print(f"\n{'#'*30}\nStarting Random Forest Evaluation\n{'#'*30}")

# Train and test random forest on the three subsets using an 80-20 split
# Create list of subsets and names
subsets = [subset_1, subset_2, subset_3]
subset_names = ["Subset 1 (1/6 size)", "Subset 2 (2/6 size)", "Subset 3 (3/6 size)"]

for i, data in enumerate(subsets):
    print(f"\n{'='*20} Evaluating {subset_names[i]} {'='*20}")

    # Feature/Target Separation
    X = data.drop('Class', axis=1)
    y = data['Class']

    # 80-20 Stratified Split (Crucial for 0.2% imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Scaling (Preventing Data Leakage by fitting only on Train)
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled['Amount'] = scaler.fit_transform(X_train[['Amount']])
    X_test_scaled['Amount'] = scaler.transform(X_test[['Amount']])

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)

    # Get Binary Predictions and Probabilities
    y_pred_rf = rf.predict(X_test_scaled)
    y_probs_rf = rf.predict_proba(X_test_scaled)[:, 1] # Probabilities for the 'Fraud' class

    # Calculate Metrics
    auprc = average_precision_score(y_test, y_probs_rf)

    # Print RF Results
    print(f"Total Test Samples: {len(y_test)} | Actual Fraud in Test: {y_test.sum()}")
    print(f"AUPRC Score: {auprc:.4f}")

    # Stratified Cross-Validation on the Training Data
    # This checks if the model performance is stable across different folds of subset 3
    print(f"Running 5-Fold Stratified CV...")
    cv_auprc = cross_val_score(rf, X_train_scaled, y_train, cv=skf, scoring='average_precision')
    print(f"Mean CV AUPRC Stability: {cv_auprc.mean():.4f} (+/- {cv_auprc.std():.4f})")

    # Print Classification Report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred_rf, target_names=['Legitimate (0)', 'Fraud (1)']))

    # Confusion Matrix Visualization with Seaborn Heatmap
    plt.figure(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False)
    plt.title(f'Random Forest Confusion Matrix: {subset_names[i]}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

# ---------------Feature Importance from Random Forest Model---------------

# Extract and store RF importance
rf_importances = pd.Series(rf.feature_importances_, index=X_train.columns)
rf_importances = rf_importances.sort_values(ascending=False)

# Plot Random Forest Feature Importance
plt.figure(figsize=(10, 6))
rf_importances.head(10).plot(kind='bar', color='seagreen')
plt.title("Random Forest: Top 10 Predictive Features for Fraud")
plt.ylabel("Importance Score")
plt.xticks(rotation=45)
plt.show()

# ---------------Gradient Boosting Model Training and Testing---------------

print(f"\n{'#'*30}\nStarting Gradient Boosting Evaluation\n{'#'*30}")

# Initialize a list to store GB results
gb_auprc_results = []

for i, data in enumerate(subsets):
    print(f"\n{'='*20} GB: Evaluating {subset_names[i]} {'='*20}")

    # Feature/Target Separation
    X_gb = data.drop('Class', axis=1)
    y_gb = data['Class']

    # 80-20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_gb, y_gb, test_size=0.20, random_state=42, stratify=y_gb
    )

    # Scaling (Scaling 'Amount' based on training data)
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled['Amount'] = scaler.fit_transform(X_train[['Amount']])
    X_test_scaled['Amount'] = scaler.transform(X_test[['Amount']])

    # Initialize and Train Gradient Boosting Classifier
    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gbc.fit(X_train_scaled, y_train)

    # Stratified Cross-Validation on the Training Data
    # This checks if the model performance is stable across different folds of subset 3
    print(f"Running 5-Fold Stratified CV...")
    cv_auprc = cross_val_score(gbc, X_train_scaled, y_train, cv=skf, scoring='average_precision')
    print(f"Mean CV AUPRC: {cv_auprc.mean():.4f} (+/- {cv_auprc.std():.4f})")

    # Get Binary Predictions and Probabilities for Test Set
    y_pred_gb = gbc.predict(X_test_scaled)
    y_probs_gb = gbc.predict_proba(X_test_scaled)[:, 1] 

    # Calculate Test Metrics 
    auprc_gb = average_precision_score(y_test, y_probs_gb)
    gb_auprc_results.append(auprc_gb)

    # Print Results
    print(f"Total Test Samples: {len(y_test)} | Actual Fraud in Test: {y_test.sum()}")
    print(f"Test AUPRC Score: {auprc_gb:.4f}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred_gb, target_names=['Legitimate (0)', 'Fraud (1)']))

    # Confusion Matrix Visualization
    plt.figure(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred_gb)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False)
    plt.title(f'GB Confusion Matrix: {subset_names[i]}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

# ---------------Gradient Boosting with Hyperparameter Tuning---------------

print(f"\n{'#'*30}\nStarting Hyperparameter Tuning for Gradient Boosting\n{'#'*30}")

# We will use Subset 3 for tuning as it has the most data
X_tune = subset_3.drop('Class', axis=1)
y_tune = subset_3['Class']

# Prepare data for tuning
X_train_tune, _, y_train_tune, _ = train_test_split(X_tune, y_tune, test_size=0.20, random_state=42, stratify=y_tune)
scaler_tune = StandardScaler()
X_train_tune['Amount'] = scaler_tune.fit_transform(X_train_tune[['Amount']])

# Define Grid
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'max_depth': [2, 3, 4],
    'subsample': [0.7, 1.0]
}

# Run Grid Search using Stratified K-Fold (skf)
grid_search = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_grid=param_grid,
    scoring='average_precision', # Changed to average_precision to match AUPRC goals
    cv=skf,                      # Now using the 5-fold Stratified object
    n_jobs=-1,
    verbose=1
)

print("Tuning parameters on Subset 3...")
grid_search.fit(X_train_tune, y_train_tune)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV AUPRC score: {grid_search.best_score_:.4f}")

# ---------------Feature Importance from Tuned Gradient Boosting---------------

# Extract from the best estimator found by GridSearchCV
gb_importances = pd.Series(grid_search.best_estimator_.feature_importances_, index=X_train.columns)
gb_importances = gb_importances.sort_values(ascending=False)

# Plot Gradient Boosting Feature Importance
plt.figure(figsize=(10, 6))
gb_importances.head(10).plot(kind='bar', color='royalblue')
plt.title("Gradient Boosting: Top 10 Predictive Features for Fraud")
plt.ylabel("Importance Score")
plt.xticks(rotation=45)
plt.show()

# ---------------K-Means Clustering Analysis---------------

print(f"\n{'#'*30}\nStarting K-Means Clustering Analysis\n{'#'*30}")

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Use Subset 3 for consistency with large-scale analysis
X_km = subset_3.drop('Class', axis=1)
y_km = subset_3['Class']

# Feature Scaling (K-Means is distance-based, so ALL features need scaling)
scaler_km = StandardScaler()
X_scaled_km = scaler_km.fit_transform(X_km)

# Finding the Optimal K (Elbow Method)
print("Calculating Elbow Method to find optimal clusters...")
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled_km)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Inertia)')
plt.show()

# Fit K-Means using K=2 to see if it naturally separates Fraud vs Legit)
print("Fitting K-Means with K=2...")
kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled_km)

# Evaluate Cluster-to-Class Alignment - check if Cluster 0 or 1 contains the majority of fraud cases
km_results = pd.DataFrame({'Actual_Class': y_km, 'Cluster': clusters})
print("\nCluster vs. Actual Class Distribution:")
print(pd.crosstab(km_results['Actual_Class'], km_results['Cluster']))

# Visualization using PCA (Reducing 30 features to 2D for plotting)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled_km)

plt.figure(figsize=(12, 5))

# Plot 1: Actual Classes
plt.subplot(1, 2, 1)
plt.scatter(X_pca[y_km == 0, 0], X_pca[y_km == 0, 1], c='seagreen', label='Legit', alpha=0.5, s=10)
plt.scatter(X_pca[y_km == 1, 0], X_pca[y_km == 1, 1], c='salmon', label='Fraud', alpha=0.9, s=30)
plt.title('Actual Fraud Labels (PCA-Reduced)')
plt.legend()

# Plot 2: K-Means Clusters
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.5, s=10)
plt.title('K-Means Cluster Assignments (K=2)')

plt.tight_layout()
plt.show()

# ---------------GB with Cluster Distance Feature Engineering---------------

print(f"\n{'#'*30}\nStarting Enhanced GB with Cluster Distance Engineering\n{'#'*30}")

# Using Subset 3 for the final comparison
X_eng = subset_3.drop('Class', axis=1)
y_eng = subset_3['Class']

# 80-20 Stratified Split
X_train, X_test, y_train, y_test = train_test_split(
    X_eng, y_eng, test_size=0.20, random_state=42, stratify=y_eng
)

# Full Feature Scaling (necessary for distance calculations)
scaler_final = StandardScaler()
X_train_scaled = pd.DataFrame(scaler_final.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler_final.transform(X_test), columns=X_test.columns)

# Generate Cluster Distances (Unsupervised Learning))
# We use K=8 to represent different "modes" of normal spending behavior
kmeans_eng = KMeans(n_clusters=8, init='k-means++', random_state=42, n_init=10)
kmeans_eng.fit(X_train_scaled)
train_dist = kmeans_eng.transform(X_train_scaled)
test_dist = kmeans_eng.transform(X_test_scaled)

# Add distances as new features (Dist_0 to Dist_7)
dist_cols = [f'Dist_C{i}' for i in range(8)]
X_train_final = pd.concat([X_train_scaled, pd.DataFrame(train_dist, columns=dist_cols)], axis=1)
X_test_final = pd.concat([X_test_scaled, pd.DataFrame(test_dist, columns=dist_cols)], axis=1)

# Train Enhanced GB Model
# Using the best parameters discovered from your earlier Grid Search
gb_enhanced = GradientBoostingClassifier(
    n_estimators=300, 
    learning_rate=0.05, 
    max_depth=4, 
    subsample=0.7, 
    random_state=42
)
gb_enhanced.fit(X_train_final, y_train)

# Results
y_probs_enhanced = gb_enhanced.predict_proba(X_test_final)[:, 1]
auprc_enhanced = average_precision_score(y_test, y_probs_enhanced)
print(f"Enhanced AUPRC (with Cluster Distances): {auprc_enhanced:.4f}")

# ---------------Final Model Comparison---------------

print(f"\n{'#'*30}\nFinal Model Comparison\n{'#'*30}")

# Gather results (Assuming 'auprc' is from your RF and 'auprc_gb' is from standard GB)
comparison_data = {
    'Model': ['Random Forest', 'Standard GB', 'Enhanced GB (K-Means Dist)'],
    'AUPRC Score': [auprc, gb_auprc_results[-1], auprc_enhanced]
}

df_compare = pd.DataFrame(comparison_data)

# Visualization
plt.figure(figsize=(10, 6))
sns.barplot(data=df_compare, x='Model', y='AUPRC Score', palette='viridis')
plt.ylim(df_compare['AUPRC Score'].min() - 0.05, 1.0) # Zoom in to see differences
plt.title('Performance Comparison: Area Under Precision-Recall Curve')
plt.ylabel('AUPRC (Higher is Better)')

# Add value labels on top of bars
for i, val in enumerate(df_compare['AUPRC Score']):
    plt.text(i, val + 0.005, f'{val:.4f}', ha='center', fontweight='bold')

plt.show()

print("\nPerformance Summary Table:")
print(df_compare)