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

# Get descriptive statistics for Amount
amount_summary = df_cc_fraud['Amount'].describe()

print("--- Transaction Amount Range Summary ---")
print(f"Minimum: ${amount_summary['min']:.2f}")
print(f"Maximum: ${amount_summary['max']:.2f}")
print(f"Median:  ${amount_summary['50%']:.2f}")
print(f"Mean:    ${amount_summary['mean']:.2f}")
print(f"Range:   ${amount_summary['max'] - amount_summary['min']:.2f}")

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
    y_pred = rf.predict(X_test_scaled)
    y_probs = rf.predict_proba(X_test_scaled)[:, 1] # Probabilities for the 'Fraud' class

    # Calculate Metrics
    auprc = average_precision_score(y_test, y_probs)

    # Print RF Results
    print(f"Total Test Samples: {len(y_test)} | Actual Fraud in Test: {y_test.sum()}")
    print(f"AUPRC Score: {auprc:.4f}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate (0)', 'Fraud (1)']))

    # Confusion Matrix Visualization with Seaborn Heatmap
    plt.figure(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False)
    plt.title(f'Random Forest Confusion Matrix: {subset_names[i]}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

# ---------------Stratified Cross-Validation on Subset 3---------------

# Define the Stratified Split
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Re-prepare Subset 3 features/target
X3 = subset_3.drop('Class', axis=1)
y3 = subset_3['Class']

# Run CV specifically for AUPRC (average_precision)
cv_auprc = cross_val_score(rf, X3, y3, cv=skf, scoring='average_precision')

print(f"Subset 3 AUPRC Stability: {cv_auprc.mean():.4f} (+/- {cv_auprc.std():.4f})")

# ---------------Feature Importance from Random Forest Model---------------

# Extract importance from trained Random Forest model
importances = pd.Series(rf.feature_importances_, index=X_train.columns)
importances = importances.sort_values(ascending=False)

# Plot the top 10 features
importances.head(10).plot(kind='barh', color='skyblue')
plt.title("Feature Importance - Top 10")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# ---------------Gradient Boosting Model Training and Testing---------------

print(f"\n{'#'*30}\nStarting Gradient Boosting Evaluation\n{'#'*30}")

# Initialize a list to store GB results for a final comparison if desired
gb_auprc_results = []

for i, data in enumerate(subsets):
    print(f"\n{'='*20} GB: Evaluating {subset_names[i]} {'='*20}")

    # Feature/Target Separation
    X_gb = data.drop('Class', axis=1)
    y_gb = data['Class']

    # 80-20 Stratified Split (Matching your RF split logic)
    X_train, X_test, y_train, y_test = train_test_split(
        X_gb, y_gb, test_size=0.20, random_state=42, stratify=y_gb
    )

    # Scaling (Preventing Data Leakage by fitting only on Train)
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled['Amount'] = scaler.fit_transform(X_train[['Amount']])
    X_test_scaled['Amount'] = scaler.transform(X_test[['Amount']])

    # Train Gradient Boosting Classifier
    # We use common defaults; max_depth is lower than RF to prevent overfitting
    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gbc.fit(X_train_scaled, y_train)

    # Get Binary Predictions and Probabilities
    y_pred_gb = gbc.predict(X_test_scaled)
    y_probs_gb = gbc.predict_proba(X_test_scaled)[:, 1] 

    # Calculate Metrics (Identical to your RF metrics)
    auprc_gb = average_precision_score(y_test, y_probs_gb)
    gb_auprc_results.append(auprc_gb)

    # Print Results in the same format as RF
    print(f"Total Test Samples: {len(y_test)} | Actual Fraud in Test: {y_test.sum()}")
    print(f"AUPRC Score: {auprc_gb:.4f}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred_gb, target_names=['Legitimate (0)', 'Fraud (1)']))

    # Confusion Matrix Visualization with Seaborn Heatmap
    plt.figure(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False)
    plt.title(f'Gradient Boosting Confusion Matrix: {subset_names[i]}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()