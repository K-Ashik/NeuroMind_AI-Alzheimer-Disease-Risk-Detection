import zipfile
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- SETUP ---
output_dir = "Alzheimers_EDA"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. LOAD DATA
print(">>> Loading TADPOLE Dataset...")
with zipfile.ZipFile('Tadpole_data.zip', 'r') as z:
    with z.open('TADPOLE_D1_D2.csv') as f:
        df = pd.read_csv(f, low_memory=False)

print(f"Original Shape: {df.shape}")

# 2. INTELLIGENT RENAMING (The Fix)
# We search for the specific UPENN biomarker columns and rename them to standard names.
print("\n--- STANDARDIZING COLUMN NAMES ---")

# Define the keywords we are looking for
search_map = {
    'ABETA': 'ABETA_UPENNBIOMK9',
    'TAU':   'TAU_UPENNBIOMK9',
    'PTAU':  'PTAU_UPENNBIOMK9'
}

rename_dict = {}
for standard_name, keyword in search_map.items():
    # Find any column that contains the keyword
    matches = [col for col in df.columns if keyword in col]
    
    if len(matches) > 0:
        # Take the first match (usually the most comprehensive batch)
        original_col = matches[0]
        rename_dict[original_col] = standard_name
        print(f"✔ Found {standard_name} in: '{original_col}'")
    else:
        print(f"❌ Warning: Could not find column for {standard_name}")

# Apply the renaming
if rename_dict:
    df = df.rename(columns=rename_dict)
    print("Renaming Complete.")
else:
    print("CRITICAL ERROR: No protein data found. Please check dataset.")
    exit()

# DEFINING TARGET COLUMNS (Now safe to use)
protein_cols = ['ABETA', 'TAU', 'PTAU'] 
mri_cols = ['Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform']
clinical_cols = ['DX_bl', 'AGE', 'PTGENDER', 'APOE4', 'MMSE', 'CDR']

# 3. DATA CLEANING
print("\n--- CLEANING DATA ---")

# Filter 1: Drop rows where Protein data is missing
clean_df = df.dropna(subset=protein_cols).copy() # .copy() prevents SettingWithCopyWarning
print(f"Rows with Protein Data: {clean_df.shape[0]}")

# Filter 2: Fix "Messy" Numbers (e.g., ">1700")
def clean_numeric(x):
    if pd.isna(x): return np.nan
    x = str(x) # Ensure string
    if '>' in x: return float(x.replace('>', '')) + 1
    if '<' in x: return float(x.replace('<', '')) - 1
    try:
        return float(x)
    except ValueError:
        return np.nan

for col in protein_cols:
    clean_df[col] = clean_df[col].apply(clean_numeric)
    
# Drop rows that became NaN after cleaning
clean_df = clean_df.dropna(subset=protein_cols)
print(f"Cleaned Numerical Rows: {clean_df.shape[0]}")

# 4. EDA VISUALIZATION

# Chart 1: The "AD Signature" (Amyloid vs Tau)
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=clean_df, 
    x='ABETA', 
    y='TAU', 
    hue='DX_bl', # Diagnosis at baseline
    palette='viridis',
    alpha=0.6
)
plt.title("The Alzheimer's Signature: Amyloid vs. Tau", fontsize=14)
plt.xlabel("Amyloid Beta (Low = Bad)")
plt.ylabel("Total Tau (High = Bad)")
plt.savefig(f"{output_dir}/01_amyloid_tau_scatter.png")
plt.close()

# Chart 2: Hippocampus Atrophy
# Check if Hippocampus exists, if not, skip plot 
if 'Hippocampus' in clean_df.columns:
    plt.figure(figsize=(8, 6))
    clean_df['Hippocampus'] = pd.to_numeric(clean_df['Hippocampus'], errors='coerce')
    sns.boxplot(x='DX_bl', y='Hippocampus', data=clean_df, palette="Set2")
    plt.title("Hippocampal Volume by Diagnosis", fontsize=14)
    plt.ylabel("Volume (mm3)")
    plt.xticks(rotation=15)
    plt.savefig(f"{output_dir}/02_hippocampus_boxplot.png")
    plt.close()
else:
    print("Skipping Hippocampus plot (column not found)")

# Chart 3: Genetic Risk (APOE4)
if 'APOE4' in clean_df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(x='APOE4', hue='DX_bl', data=clean_df, palette="coolwarm")
    plt.title("Genetics: APOE4 Copies vs. Diagnosis", fontsize=14)
    plt.xlabel("Number of APOE4 Alleles (0, 1, 2)")
    plt.savefig(f"{output_dir}/03_apoe_risk.png")
    plt.close()

print(f"\n>>> EDA COMPLETE. Check the '{output_dir}' folder.")
