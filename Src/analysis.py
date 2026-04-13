
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# Settings
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# Fixed Kaggle Path
base_path = "/kaggle/input/datasets/kidistbelaygetachew/ecommerce"
file_name = "ecommerce_data_small-1 (1).csv"
full_path = os.path.join(base_path, file_name)

try:
    # Load and clean data
    data = pd.read_csv(full_path)
    data = data.drop_duplicates()
    data = data.loc[:, ~data.columns.duplicated()]
    print(f"Data successfully loaded from: {full_path}")
except Exception as e:
    print(f"Error: {e}")
    # Fallback for different Kaggle directory structures
    print("Attempting secondary path...")
    full_path = f"/kaggle/input/{file_name}"
    try:
        data = pd.read_csv(full_path)
    except:
        print("File not found. Please check the 'Data' tab in Kaggle for the correct path.")
        exit()

# Processing Date and Day
data['purchase_date'] = pd.to_datetime(data['purchase_date'])
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
data['day_name'] = pd.Categorical(data['purchase_date'].dt.day_name(), categories=days, ordered=True)

# Feature Engineering
data['loyalty_index'] = (data['customer_tenure_months'] * data['satisfaction_score']) / 100
data['customer_group'] = pd.cut(data['customer_tenure_months'], 
                                bins=[-1, 6, 12, 24, 60, 999], 
                                labels=['Newbie', 'Rising', 'Regular', 'Loyal', 'Veteran'])

# Outlier Calculation (IQR Method)
q1, q3 = data['purchase_amount'].quantile([0.25, 0.75])
iqr = q3 - q1
cutoff = q3 + (1.5 * iqr)

# Regional Aggregation
regional_spend = data.pivot_table(index='location', 
                                  columns='category', 
                                  values='purchase_amount', 
                                  aggfunc='mean').fillna(0)

# Visualization
fig, ax = plt.subplots(2, 2, figsize=(16, 12))
plt.subplots_adjust(hspace=0.4)

# 1. Spend Distribution
sns.histplot(data['purchase_amount'], bins=40, kde=True, ax=ax[0,0], color='royalblue')
ax[0,0].axvline(cutoff, color='red', linestyle='--', label='High Spend Limit')
ax[0,0].set_title('Where is the money? (Spend Distribution)')
ax[0,0].legend()

# 2. Satisfaction by Tenure
sns.boxplot(data=data, x='customer_group', y='satisfaction_score', palette='Set2', ax=ax[0,1])
ax[0,1].set_title('Are old customers happier?')

# 3. Weekly Trends
sns.lineplot(data=data, x='day_name', y='purchase_amount', estimator='sum', marker='o', ax=ax[1,0])
ax[1,0].set_title('Total Sales by Day of Week')
ax[1,0].tick_params(axis='x', rotation=30)

# 4. Regional Spending Heatmap
sns.heatmap(regional_spend, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax[1,1])
ax[1,1].set_title('Avg Spend: Region vs Category')

plt.show()
