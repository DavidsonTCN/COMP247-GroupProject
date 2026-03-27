# ============================================================
# COMP 247 - Deliverable 1: Data Exploration
# Dataset: Toronto Traffic Collisions Open Data
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create plots folder
os.makedirs('plots', exist_ok=True)

# ============================================================
# 1. LOAD DATASET
# ============================================================
df = pd.read_csv(r'C:\Users\edwin\OneDrive\Documents\COMP247-GroupProject\Traffic_Collisions_Open_Data_2053198073974531286.csv')

print("=" * 60)
print("           COMP 247 - DATA EXPLORATION REPORT")
print("=" * 60)

# ============================================================
# 2. DATASET OVERVIEW
# ============================================================
print("\n>>> SECTION 1: DATASET OVERVIEW")
print("-" * 60)
print(f"Total Rows    : {df.shape[0]}")
print(f"Total Columns : {df.shape[1]}")
print(f"\nColumn Names  : {df.columns.tolist()}")

print("\nData Types:")
print(df.dtypes.to_string())

print("\nFirst 5 Rows:")
print(df.head().to_string())

# ============================================================
# 3. BASIC STATISTICS
# ============================================================
print("\n>>> SECTION 2: BASIC STATISTICS")
print("-" * 60)
print("\nNumeric Columns:")
print(df.describe().to_string())

print("\nCategorical Columns:")
print(df.describe(include='object').to_string())

# ============================================================
# 4. MISSING VALUES
# ============================================================
print("\n>>> SECTION 3: MISSING VALUES")
print("-" * 60)
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': round(missing_pct, 2)
}).sort_values('Missing %', ascending=False)

has_missing = missing_df[missing_df['Missing Count'] > 0]
if has_missing.empty:
    print("No missing values found!")
else:
    print(has_missing.to_string())

plt.figure(figsize=(8, 4))
cols = missing_df[missing_df['Missing Count'] > 0].index.tolist()
counts = missing_df[missing_df['Missing Count'] > 0]['Missing Count'].tolist()
plt.bar(cols, counts, color='tomato')
plt.title('Columns with Missing Values (Count)')
plt.ylabel('Missing Count')
plt.xlabel('Columns')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('plots/01_missing_values.png')
plt.show()

# ============================================================
# 5. TARGET VARIABLE
# ============================================================
print("\n>>> SECTION 4: TARGET VARIABLE (FATALITIES)")
print("-" * 60)
df['TARGET'] = (df['FATALITIES'] > 0).astype(int)
target_counts = df['TARGET'].value_counts()
target_pct = (df['TARGET'].value_counts(normalize=True) * 100).round(2)

print(f"Fatal     (1) : {target_counts[1]} ({target_pct[1]}%)")
print(f"Non-Fatal (0) : {target_counts[0]} ({target_pct[0]}%)")

plt.figure(figsize=(6, 4))
classes = ['Non-Fatal (0)', 'Fatal (1)']
counts = [target_counts[0], target_counts[1]]
colors = ['steelblue', 'tomato']
bars = plt.bar(classes, counts, color=colors)
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')

# Add count labels on top of each bar
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
             f'{count:,}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('plots/02_class_distribution.png')
plt.show()

# ============================================================
# 6. COLLISIONS OVER TIME
# ============================================================
print("\n>>> SECTION 5: COLLISIONS OVER TIME")
print("-" * 60)
print("Collisions per Year:")
print(df['OCC_YEAR'].value_counts().sort_index().to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df['OCC_YEAR'].value_counts().sort_index().plot(
    kind='line', marker='o', color='steelblue', ax=axes[0])
axes[0].set_title('Collisions per Year')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Count')
axes[0].grid(True)

df['OCC_MONTH'].value_counts().sort_index().plot(
    kind='bar', color='steelblue', ax=axes[1])
axes[1].set_title('Collisions per Month')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('plots/03_collisions_over_time.png')
plt.show()

# ============================================================
# 7. COLLISIONS BY DAY & HOUR
# ============================================================
print("\n>>> SECTION 6: COLLISIONS BY DAY & HOUR")
print("-" * 60)
print("Collisions by Day of Week:")
print(df['OCC_DOW'].value_counts().to_string())

print("\nCollisions by Hour:")
print(df['OCC_HOUR'].value_counts().sort_index().to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df['OCC_DOW'].value_counts().plot(kind='bar', color='steelblue', ax=axes[0])
axes[0].set_title('Collisions by Day of Week')
axes[0].set_xlabel('Day')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=45)

df['OCC_HOUR'].value_counts().sort_index().plot(kind='bar', color='steelblue', ax=axes[1])
axes[1].set_title('Collisions by Hour of Day')
axes[1].set_xlabel('Hour')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('plots/04_collisions_day_hour.png')
plt.show()

# ============================================================
# 8. COLLISION TYPES
# ============================================================
print("\n>>> SECTION 7: COLLISION TYPES")
print("-" * 60)
collision_types = pd.Series({
    'INJURY_COLLISIONS': (df['INJURY_COLLISIONS'] == 'YES').sum(),
    'FTR_COLLISIONS':    (df['FTR_COLLISIONS'] == 'YES').sum(),
    'PD_COLLISIONS':     (df['PD_COLLISIONS'] == 'YES').sum(),
    'FATALITIES':         df['FATALITIES'].sum()
})
print(collision_types.to_string())

plt.figure(figsize=(8, 5))
collision_types.plot(kind='bar', color=['steelblue', 'orange', 'green', 'tomato'])
plt.title('Total by Collision Type')
plt.ylabel('Total Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('plots/05_collision_types.png')
plt.show()

# ============================================================
# 9. VEHICLE / PERSON TYPES
# ============================================================
print("\n>>> SECTION 8: VEHICLE / PERSON TYPES INVOLVED")
print("-" * 60)
vehicle_types = pd.Series({
    'AUTOMOBILE': (df['AUTOMOBILE'] == 'YES').sum(),
    'MOTORCYCLE':  (df['MOTORCYCLE'] == 'YES').sum(),
    'PASSENGER':   (df['PASSENGER'] == 'YES').sum(),
    'BICYCLE':     (df['BICYCLE'] == 'YES').sum(),
    'PEDESTRIAN':  (df['PEDESTRIAN'] == 'YES').sum()
})
print(vehicle_types.to_string())

plt.figure(figsize=(8, 5))
vehicle_types.plot(kind='bar', color='steelblue')
plt.title('Vehicle / Person Types Involved')
plt.ylabel('Total Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('plots/06_vehicle_types.png')
plt.show()

# ============================================================
# 10. TOP NEIGHBOURHOODS & DIVISIONS
# ============================================================
print("\n>>> SECTION 9: TOP NEIGHBOURHOODS & DIVISIONS")
print("-" * 60)
print("Top 10 Neighbourhoods:")
print(df['NEIGHBOURHOOD_158'].value_counts().head(10).to_string())

print("\nCollisions by Division:")
print(df['DIVISION'].value_counts().to_string())

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

df[df['NEIGHBOURHOOD_158'] != 'NSA']['NEIGHBOURHOOD_158'].value_counts().head(10).plot(
    kind='bar', color='steelblue', ax=axes[0])
axes[0].set_title('Top 10 Neighbourhoods')
axes[0].set_xlabel('Neighbourhood')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=45)

df[df['DIVISION'] != 'NSA']['DIVISION'].value_counts().plot(
    kind='bar', color='steelblue', ax=axes[1])
axes[1].set_title('Collisions by Division')
axes[1].set_xlabel('Division')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('plots/07_neighbourhoods_divisions.png')
plt.show()

# ============================================================
# 11. CORRELATION HEATMAP
# ============================================================
print("\n>>> SECTION 10: CORRELATION HEATMAP")
print("-" * 60)
numeric_df = df.select_dtypes(include=[np.number])
print("Numeric columns used:")
print(numeric_df.columns.tolist())

plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f',
            cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap (Numeric Features)')
plt.tight_layout()
plt.savefig('plots/08_correlation_heatmap.png')
plt.show()

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("              EXPLORATION SUMMARY")
print("=" * 60)
print(f"Total Records       : {df.shape[0]}")
print(f"Total Features      : {df.shape[1]}")
print(f"Fatal Collisions    : {target_counts[1]} ({target_pct[1]}%)")
print(f"Non-Fatal Collisions: {target_counts[0]} ({target_pct[0]}%)")
print(f"Missing Values      : {df.isnull().sum().sum()}")
print(f"Years Covered       : {df['OCC_YEAR'].min()} - {df['OCC_YEAR'].max()}")
print("=" * 60)
print("\n✅ Deliverable 1 Complete! Plots saved to 'plots' folder.")