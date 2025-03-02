from import_data import *

# Use the columnnames and units from df
df.columnnames = [
    'Cement',
    'Blast Furnace Slag',
    'Fly Ash',
    'Water',
    'Superplasticizer',
    'Coarse Aggregate',
    'Fine Aggregate',
    'Age',
    'Concrete compressive strength'
]

df.units = [
    '(kg in a m³ mixture)',
    '(kg in a m³ mixture)',
    '(kg in a m³ mixture)',
    '(kg in a m³ mixture)',
    '(kg in a m³ mixture)',
    '(kg in a m³ mixture)',
    '(kg in a m³ mixture)',
    '(day)',
    '(MPa)'
]

# 1. Histograms for all attributes in one
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
axes = axes.ravel()

for i, (column, unit) in enumerate(zip(df.columnnames, df.units)):
    axes[i].hist(X[:, i], bins=20, edgecolor='k', color='skyblue', density=True)
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f"{column} {unit}")
    
    mean = X[:, i].mean()
    q25 = np.percentile(X[:, i], 25)
    median = np.median(X[:, i])
    q75 = np.percentile(X[:, i], 75)
    
    axes[i].axvline(mean, color='red', linestyle='dashed', linewidth=1, label='Mean')
    axes[i].axvline(median, color='green', linestyle='dashed', linewidth=1, label='Median')
    axes[i].axvline(q25, color='orange', linestyle='dashed', linewidth=1, label='25% Quartile')
    axes[i].axvline(q75, color='orange', linestyle='dashed', linewidth=1, label='75% Quartile')
    
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center')
plt.tight_layout()
plt.show()

# 2. Boxplot for all attributes in one
plt.figure(figsize=(14, 10))
plt.boxplot(X, labels=[f"{col}\n{unit}" for col, unit in zip(df.columnnames, df.units)])
plt.title('Boxplots of Concrete Components')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 3. Nine individual boxplots with details
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.ravel()

for i, (column, unit) in enumerate(zip(df.columnnames, df.units)):
    axes[i].boxplot(X[:, i])

    axes[i].set_title(column)
    axes[i].set_ylabel(unit)
    
    mean = X[:, i].mean()
    q25 = np.percentile(X[:, i], 25)
    median = np.median(X[:, i])
    q75 = np.percentile(X[:, i], 75)
    
    # Display statistics on the boxplot
    stat_text = f"Mean: {mean:.2f}\nMedian: {median:.2f}\n25%: {q25:.2f}\n75%: {q75:.2f}"
    axes[i].text(1.05, 0.95, stat_text, transform=axes[i].transAxes, 
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8), 
                 verticalalignment='top', color='black')

plt.tight_layout()
plt.show()