print("importing")
from import_data import *
import matplotlib.pyplot as plt
import seaborn as sns
import re
print("ended")

# Set the color palette to pastel colors
sns.set_palette("pastel")

raw_data, X, y, N, M, C, classNames, attributeNames, y2 = import_data()
raw_data = raw_data.iloc[:, :-1]

# Plot histograms
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.ravel()

for i in range(len(raw_data.columns)):
    column_name = raw_data.columns[i]
    
    column_name = re.sub(r"\(component \d+\)", "", column_name).strip()
    
    column_data = raw_data.iloc[:, i]
    
    axes[i].hist(column_data, bins=20, edgecolor='k', color=sns.color_palette("pastel")[i % len(sns.color_palette("pastel"))], density=False)
    axes[i].set_ylabel('Count')
    axes[i].set_title(column_name)
    
    mean = column_data.mean()
    q25 = column_data.quantile(0.25)
    median = column_data.median()
    q75 = column_data.quantile(0.75)
    
    axes[i].axvline(mean, color='red', linestyle='dashed', linewidth=1, label='Mean')
    axes[i].axvline(median, color='green', linestyle='dashed', linewidth=1, label='Median')
    axes[i].axvline(q25, color='orange', linestyle='dashed', linewidth=1, label='25% Quartile')
    axes[i].axvline(q75, color='orange', linestyle='dashed', linewidth=1, label='75% Quartile')

for j in range(len(raw_data.columns), len(axes)):
    axes[j].set_visible(False)
    
plt.tight_layout()

plt.savefig('/Users/linusjuni/Desktop/histograms.png')

plt.show()

# Create boxplots with pastel colors
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.ravel()

for i in range(len(raw_data.columns)):
    column_name = raw_data.columns[i]
    
    column_name = re.sub(r"\(component \d+\)", "", column_name).strip()
    
    column_data = raw_data.iloc[:, i]
    
    axes[i].boxplot(column_data, vert=False, patch_artist=True, 
                    boxprops=dict(facecolor=sns.color_palette("pastel")[i % len(sns.color_palette("pastel"))]),
                    medianprops=dict(color='black'))
    axes[i].set_title(column_name)
    axes[i].set_xlabel('Values')
    
for j in range(len(raw_data.columns), len(axes)):
    axes[j].set_visible(False)
    
plt.tight_layout()

plt.savefig('/Users/linusjuni/Desktop/boxplots.png')

plt.show()

# Create correlation scatter plots
fig, axes = plt.subplots(3, 3, figsize=(14, 12))
axes = axes.ravel()

for i in range(len(raw_data.columns)):
    column_name = raw_data.columns[i]
    
    column_name_title = re.sub(r"\(component \d+\)", "", column_name).strip()
    column_name_axis = re.sub(r"\(.*?\)", "", column_name).strip()

    column_data = raw_data.iloc[:, i]
    
    axes[i].scatter(column_data, y2, alpha=0.6, 
                   color=sns.color_palette("pastel")[i % len(sns.color_palette("pastel"))], 
                   edgecolors='none')
    
    axes[i].set_xlabel(column_name_title)
    axes[i].set_ylabel('Concrete Compressive Strength (MPa)')
    axes[i].set_title(f'{column_name_axis} vs. Strength')
    
    axes[i].grid(True, linestyle='--', alpha=0.6)

for j in range(len(raw_data.columns), len(axes)):
    axes[j].set_visible(False)
    
plt.tight_layout()

plt.savefig('/Users/linusjuni/Desktop/correlation_plots.png')

plt.show()

# Create correlation matrix
correlation_data = raw_data.copy()

corr_matrix = correlation_data.corr()

corr_matrix.columns = [re.sub(r"\(.*?\)", "", col).strip() for col in corr_matrix.columns]
corr_matrix.index = [re.sub(r"\(.*?\)", "", idx).strip() for idx in corr_matrix.index]

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
            annot=True, fmt='.2f', square=True, linewidths=.5,
            cbar_kws={"shrink": .8})

plt.title('Correlation Matrix of Attributes', fontsize=16)
plt.tight_layout()

plt.savefig('/Users/linusjuni/Desktop/correlation_matrix.png')

plt.show()

# Create covariance matrix
covariance_data = raw_data.copy()
covariance_data['Concrete Compressive Strength'] = y2  # Include target variable

cov_matrix = covariance_data.cov()

cov_matrix.columns = [re.sub(r"\(.*?\)", "", col).strip() for col in cov_matrix.columns]
cov_matrix.index = [re.sub(r"\(.*?\)", "", idx).strip() for idx in cov_matrix.index]

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(cov_matrix, dtype=bool), k=1)

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(cov_matrix, mask=mask, cmap=cmap, center=0,
            annot=True, fmt='.2f', square=True, linewidths=.5,
            cbar_kws={"shrink": .8})

plt.title('Covariance Matrix of Attributes', fontsize=16)
plt.tight_layout()

plt.savefig('/Users/linusjuni/Desktop/covariance_matrix.png')

plt.show()