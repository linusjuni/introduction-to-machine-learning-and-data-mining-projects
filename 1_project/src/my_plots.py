from import_data import *
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Set the color palette to pastel colors
sns.set_palette("pastel")

raw_data, X, y, N, M, C, classNames, attributeNames = import_data()
raw_data = raw_data.iloc[:, :-1]

fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.ravel()

for i in range(len(raw_data.columns)):
    column_name = raw_data.columns[i]
    
    # Remove anything inside parentheses (including the parentheses)
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

# Save the plot as a PNG file on the desktop
plt.savefig('/Users/linusjuni/Desktop/histograms.png')

plt.show()

# Create boxplots with pastel colors
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.ravel()

for i in range(len(raw_data.columns)):
    column_name = raw_data.columns[i]
    
    # Remove anything inside parentheses (including the parentheses)
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

# Save the boxplot as a PNG file on the desktop
plt.savefig('/Users/linusjuni/Desktop/boxplots.png')

plt.show()