import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Data from your successful Model 06 run
data = {
    'Feature': ['MMSCORE', 'NRGN', 'IL16', 'NPTX1', 'APOE4', 'VCAM1', 'Aβ40', 'ACHE', 
                'IGF1R', 'Aβ38', 'IL7', 'PRDX6', 'is_male', 'NPY', 'UCHL1'],
    'Impact': [1.2376, 0.6305, 0.5599, 0.4492, 0.4243, 0.4240, 0.4167, 0.4011, 
               0.3912, 0.3832, 0.3809, 0.3725, 0.3669, 0.3474, 0.3451]
}

df = pd.DataFrame(data)

# Setup the plot
plt.figure(figsize=(10, 8))
sns.set_theme(style="whitegrid")
palette = sns.color_palette("viridis", len(df))

# Create barplot
ax = sns.barplot(x='Impact', y='Feature', data=df, palette=palette)

# Add titles and labels
plt.title('Top 15 Predictors: MCI vs. Cognitively Normal (AUC: 0.8963)', fontsize=15, fontweight='bold')
plt.xlabel('Absolute Coefficient Weight (L1 Lasso)', fontsize=12)
plt.ylabel('Biomarker / Clinical Feature', fontsize=12)

# Save the plot
plt.tight_layout()
plt.savefig('reports/figures/final_feature_importance.png', dpi=300)
print("✅ Final plot saved to reports/figures/final_feature_importance.png")
plt.show()