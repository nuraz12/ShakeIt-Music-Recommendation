import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("tracks_features.csv")
# Sadece numerik kolonları seç
numeric_data = data.select_dtypes(include='number')

# Korelasyon matrisini hesapla
corr_matrix = numeric_data.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
print(corr_matrix)
plt.show()