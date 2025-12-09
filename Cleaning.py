import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv(r"Project.py/lung_cancer_dataset.csv")

print(df.info())
 
df.drop(columns=['patient_id'], inplace=True)
df['alcohol_consumption'].fillna('Not drinking', inplace=True)
print(df.info())
print(df.describe())
print(df.head())
print(df['alcohol_consumption'].unique())
print(df.info())
print(df['lung_cancer'].value_counts())

sns.boxplot(x=df['pack_years'])
plt.title("Outliers in pack_years")
plt.show()





# corr_target = df.corr()['lung_cancer'].sort_values(ascending=False)
# print("\nCorrelation with Lung Cancer:\n")
# print(corr_target)
# print(df.head())

df.to_csv("processed_lung_cancer.csv", index=False)
