# project_2
# project by israel fadlon


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# DATA Preparation 


# 1 #

# טעינת הנתונים למודל
data = "/content/drive/MyDrive/Classroom/עותק של Covid19_With_GDP_Values.csv"
df = pd.read_csv(data)
df.head()
df.columns

# הסרה של עמודות לא רלוונטיות שמפריעות למודל
df = df.drop(columns=["Province/State", "Unnamed: 0"], errors='ignore')

# 2 #

 # טיפול בערכים חסרים
df.replace(0, pd.NA, inplace=True)
df.dropna(subset=['CPI'], inplace=True)

# 3 #

 # הסרת כפילויות
df.drop_duplicates(inplace=True)

# 4 #

# אין צורך ב בפונקצית get.dummies

# שמירה על עמודות YEAR
if 'Date' in df.columns:
    df['Year'] = pd.to_datetime(df['Date']).dt.year

# 5 #


 # DATA EXPLORATION 
 # שמירה על עמודות YEAR
if 'Date' in df.columns:
    df['Year'] = pd.to_datetime(df['Date']).dt.year
# GDP בדיקת מתאם
corr = df.corr(numeric_only=True)
corr_with_gdp = corr['GDP'].sort_values(ascending=False)
corr_with_gdp


        




# 1 # 

# GDP בדיקת מתאם
corr = df.corr(numeric_only=True)
corr_with_gdp = corr['GDP'].sort_values(ascending=False)
corr_with_gdp
# רשימת העמודות החשובות בלבד
important_cols = ['GDP', 'Confirmed', 'Deaths', 'Recovered', 'Unemployment', 'CPI']

#  סינון לפי שנה לדיוק המודל
df_2021 = df[df['Year'] == 2021][important_cols]
df_2022 = df[df['Year'] == 2022][important_cols]

# חישוב הקורלציה
corr_2021 = df_2021.corr(numeric_only=True)
corr_2022 = df_2022.corr(numeric_only=True)

# 2 # 

# Heatmap לשנת 2021
plt.figure(figsize=(6,5))
sns.heatmap(corr_2021, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap – 2021 (Selected Features)")
plt.show()

# Heatmap לשנת 2022
plt.figure(figsize=(6,5))
sns.heatmap(corr_2022, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap – 2022 (Selected Features)")
plt.show()

