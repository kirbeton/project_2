# project_2
project by israel fadlon


import pandas as pd

# טעינת הנתונים למודל
data = "/content/drive/MyDrive/Classroom/עותק של Covid19_With_GDP_Values.csv"
df = pd.read_csv(data)
df.head()
df.columns


# # # DATA EXPOLRATION # # #

# 1 #


 # בדיקת שמות העמודות
df.info()
df.columns

# 2 #

 # טיפול בערכים חסרים
df.replace(0, pd.NA, inplace=True)
df.dropna(subset=['CPI'], inplace=True)

# 3 #

 # הסרת כפילויות
df.drop_duplicates(inplace=True)

# 4 #

 # המרת משתנים קטגורים לדיוק המודל
df = pd.get_dummies(df, columns=['Country/Region', 'Province/State'], drop_first=True)

# שמירה על עמודות YEAR
if 'Date' in df.columns:
    df['Year'] = pd.to_datetime(df['Date']).dt.year

# 5 #


# GDP בדיקת מתאם
corr = df.corr(numeric_only=True)
corr_with_gdp = corr['GDP'].sort_values(ascending=False)
corr_with_gdp


        # DATA EXPLORATION #


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# רשימת העמודות החשובות בלבד
important_cols = ['GDP', 'Confirmed', 'Deaths', 'Recovered', 'Unemployment', 'CPI']

#  סינון לפי שנה לדיוק המודל
df_2021 = df[df['Year'] == 2021][important_cols]
df_2022 = df[df['Year'] == 2022][important_cols]

# חישוב הקורלציה
corr_2021 = df_2021.corr(numeric_only=True)
corr_2022 = df_2022.corr(numeric_only=True)

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


