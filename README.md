<p align="center">
  <b>✦ Project II - COVID-19 Economic Analysis ✦</b><br>
  ניתוח השפעת הקורונה על מדדי כלכלה עולמיים (GDP, אבטלה ו-CPI)
</p>

---

## 📘 Supervised Learning – Regression Problem

הפרויקט מנתח את השפעת מגפת הקורונה על הכלכלה העולמית באמצעות נתוני COVID-19 ונתונים כלכליים כמו:
**GDP**, **Unemployment**, ו-**CPI**.

---

## 📊 מטרת הפרויקט
המטרה היא לחזות את **התוצר המקומי הגולמי (GDP)** של מדינות שונות,
בהתבסס על נתוני הקורונה והמדדים הכלכליים הנלווים.

---

## 🧮 קוד מלא לניתוח ו-Model Training

```python
# project_2
# Project by Israel Fadlon

# ============================================================
# 📘 Supervised Learning - Regression Problem: COVID-19 & GDP
# ============================================================

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib


# =======================
# 1️⃣ DATA PREPARATION
# =======================

data = "/content/drive/MyDrive/Classroom/עותק של Covid19_With_GDP_Values.csv"
df = pd.read_csv(data)

df = df.drop(columns=["Province/State", "Unnamed: 0"], errors='ignore')
df.replace(0, pd.NA, inplace=True)
df.dropna(subset=['CPI'], inplace=True)
df.drop_duplicates(inplace=True)

if 'Date' in df.columns:
    df['Year'] = pd.to_datetime(df['Date']).dt.year


# =======================
# 2️⃣ DATA EXPLORATION
# =======================

corr = df.corr(numeric_only=True)
corr_with_gdp = corr['GDP'].sort_values(ascending=False)
print("🔍 Correlation with GDP:\n", corr_with_gdp)

important_cols = ['GDP', 'Confirmed', 'Deaths', 'Recovered', 'Unemployment', 'CPI']

df_2021 = df[df['Year'] == 2021][important_cols]
df_2022 = df[df['Year'] == 2022][important_cols]

plt.figure(figsize=(6,5))
sns.heatmap(df_2021.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap – 2021")
plt.show()

plt.figure(figsize=(6,5))
sns.heatmap(df_2022.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap – 2022")
plt.show()


# @title

<div align= "left">

#(CorrelationAnalysis)       תובנות מתוך בדיקת
## 1. קשרים חזקים עם העמודה המרכזית (GDP)

- גם בשנת 2021 וגם בשנת 2022, ניתן לראות ש־GDP כמעט לא מתואם משמעותית עם אף משתנה אחר.
ערכי הקורלציה עם Unemployment ו־CPI הם בסביבות ‎-0.09‎ עד ‎-0.02‎ בלבד — כלומר, קשר חלש מאוד.

- 📍 פירוש:
זה מצביע על כך שבטווח הנתונים שלך, השינוי בתוצר (GDP) לא הוסבר באופן ישיר ע"י שיעור האבטלה או מדד המחירים (CPI).
כנראה שההשפעה של הקורונה, מדיניות ממשלתית ותנאים חיצוניים יצרו שונות שלא תלויה ישירות במדדים האלו.

###❤️ 2. קשרים בין מאפיינים כלכליים (features)

- הקשר בין Unemployment ל־CPI גם הוא כמעט אפסי (‎0.04–0.08‎).
כלומר, בתקופה הזו — לא נצפתה אינפלציה משמעותית כתוצאה מעלייה באבטלה או להפך.

- 📍 פירוש כלכלי אפשרי:
בשנים שלאחר הקורונה (2021–2022), השווקים היו במצב לא יציב. ממשלות הפעילו מדיניות מוניטרית ופיסקלית חזקה (כמו סובסידיות, הדפסות כסף, והורדות ריבית),
שפגעו במתאם “הטבעי” שבין אבטלה לאינפלציה (עקומת פיליפס).

###💬 3. השוואה בין השנים

בין 2021 ל־2022 אין שינוי דרמטי, אבל ניתן  לראות   Unemployment שמר על קשר שלילי עקבי עם GDP (בערך ‎-0.09‎).
כלומר, ככל שהתוצר גבוה יותר — שיעור האבטלה מעט נמוך יותר, גם אם החלש.

📍 פירוש:
הדבר תואם את ההיגיון הכלכלי — ככל שהמדינה מייצרת יותר (צמיחה כלכלית), כך נפתחות יותר משרות, והאבטלה יורדת.

📈 4. סיכום כולל
שנה	מתאם חזק ביותר	משמעות כלכלית
2021	‎GDP ↔ Unemployment (שלילי, ‎≈‎ -0.09‎)	התאוששות מהקורונה – יותר תוצר → פחות אבטלה
2022	‎GDP ↔ Unemployment (שלילי, ‎≈‎ -0.09‎)	מגמה דומה, אך פחות השפעה – כלכלה מתחילה להתייצב
שתיהן	‎CPI עם כל השאר – נמוך מאוד	אינפלציה לא הייתה קשורה ישירות לשינויי תוצר או תעסוקה


## 🧩 מסקנה כללית
הנתונים מראים שבתקופה הנבדקת (2021–2022),
הכלכלה הייתה בתהליך התאוששות אך עדיין לא במצב נורמלי, ולכן
הקורלציות בין המשתנים הכלכליים הן חלשות יחסית.

####📊 ניתן לפרש את זה כך:

הכלכלה הייתה מושפעת יותר מגורמים חיצוניים (כמו מדיניות ממשלתית או מגבלות בריאותיות)
ופחות מהקשרים הכלכליים ה”רגילים”.


