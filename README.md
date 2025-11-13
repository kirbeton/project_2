# project_2
# project by israel fadlon


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# טעינת הנתונים למודל
data = "/content/drive/MyDrive/Classroom/עותק של Covid19_With_GDP_Values.csv"
df = pd.read_csv(data)
df.head()
df.columns


# DATA Preparation 

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

 # DATA EXPLORATION 
 # שמירה על עמודות YEAR
if 'Date' in df.columns:
    df['Year'] = pd.to_datetime(df['Date']).dt.year
# GDP בדיקת מתאם
corr = df.corr(numeric_only=True)
corr_with_gdp = corr['GDP'].sort_values(ascending=False)
corr_with_gdp


        


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1 # 
# GDP בדיקת מתאם
corr = df.corr(numeric_only=True)
corr_with_gdp = corr['GDP'].sort_values(ascending=False)
corr_with_gd
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




# 3 #
# (CorrelationAnalysis)       תובנות מתוך בדיקת 
## 1. קשרים חזקים עם העמודה המרכזית (GDP)

- גם בשנת 2021 וגם בשנת 2022, ניתן לראות ש־GDP כמעט לא מתואם משמעותית עם אף משתנה אחר. 
ערכי הקורלציה עם Unemployment ו־CPI הם בסביבות ‎-0.09‎ עד ‎-0.02‎ בלבד — כלומר, קשר חלש מאוד.

- 📍 פירוש:
זה מצביע על כך שבטווח הנתונים שלך, השינוי בתוצר (GDP) לא הוסבר באופן ישיר ע"י שיעור האבטלה או מדד המחירים (CPI).
כנראה שההשפעה של הקורונה, מדיניות ממשלתית ותנאים חיצוניים יצרו שונות שלא תלויה ישירות במדדים האלו.

### ❤️ 2. קשרים בין מאפיינים כלכליים (features)

- הקשר בין Unemployment ל־CPI גם הוא כמעט אפסי (‎0.04–0.08‎).
כלומר, בתקופה הזו — לא נצפתה אינפלציה משמעותית כתוצאה מעלייה באבטלה או להפך.

- 📍 פירוש כלכלי אפשרי:
בשנים שלאחר הקורונה (2021–2022), השווקים היו במצב לא יציב. ממשלות הפעילו מדיניות מוניטרית ופיסקלית חזקה (כמו סובסידיות, הדפסות כסף, והורדות ריבית),
שפגעו במתאם “הטבעי” שבין אבטלה לאינפלציה (עקומת פיליפס).

### 💬 3. השוואה בין השנים

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

### 📊 ניתן לפרש את זה כך:

הכלכלה הייתה מושפעת יותר מגורמים חיצוניים (כמו מדיניות ממשלתית או מגבלות בריאותיות)
ופחות מהקשרים הכלכליים ה”רגילים”.






