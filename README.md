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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
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

```


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















# LINEAR REGRESSION MULTIE MODEL TRAINING

```
# טעינה והכנה בסיסית של הנתונים
if 'Year' not in df.columns and 'Date' in df.columns:
    df['Year'] = pd.to_datetime(df['Date']).dt.year
df = df[df['Year'].isin([2021, 2022])].copy() 

# הגדרת העמודות החשובות למודל
model_numeric_cols = ['Confirmed', 'Deaths', 'Recovered', 'Unemployment', 'CPI', 'GDP']

# הגדרת העמודות למספרים
for col in model_numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# טיפול בערכים חסרים
df[model_numeric_cols] = df[model_numeric_cols].fillna(df[model_numeric_cols].mean())

# בנייצ מודל לשנים 2021 ,2022 
for year in [2021, 2022]:
    print(f"\n🧩 ===== ניתוח עבור השנה {year} ====")

    df_year = df[df['Year'] == year]

    #הגדרת משתנים למודל 
    X = df_year[['Confirmed', 'Deaths', 'Recovered', 'Unemployment', 'CPI']]
    y = df_year['GDP']

    #  נרמול לסטיית תקן 1 וממוצע 0
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #  חילוק של הנתונים ל 70%  אימון המודל ,ו 30% בדיקה אמיתית
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    # 2 # 

    # מחשב את הקשר בין המשתנים (X) לבין התוצר (GDP).
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 3 #

    # תוצאות
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')

    print(f"📊 Train R²: {train_score:.3f}")
    print(f"📈 Test R²: {test_score:.3f}")
    print("🔁 Cross Validation R² scores:", cv_scores)
    print("⭐ Average R²:", np.mean(cv_scores))


#🧩 ===== ניתוח עבור השנה 2021 ====
#📊 Train R²: 0.052
#📈 Test R²: 0.360
#🔁 Cross Validation R² scores: [-3.62191755e-02  4.37463648e-01  3.09127470e-01 -6.32888672e-01
# -8.91820078e+01]
#⭐ Average R²: -17.820904911486192

#🧩 ===== ניתוח עבור השנה 2022 ====
#📊 Train R²: 0.054
#📈 Test R²: 0.257
#🔁 Cross Validation R² scores: [-3.80816578e-02  3.98185317e-01 -2.47026004e-01 -1.23598581e-01
# -5.64311892e+01]
#⭐ Average R²: -11.288342021292618


# Multi model training 

# 1 # 


# מאפיינים ותיוג
X = df[['Confirmed', 'Deaths', 'Recovered', 'Unemployment', 'CPI']]
y = df['GDP']

# נרמול
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# מודל ליניארי רגיל
lin_model = LinearRegression()

# K-Fold Cross Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(lin_model, X_scaled, y, cv=kfold, scoring='r2')

print("Linear Regression CV R² Scores:", scores)
print("Average R²:", np.mean(scores))

# 2 # 


# להוסיף רגולריזציה (L2 penalty) שמונעת מהמודל “להגזים” עם מקדמים גדולים מדי.
# זה עוזר במקרים של multicollinearity או נתונים רועשים.
alphas = np.logspace(-3, 3, 50)


ridge_model = RidgeCV(alphas=alphas, cv=5, scoring='r2')
ridge_model.fit(X_scaled, y)

# בדיקה
print(f"Optimal alpha (λ): {ridge_model.alpha_}")
print(f"R² Score (using best λ): {ridge_model.score(X_scaled, y):.3f}")

# 3 # 



# טווח ערכים של λ (אלפא)
alphas = np.logspace(-3, 3, 50)

lasso_model = LassoCV(alphas=alphas, cv=5, random_state=42)
lasso_model.fit(X_scaled, y)

print(f"Optimal alpha (λ): {lasso_model.alpha_}")
print(f"R² Score (using best λ): {lasso_model.score(X_scaled, y):.3f}")


# פה אפשר גם לראות כמה מקדמים נשארו עם !=0
print("Number of features kept:", np.sum(lasso_model.coef_ != 0))

# 4 # 

# לאפשר למודל “להתכופף” — כלומר לזהות קשרים לא ליניאריים בין המשתנים


degrees = [1, 2, 3, 4, 5]
avg_scores = []

for d in degrees:
    poly_model = make_pipeline(PolynomialFeatures(d), LinearRegression())
    score = cross_val_score(poly_model, X_scaled, y, cv=5, scoring='r2').mean()
    avg_scores.append(score)
    print(f"Degree {d} → Mean R²: {score:.3f}")

# בחירת הדרגה הטובה ביותר
best_degree = degrees[np.argmax(avg_scores)]
print(f"\nOptimal Polynomial Degree: {best_degree}")





# אותם מאפיינים ותיוגים
X = df[['Confirmed', 'Deaths', 'Recovered', 'Unemployment', 'CPI']]
y = df['GDP']

# נרמול
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# הגדרות אופטימליות מהשלב הקודם
ridge_opt_alpha =  ridge_model.alpha_
lasso_opt_alpha = lasso_model.alpha_
best_degree = best_degree  # מהפולינומי הקודם

# בניית המודלים הסופיים
models = {
    "Linear Regression": LinearRegression(),
    "RidgeCV": RidgeCV(alphas=[ridge_opt_alpha]),
    "LassoCV": LassoCV(alphas=[lasso_opt_alpha]),
    f"Polynomial (deg={best_degree})": make_pipeline(PolynomialFeatures(best_degree), LinearRegression())
}



print("===== Optimal Parameters & Coefficients =====")
for name, model in models.items():
    model.fit(X_scaled, y)
    print(f"\n🔹 {name}")
    if hasattr(model, "alpha_"):
        print(f"Optimal λ (alpha): {model.alpha_}")
    if hasattr(model, "coef_"):
        print("Beta Coefficients:")
        for feature, coef in zip(['Confirmed', 'Deaths', 'Recovered', 'Unemployment', 'CPI'], model.coef_):
            print(f"  {feature}: {coef:.4f}")
    elif hasattr(model[-1], "coef_"):  # למודלים עם pipeline
        print("Beta Coefficients (Polynomial):")
        print(model[-1].coef_)

results = []

for name, model in models.items():
    y_pred = model.predict(X_scaled)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = model.score(X_scaled, y)
    results.append([name, mae, mse, rmse, r2])

results_df = pd.DataFrame(results, columns=['Model', 'MAE', 'MSE', 'RMSE', 'R²'])
print("\n===== Model Evaluation =====")
print(results_df)

plt.figure(figsize=(8,5))
plt.bar(results_df['Model'], results_df['R²'], color='skyblue')
plt.title('📈 Model Accuracy (R² Comparison)')
plt.ylabel('R² Score')
plt.xticks(rotation=30)
plt.show()


best_model_name = results_df.loc[results_df['R²'].idxmax(), 'Model']
print(f"\n✅ Best Performing Model: {best_model_name}")


best_model = models[best_model_name]
best_model.fit(X_scaled, y)
print("\n🏁 Final model trained on full dataset!")

    


joblib.dump(best_model, "final_model.joblib")
joblib.dump(scaler, "scaler.joblib")

# במקרה של מודל פולינומי – שמור גם את ה-Polynomial Converter
if "Polynomial" in best_model_name:
    joblib.dump(best_model.named_steps['polynomialfeatures'], "poly_converter.joblib")

print("\n💾 Model and preprocessing saved successfully!")


loaded_model = joblib.load("final_model.joblib")
loaded_scaler = joblib.load("scaler.joblib")

if "Polynomial" in best_model_name:
    loaded_poly = joblib.load("poly_converter.joblib")
    print("✅ Polynomial converter loaded too!")

print("\n🚀 Model and preprocessing reloaded successfully and ready for inference.")


```

## 📘 Summary & Discussion
לאחר בדיקת כל ארבעת המודלים נמצא כי:
- ערכי ה־R² נמוכים → קשר חלש בין מדדי בריאות (COVID) ל־GDP.
- המודל היציב ביותר הוא **LassoCV**, שמראה רגולריזציה טובה על נתונים רועשים.
- ניתן לשפר את המודלים על ידי הוספת משתנים חיצוניים (מדיניות ממשלתית, חוב לאומי וכו').

🎯 מסקנה:
המודלים הליניאריים אינם מתאימים לנתוני תקופת הקורונה שבה הקשרים בין המדדים הכלכליים נשברו.
