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
# 📊 Correlation Analysis — תובנות מרכזיות

## 1. קשר בין מדדי הקורונה (COVID-19) ל-GDP

בשנים 2021–2022 נמצא כי התוצר הלאומי (GDP) מציג **מתאם חלש מאוד** עם כל מדדי הקורונה:

- Confirmed ↔ GDP ≈ 0.24  
- Deaths ↔ GDP ≈ 0.22  
- Recovered ↔ GDP ≈ 0.13  

**משמעות:**  
רמת התחלואה אינה מסבירה את מצב הכלכלה של המדינה.  
מדינות בעלות כלכלות גדולות וקטנות חוו את הקורונה בצורה שונה — ולכן המתאם בממוצע נמוך מאוד.

ה-GDP מושפע בעיקר מגורמים מבניים כגון:
- גודל הכלכלה  
- משאבים טבעיים  
- מדיניות ממשלתית  
- פעילות מסחר וייצור  

לכן מדדי COVID לבדם אינם מצליחים לנבא אותו.

---

## 2. קשרים בין מדדי הקורונה

כאן מופיעים דפוסים חזקים וברורים:

- Confirmed ↔ Recovered ≈ 0.95  
- Confirmed ↔ Deaths ≈ 0.88–0.89  
- Deaths ↔ Recovered ≈ 0.79  

**משמעות:**  
מדינות שבהן מספר הנדבקים גבוה יותר נוטות להציג גם מספר גבוה של מחלימים ומספר גבוה של נפטרים — דפוס טבעי במגפות.

---

## 3. קשר בין Unemployment ו-CPI לשאר המשתנים

### Unemployment (אבטלה)
- קשר חלש עם GDP ≈ −0.09  
- קשר כמעט אפסי עם מדדי הקורונה  

האבטלה בתקופה זו הושפעה יותר מהחלטות ממשלתיות, סגרים ומדיניות תעסוקה — ולא ישירות מהיקף התחלואה.

### CPI (מדד מחירים)
- מתאם כמעט אפסי עם כל שאר המשתנים (≈ −0.04 עד 0.04)  

חלק מהמדינות חסרות ערכי CPI, ובכל מקרה לא נמצא דפוס כללי שמסביר את הקשר בין מדדי הבריאות לאינפלציה בתקופה זו.

---

## 4. השוואה בין השנים 2021 ל-2022

***דפוסי הקורלציה כמעט זהים בשתי השנים:***

- קשרים חזקים בין Confirmed–Deaths–Recovered  
- קשר חלש בין GDP למשתנים אחרים  
- CPI ואבטלה שומרים על מבנה קורלציה חלש ולא עקבי  

**משמעות:**  
נתוני הבריאות של הקורונה לא השפיעו בצורה ישירה על הכלכלה של המדינות גם בשנת 2021 וגם בשנת 2022.

---

## 📌 מסקנה כוללת

- משתני הקורונה מתואמים מאוד זה לזה — דפוס מגפה טבעי.  
- אין קשר מובהק בין COVID-19 ל-GDP.  
- CPI ואבטלה כמעט שאינם מציגים קשרים מובהקים לשאר המשתנים.  

לכן, ההשפעה הכלכלית של הקורונה **אינה מופיעה בצורה מובהקת** במאגר הנתונים הזה, וזה מסביר מדוע מודלי הרגרסיה מציגים R² נמוך — תוצאה תקינה לחלוטין בהתחשב בנתונים.







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

לאחר בדיקת כלל המודלים (Linear, Ridge, Lasso, Polynomial), מתקבל כי ערכי ה־R² נמוכים יחסית.
המשמעות היא שמדדי הבריאות של הקורונה (Confirmed, Deaths, Recovered) וכן מדדי המאקרו הבסיסיים
(CPI, Unemployment) **אינם מסוגלים להסביר בצורה טובה את השונות ב-GDP בין המדינות**.

כל ארבעת המודלים מציגים ביצועים דומים מאוד, כאשר:
- **Linear Regression** מציג את ערך ה־R² הגבוה ביותר (גם אם בפער קטן מאוד).  
- הוספת רגולריזציה (Ridge/Lasso) לא שיפרה את הביצועים באופן משמעותי.
- Polynomial Regression בדרגות גבוהות קרס לחלוטין (overfitting) — ורק דרגה 1 עבדה, למעשה כמו מודל ליניארי.

### 🎯 מסקנה:
הנתונים מראים כי **אין קשר מובהק בין מדדי COVID-19 לבין רמת התוצר (GDP)**.
לכן מודלים ליניאריים או פולינומיים אינם מסוגלים לחזות את ה-GDP בצורה טובה בעזרת משתנים אלו בלבד.

כדי לשפר את הביצועים יש צורך להוסיף משתנים חיצוניים כגון:
- גודל אוכלוסייה  
- היקף מסחר בינלאומי  
- הוצאה ממשלתית  
- שיעורי צמיחה קודמים  
- חוב לאומי  
- מדדי פיתוח (HDI)  

מודלים מבוססי מגמות מאקרו־כלכליות יתאימו הרבה יותר למשימה.
