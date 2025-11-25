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
# כל הנתונים בעמודה הזאת שהיא המחוז של כל מדינה הם "0" או מילים לא רלונטיות , אז לדטא של המודל אין צורך בה   
df = df.drop(columns=["Province/State", "Unnamed: 0"], errors='ignore')
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





```python
# 📌 Customer Churn Prediction – Full Project Code
## ✔️ כולל Data Preparation, EDA, Training, Evaluation, Deployment

```python
# =========================================================
# 📦 Imports – כל הייבוא מרוכז בתחילת הקוד
# =========================================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    confusion_matrix
)
import joblib

# =========================================================
# 📌 Data Preparation
# =========================================================

# טעינת הנתונים
path = "/content/drive/MyDrive/Classroom/עותק של customer_churn_dataset.csv"
df = pd.read_csv(path)

df.head()
df.columns

# עמודת CustomerID היא מזהה בלבד → לא נותנת מידע על התנהגות → מסירים
df = df.drop(columns=["CustomerID"])
df.columns

# בדיקת ערכים חסרים
print("Missing Values:")
print(df.isnull().sum())

# בדיקת כפילויות
print("Duplicate Rows:", df.duplicated().sum())

# קידוד משתנים קטגוריאליים
df = pd.get_dummies(df, drop_first=True)
df.head()


# =========================================================
# 📊 Data Exploration
# =========================================================

# חישוב קורלציה
corr = df.corr()
corr

# Heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Pairplot
sns.pairplot(df, height=2)
plt.show()


# =========================================================
# 📌 Model Training – Scaling + Splitting
# =========================================================

# X & y
X = df.drop("Churn", axis=1)
y = df["Churn"]

# חלוקה ל־Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# נרמול (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled[:5]


# =========================================================
# 🤖 Logistic Regression – Grid Search
# =========================================================

log_params = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs']
}

log_model = LogisticRegression(max_iter=500)
log_grid = GridSearchCV(log_model, log_params, cv=5, scoring='f1')
log_grid.fit(X_train_scaled, y_train)

print("Optimal Logistic Regression Parameters:")
print(log_grid.best_params_)

log_best = log_grid.best_estimator_


# =========================================================
# 🤖 KNN – Grid Search
# =========================================================

knn_params = {
    'n_neighbors': list(range(1, 21)),
    'weights': ['uniform', 'distance']
}

knn = KNeighborsClassifier()
knn_grid = GridSearchCV(knn, knn_params, cv=5, scoring='f1')
knn_grid.fit(X_train_scaled, y_train)

print("Optimal KNN Parameters:")
print(knn_grid.best_params_)

knn_best = knn_grid.best_estimator_


# =========================================================
# 🤖 SVM – Grid Search
# =========================================================

svm_params = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto']
}

svm = SVC(probability=True)
svm_grid = GridSearchCV(svm, svm_params, cv=5, scoring='f1')
svm_grid.fit(X_train_scaled, y_train)

print("Optimal SVM Parameters:")
print(svm_grid.best_params_)

svm_best = svm_grid.best_estimator_


# =========================================================
# 🌲 Random Forest – Grid Search
# =========================================================

rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'bootstrap': [True, False]
}

rf = RandomForestClassifier()
rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='f1')
rf_grid.fit(X_train, y_train)

print("Optimal Random Forest Parameters:")
print(rf_grid.best_params_)

rf_best = rf_grid.best_estimator_


# =========================================================
# 📌 Predictions (לפי דרישות הפרויקט)
# =========================================================

y_pred_log = log_best.predict(X_test_scaled)
y_proba_log = log_best.predict_proba(X_test_scaled)[:, 1]

y_pred_knn = knn_best.predict(X_test_scaled)
y_proba_knn = knn_best.predict_proba(X_test_scaled)[:, 1]

y_pred_svm = svm_best.predict(X_test_scaled)
y_proba_svm = svm_best.predict_proba(X_test_scaled)[:, 1]

y_pred_rf = rf_best.predict(X_test)
y_proba_rf = rf_best.predict_proba(X_test)[:, 1]


# =========================================================
# 📌 Model Evaluation – Accuracy, Recall, F1
# =========================================================

results = {"Model": [], "Accuracy": [], "Recall": [], "F1 Score": []}

def evaluate_model(name, y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results["Model"].append(name)
    results["Accuracy"].append(acc)
    results["Recall"].append(rec)
    results["F1 Score"].append(f1)

    print(f"\n{name}:")
    print("Accuracy:", acc)
    print("Recall:", rec)
    print("F1 Score:", f1)


evaluate_model("Logistic Regression", y_test, y_pred_log)
evaluate_model("KNN", y_test, y_pred_knn)
evaluate_model("SVM", y_test, y_pred_svm)
evaluate_model("Random Forest", y_test, y_pred_rf)


# =========================================================
# 📌 Confusion Matrices
# =========================================================

def plot_confusion(model_name, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.title(f"Confusion Matrix – {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

plot_confusion("Logistic Regression", y_test, y_pred_log)
plot_confusion("KNN", y_test, y_pred_knn)
plot_confusion("SVM", y_test, y_pred_svm)
plot_confusion("Random Forest", y_test, y_pred_rf)


# =========================================================
# 📌 Model Comparison Table
# =========================================================

results_df = pd.DataFrame(results)
results_df


# =========================================================
# 🏆 Selecting the Best Model
# =========================================================

best_model_name = results_df.iloc[results_df["F1 Score"].idxmax()]["Model"]
print("Best Model:", best_model_name)


# =========================================================
# 🎯 Training Best Model on Full Dataset
# =========================================================

if best_model_name == "Logistic Regression":
    final_model = log_best
    X_all_scaled = scaler.fit_transform(X)
    final_model.fit(X_all_scaled, y)

elif best_model_name == "KNN":
    final_model = knn_best
    X_all_scaled = scaler.fit_transform(X)
    final_model.fit(X_all_scaled, y)

elif best_model_name == "SVM":
    final_model = svm_best
    X_all_scaled = scaler.fit_transform(X)
    final_model.fit(X_all_scaled, y)

elif best_model_name == "Random Forest":
    final_model = rf_best
    final_model.fit(X, y)  # RF לא חייב Scaling


# =========================================================
# 💾 Export Model + Scaler
# =========================================================

joblib.dump(final_model, "final_model.joblib")
joblib.dump(scaler, "scaler.joblib")

print("Model and scaler saved successfully!")


# =========================================================
# 🔄 Loading Saved Model Back
# =========================================================

loaded_model = joblib.load("final_model.joblib")
loaded_scaler = joblib.load("scaler.joblib")

print("Model and scaler loaded successfully!")
```
