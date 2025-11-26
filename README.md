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
# ============================================================
# 📌 Regression – Multi Model Training (Correct Version)
# ============================================================

# Features & target
X = df[['Confirmed', 'Deaths', 'Recovered', 'Unemployment', 'CPI']]
y = df['GDP']

# Train/Test Split (NO data leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scaling – fit ONLY on train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ============================================================
# 1️⃣ Linear Regression
# ============================================================
lin_model = LinearRegression()
lin_model.fit(X_train_scaled, y_train)
lin_r2 = lin_model.score(X_test_scaled, y_test)

# ============================================================
# 2️⃣ Ridge Regression (with CV)
# ============================================================
alphas = np.logspace(-3, 3, 50)

ridge = RidgeCV(alphas=alphas, cv=5, scoring='r2')
ridge.fit(X_train_scaled, y_train)
ridge_r2 = ridge.score(X_test_scaled, y_test)

# ============================================================
# 3️⃣ Lasso Regression (with CV)
# ============================================================
lasso = LassoCV(alphas=alphas, cv=5, random_state=42)
lasso.fit(X_train_scaled, y_train)
lasso_r2 = lasso.score(X_test_scaled, y_test)

# ============================================================
# 4️⃣ Polynomial Regression – choose optimal degree using CV
# ============================================================
degrees = [1, 2, 3, 4]
poly_scores = []

for d in degrees:
    poly = make_pipeline(StandardScaler(), PolynomialFeatures(d), LinearRegression())
    cv_score = cross_val_score(poly, X, y, cv=5, scoring='r2').mean()
    poly_scores.append(cv_score)
    print(f"Degree {d} → Mean CV R²: {cv_score:.3f}")

best_degree = degrees[np.argmax(poly_scores)]

# Train final polynomial model using best degree
best_poly = make_pipeline(StandardScaler(), PolynomialFeatures(best_degree), LinearRegression())
best_poly.fit(X_train, y_train)
poly_r2 = best_poly.score(X_test, y_test)

# ============================================================
# 5️⃣ Compare Models
# ============================================================
results = {
    "Linear Regression": lin_r2,
    "Ridge Regression": ridge_r2,
    "Lasso Regression": lasso_r2,
    f"Polynomial (deg={best_degree})": poly_r2
}

print("\n=== R² Scores ===")
for name, score in results.items():
    print(f"{name}: {score:.4f}")

best_model_name = max(results, key=results.get)
print(f"\n🏆 Best Model: {best_model_name}")

# Select final model
if best_model_name == "Linear Regression":
    final_model = lin_model
elif best_model_name == "Ridge Regression":
    final_model = ridge
elif best_model_name == "Lasso Regression":
    final_model = lasso
else:
    final_model = best_poly

# ============================================================
# 6️⃣ Save model & scaler
# ============================================================
joblib.dump(final_model, "final_regression_model.joblib")
joblib.dump(scaler, "regression_scaler.joblib")

print("\n💾 Regression model saved!")

# Reload test
loaded_model = joblib.load("final_regression_model.joblib")
loaded_scaler = joblib.load("regression_scaler.joblib")

print("✅ Model and scaler loaded successfully.")

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



---

<br><br><br>


<div align="center">
 



 <h1> 📌 Customer Churn Prediction – Full Project Code</h1>
✔️ כולל Data Preparation, EDA, Training, Evaluation, Deployment
</div>

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


# 🎓 Project II – COVID-19 Economic Impact & Customer Churn Prediction  
### *Supervised Learning: Regression + Classification*

<p align="center">
  <img src="https://img.icons8.com/color/96/laptop.png" width="120">
</p>

פרויקט זה משלב בין **חיזוי כלכלי** לבין **חיזוי נטישת לקוחות**, תוך שימוש במודלים מתקדמים של למידת מכונה.

---

# 📁 תוכן העניינים
## **1. חלק א׳ – רגרסיה: חיזוי GDP**
## **2. חלק ב׳ – סיווג: חיזוי נטישת לקוחות (Churn)**
## **3. תובנות מרכזיות**
## **4. מסקנה סופית**

---

# 📌 חלק א׳ — רגרסיה: חיזוי GDP

מטרת המשימה:  
לנתח את הקשר בין נתוני COVID-19 ומדדים כלכליים לבין **התוצר המקומי הגולמי (GDP)** של מדינות.

---

# 📊 תובנות מרכזיות — Correlation Analysis  

## **1️⃣ קשר בין מדדי הקורונה ל־GDP**

בשנים **2021–2022**, נמצא כי ה־GDP מציג **מתאם חלש מאוד** עם מדדי הקורונה:

- **Confirmed ↔ GDP ~ 0.24**  
- **Deaths ↔ GDP ~ 0.22**  
- **Recovered ↔ GDP ~ 0.13**

### 🧩 משמעות:
הקורונה **לא מסבירה את תנודות ה־GDP בין מדינות**.  
ה־GDP מושפע מגורמים מהותיים כמו:
- גודל הכלכלה  
- משאבי טבע  
- יצוא ויבוא  
- מדיניות ממשלתית  
- פעילות התעשייה  

לכן המתאם נשאר נמוך מאוד.

---

## **2️⃣ קשרים בין מדדי הקורונה**

נמצא **קשר חזק מאוד** בין מדדי המגפה:

- **Confirmed ↔ Recovered ~ 0.95**  
- **Confirmed ↔ Deaths ~ 0.88–0.89**  
- **Recovered ↔ Deaths ~ 0.79**

היגיון:  
במדינות עם הרבה נדבקים → יהיו גם הרבה מחלימים והרבה נפטרים.

---

## **3️⃣ CPI ואבטלה — מתאם כמעט אפסי**

שני המדדים הכלכליים:
- לא קשורים לקורונה  
- לא קשורים ל־GDP  
- נשלטים בעיקר ממדיניות פנימית, סגרים ומבנה שוק העבודה  

---

# 📌 חלק ב׳ — סיווג: חיזוי נטישת לקוחות (Churn)

במשימה זו נבנו מודלים שמטרתם לחזות האם לקוח ינטוש שירות.

המודלים שנבדקו:
- Logistic Regression  
- KNN  
- SVM  
- Random Forest  

בוצעו:  
✔️ Scaling  
✔️ GridSearchCV  
✔️ Confusion Matrix  
✔️ Precision / Recall / F1  
✔️ בחירת מודל מנצח  
✔️ שמירת המודל ל־Production  

---

# 🏆 המודל המנצח — Random Forest

מודל זה הגיע ל־**F1 הגבוה ביותר**, עם הביצועים היציבים ביותר.

---

# 🧠 מסקנה סופית

הפרויקט מציג תהליך מלא של Data Science:

- ניקוי וטיוב דאטה  
- EDA מעמיק  
- אימון מודלים מרובים  
- השוואת ביצועים  
- בחירת מודל מיטבי  
- שמירה וטעינה של מודלים  

---

# ⭐ סיכום תוצאות

## 🔵 חלק הרגרסיה (GDP)
- COVID לא משפיע ישירות על ה־GDP  
- Ridge Regression סיפק את היציבות הגבוהה ביותר  
- המתאמים חלשים – תוצאה תקינה בהתאם לנתונים  

## 🟢 חלק הסיווג (Churn)
- הנתונים היו חזקים וברורים  
- Random Forest נתן את תוצאות ה־F1 הטובות ביותר  
- המודל מתאים לשימוש עסקי אמיתי  

---

# ✔️ סוף
