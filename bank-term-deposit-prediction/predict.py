import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load Dataset
df = pd.read_csv("bank-full.csv", sep=';')

# Create binary target variable
df['y_bin'] = (df['y'] == 'yes').astype(int)

# Preprocess Features
X = df.drop(['y', 'y_bin'], axis=1)
X = pd.get_dummies(X, drop_first=True)
y = df['y_bin']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)
y_pred_dt = dt_clf.predict(X_test)

# Train Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

# Train XGBoost
xgb_clf = XGBClassifier(n_estimators=100, random_state=42)
xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_test)


# Evaluate Models
def evaluate_model(y_test, y_pred, model_name):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"=== {model_name} ===")
    print(f"Accuracy: {acc:.4f}")
    return acc, prec, rec, f1


# Store metrics
metrics = {}

metrics['Decision Tree'] = evaluate_model(y_test, y_pred_dt, "Decision Tree")
metrics['Random Forest'] = evaluate_model(y_test, y_pred_rf, "Random Forest")
metrics['XGBoost'] = evaluate_model(y_test, y_pred_xgb, "XGBoost")

# Plotting Performance Comparison
metrics_df = pd.DataFrame(metrics, index=['Accuracy', 'Precision', 'Recall', 'F1-score']).T

# Transpose so that metrics are on the X-axis
metrics_df.T.plot(kind='bar', figsize=(12, 8))

plt.title('Model Comparison: Accuracy, Precision, Recall, F1-score')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.xlabel('Performance Metric')
plt.legend(title='Model', loc='lower right')
plt.grid(axis='y')
plt.show()