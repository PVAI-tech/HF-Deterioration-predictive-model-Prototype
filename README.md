# HF-Deterioration-predictive-model-Prototype
Heart Failure deterioration within 24-28 hours ML model using NEWS2 Demo/prototype
pip install xgboost scikit-learn matplotlib pandas
import pandas as pd
df = pd.read_csv("synthetic_hf_deterioration_dataset.csv")
X = df.drop(columns=["deterioration_within_24h", "patient_id"])
y = df["deterioration_within_24h"]
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
df = pd.read_csv("synthetic_hf_deterioration_dataset.csv")
df.head()
df.describe()
df['deterioration_within_24h'].value_counts(normalize=True)
target_col = "deterioration_within_24h"
drop_cols = ["patient_id", target_col]

X = df.drop(columns=drop_cols)
y = df[target_col]
X = pd.get_dummies(X, columns=["sex", "consciousness_avpu"], drop_first=True)
news2 = df["news2_total"].values
y_true = y.values

# Example: treat NEWS2 >= 5 as 'high risk'
threshold = 5
news2_pred = (news2 >= threshold).astype(int)

print("Confusion matrix (NEWS2 >= 5):")
print(confusion_matrix(y_true, news2_pred))
print(classification_report(y_true, news2_pred))

# For "ROC" you can treat normalized NEWS2 as a score
news2_score_norm = np.clip(news2 / 10.0, 0, 1)
print("NEWS2 AUC:", roc_auc_score(y_true, news2_score_norm))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

print("ML AUC:", roc_auc_score(y_test, y_prob))
print("Confusion matrix (ML, 0.5 threshold):")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – HF 24h Deterioration")
plt.show()
news2_test = df.loc[X_test.index, "news2_total"].values
news2_norm = np.clip(news2_test / 10.0, 0, 1)

hybrid_raw = 0.5 * news2_norm + 0.5 * y_prob
hybrid_score = hybrid_raw * 100  # 0–100 scale
print("Hybrid AUC:", roc_auc_score(y_test, hybrid_raw))

# Optional: define high-risk band (e.g., hybrid >= 70)
hybrid_high_risk = (hybrid_score >= 70).astype(int)
print("Confusion matrix (Hybrid high-risk >=70):")
print(confusion_matrix(y_test, hybrid_high_risk))
importances = model.feature_importances_
feat_names = X.columns

imp_df = pd.DataFrame({"feature": feat_names, "importance": importances})
imp_df = imp_df.sort_values("importance", ascending=False)

imp_df.head(15)
top_n = 15
plt.figure(figsize=(8,6))
plt.barh(imp_df["feature"].head(top_n)[::-1], imp_df["importance"].head(top_n)[::-1])
plt.title("Top Feature Importances – HF Deterioration Model")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
import joblib

joblib.dump(model, "hf_deterioration_xgb_model.joblib")
imp_df.to_csv("hf_feature_importances.csv", index=False)
import joblib

joblib.dump(model, "hf_deterioration_xgb_model.joblib")
imp_df.to_csv("hf_feature_importances.csv", index=False)
example = X_test.iloc[[0]]
example_prob = model.predict_proba(example)[:,1][0]
example_news2 = df.loc[example.index, "news2_total"].values[0]
example_hybrid = 0.5 * min(example_news2/10, 1.0) + 0.5 * example_prob

print("Example NEWS2:", example_news2)
print("Example ML prob:", example_prob)
print("Example Hybrid score:", example_hybrid*100)
