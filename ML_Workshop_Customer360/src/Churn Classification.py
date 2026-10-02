# Part C - Churn Classification Task
from commen import *
from preproc import x_train_pred_c, x_test_pred_c, y_train_c, y_test_c, preprocessor, num_cols, cat_cols
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# 1. Baseline Models Evaluation
models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
    "Random Forest Baseline": RandomForestClassifier(class_weight='balanced', random_state=42)
}

plt.figure(figsize=(8, 6))

for name, model in models.items():
    model.fit(x_train_pred_c, y_train_c)
    y_pred = model.predict(x_test_pred_c)
    y_proba = model.predict_proba(x_test_pred_c)[:, 1]
    
    auc = roc_auc_score(y_test_c, y_proba)
    print(f"=== {name} ===")
    print(classification_report(y_test_c, y_pred))
    print(f"ROC-AUC: {auc:.4f}\n")
    
    fpr, tpr, _ = roc_curve(y_test_c, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

# 2. Hyperparameter Tuning for Random Forest Classifier
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'class_weight': ['balanced', None]
}

grid_search = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    param_grid,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1
)

grid_search.fit(x_train_pred_c, y_train_c)
best_clf = grid_search.best_estimator_

y_pred_best = best_clf.predict(x_test_pred_c)
y_proba_best = best_clf.predict_proba(x_test_pred_c)[:, 1]
best_auc = roc_auc_score(y_test_c, y_proba_best)

print("=== Best Tuned Random Forest Classifier ===")
print("Best Params:", grid_search.best_params_)
print(classification_report(y_test_c, y_pred_best))
print(f"Tuned ROC-AUC: {best_auc:.4f}\n")

fpr_best, tpr_best, _ = roc_curve(y_test_c, y_proba_best)
plt.plot(fpr_best, tpr_best, label=f"Tuned RF (AUC = {best_auc:.3f})", linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label="Random Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Churn Classification - ROC Curves")
plt.legend()
plt.show()

# 3. Feature Importance Analysis
ohe_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_cols)
all_feature_names = num_cols + list(ohe_feature_names)

importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': best_clf.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("=== Top Churn Predictors ===")
print(importance_df.head(10))

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'][:15][::-1], importance_df['Importance'][:15][::-1])
plt.xlabel("Importance Score")
plt.title("Top 15 Features Driving Churn")
plt.tight_layout()
plt.show()