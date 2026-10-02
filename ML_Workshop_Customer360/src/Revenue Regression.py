# Part D - Revenue Regression Task (Future12MRevenueEGP)
from commen import *
from preproc import x_train_pred_r, x_test_pred_r, y_train_r, y_test_r, preprocessor, num_cols, cat_cols
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Target and features setup
y_reg = df["Future12MRevenueEGP"]
X_reg = df.drop(columns=["CustomerID", "Churn", "Future12MRevenueEGP"])

# 2. Train-Test Split (80/20)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# 3. Transform features using existing preprocessing pipeline logic
X_train_reg_prep = preprocessor.fit_transform(X_train_reg)
X_test_reg_prep = preprocessor.transform(X_test_reg)

# 4. Linear Regression Baseline
lr = LinearRegression()
lr.fit(X_train_reg_prep, y_train_reg)
y_pred_lr = lr.predict(X_test_reg_prep)

# 5. Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_reg_prep, y_train_reg)
y_pred_rf = rf_reg.predict(X_test_reg_prep)

# 6. Evaluation Metrics (RMSE & R2)
def print_metrics(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"=== {name} ===")
    print(f"RMSE: {rmse:.2f}")
    print(f"R^2 Score: {r2:.4f}\n")

print_metrics("Linear Regression", y_test_reg, y_pred_lr)
print_metrics("Random Forest Regressor", y_test_reg, y_pred_rf)