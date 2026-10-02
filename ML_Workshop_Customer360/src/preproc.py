# Part B - Preprocessing 
from commen import *

# Target and features
y_c = df["Churn"].map({"No": 0, "Yes": 1})
y_r= df["Future12MRevenueEGP"]
# drop the ID and both targets (Future12MRevenueEGP would be leakage)
X = df.drop(columns=["CustomerID", "Churn", "Future12MRevenueEGP"])

# Split columns by type
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
print("Numeric:", num_cols)
print("Categorical:", cat_cols)

# Split first, so the test set is not used when fitting the preprocessing
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_c, test_size=0.2, random_state=42, stratify=y_c
)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_r, test_size=0.2, random_state=42
)
# Numeric: median imputation + scaling
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# Categorical: most frequent imputation + one-hot encoding
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(
    [
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols),
    ],
    sparse_threshold=0,  # return a normal (dense) array
)

# fit on train only, then transform both
x_train_pred_c = preprocessor.fit_transform(X_train_c)
x_test_pred_c = preprocessor.transform(X_test_c)
x_train_pred_r = preprocessor.transform(X_train_r)
x_test_pred_r = preprocessor.transform(X_test_r)

print("Train shape:", x_train_pred_c.shape)
print("Test shape:", x_test_pred_c.shape)
print("Missing after preprocessing:", pd.DataFrame(x_train_pred_c).isnull().sum().sum())