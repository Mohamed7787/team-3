from commen import *
df=pd.read_csv("C:\\Users\\PC\\Downloads\\ML_Workshop_Customer360\\data\\customer_360_ml_workshop.csv")
# Part A - EDA (simple version)





# 1. Basic info
print("Shape:", df.shape)
print(df.head())
print(df.info())

# 2. Missing values
print("\nMissing values:")
print(df.isnull().sum())

# 3. Descriptive statistics
print("\nStatistics:")
print(df.describe())

# 4. Churn distribution
print("\nChurn counts:")
print(df["Churn"].value_counts())
print(df["Churn"].value_counts(normalize=True) * 100)

df["Churn"].value_counts().plot(kind="bar")
plt.title("Churn distribution")
plt.show()

# 5. Numerical distributions
df.hist(figsize=(12, 8))
plt.tight_layout()
plt.show()

# 6. Churn by ContractType and AutoPay
print("\nChurn rate by ContractType:")
print(pd.crosstab(df["ContractType"], df["Churn"], normalize="index") * 100)

print("\nChurn rate by AutoPay:")
print(pd.crosstab(df["AutoPay"], df["Churn"], normalize="index") * 100)

pd.crosstab(df["ContractType"], df["Churn"], normalize="index").plot(kind="bar")
plt.title("Churn by ContractType")
plt.show()

pd.crosstab(df["AutoPay"], df["Churn"], normalize="index").plot(kind="bar")
plt.title("Churn by AutoPay")
plt.show()

