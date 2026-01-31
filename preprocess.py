import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load raw data
df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Drop customerID (not useful)
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Drop missing values
df.dropna(inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# Save processed data
df.to_csv("data/processed/telco_processed.csv", index=False)

print("Preprocessing completed successfully")
print("Final dataset shape:", df.shape)
