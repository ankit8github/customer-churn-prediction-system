import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Configure paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "clean_churn_data.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "model_artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
print(f"✓ Loaded data from {DATA_PATH}")
print(f"  Shape: {df.shape}")

# Split data into features and target
X = df.drop("Churn", axis=1)
y = df["Churn"].map({"Yes": 1, "No": 0})

# Define feature categories
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
print(f"\n✓ Feature split:")
print(f"  Categorical ({len(categorical_features)}): {categorical_features}")
print(f"  Numerical ({len(numerical_features)}): {numerical_features}")

# Build Preprocessing Pipeline with imputation
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    remainder="drop"
)
print(f"\n✓ Preprocessing pipeline configured")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✓ Train-test split:")
print(f"  Train: {X_train.shape[0]} samples")
print(f"  Test: {X_test.shape[0]} samples")

# Fit preprocessor on training data only
preprocessor.fit(X_train)
print(f"\n✓ Preprocessor fitted on training data")

# Get feature names from OneHotEncoder for reference
cat_encoder = preprocessor.named_transformers_["cat"]["encoder"]
feature_names_cat = cat_encoder.get_feature_names_out(categorical_features).tolist()
feature_names_all = numerical_features + feature_names_cat
print(f"  Total features after encoding: {len(feature_names_all)}")

# Save preprocessor artifact
preprocessor_path = ARTIFACTS_DIR / "preprocessor.pkl"
joblib.dump(preprocessor, preprocessor_path)
print(f"\n✓ Saved preprocessor to {preprocessor_path}")

# Save feature names for reference
feature_names_path = ARTIFACTS_DIR / "feature_names.pkl"
joblib.dump({
    "numerical": numerical_features,
    "categorical": categorical_features,
    "encoded": feature_names_all
}, feature_names_path)
print(f"✓ Saved feature names to {feature_names_path}")

# Transform data
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)
print(f"\n✓ Data transformed:")
print(f"  X_train shape: {X_train_processed.shape}")
print(f"  X_test shape: {X_test_processed.shape}")

# Save preprocessed data
data_path = ARTIFACTS_DIR / "train_test_data.pkl"
joblib.dump(
    (X_train_processed, X_test_processed, y_train, y_test),
    data_path
)
print(f"✓ Saved train/test data to {data_path}")

# Smoke test: verify pipeline consistency
print(f"\n✓ Smoke Test - Pipeline Consistency:")
sample_input = X_train.iloc[:1]  # Take one sample
sample_output = preprocessor.transform(sample_input)
assert sample_output.shape[1] == X_train_processed.shape[1], "Feature count mismatch!"
print(f"  ✓ Single sample shape: {sample_output.shape} (matches {X_train_processed.shape[1]} features)")
print(f"\n✅ Feature engineering pipeline complete!")
