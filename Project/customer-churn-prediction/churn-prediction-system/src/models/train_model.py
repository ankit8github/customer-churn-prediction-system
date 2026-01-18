"""
Train machine learning models for churn prediction
Supports: LogisticRegression, RandomForest, XGBoost
Saves best model to model_artifacts/
"""
import joblib
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    f1_score, precision_score, recall_score
)
import warnings
warnings.filterwarnings('ignore')

# Set paths - relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "model_artifacts"

# Load preprocessed data
data_path = ARTIFACTS_DIR / "train_test_data.pkl"
X_train, X_test, y_train, y_test = joblib.load(data_path)
print(f"✓ Loaded preprocessed data")
print(f"  X_train: {X_train.shape}")
print(f"  X_test: {X_test.shape}")

# Define models to train
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
}

print(f"\n✓ Training models...")
results = {}

for model_name, model in models.items():
    print(f"\n  Training {model_name}...", end=" ")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results[model_name] = {
        "model": model,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba
    }
    
    print(f"F1={f1:.3f}, Prec={precision:.3f}, Rec={recall:.3f}, AUC={roc_auc:.3f}")

# Select best model (by F1 score)
best_model_name = max(results, key=lambda x: results[x]["f1"])
best_model = results[best_model_name]["model"]
best_metrics = results[best_model_name]

print(f"\n✓ Best Model: {best_model_name}")
print(f"  F1 Score: {best_metrics['f1']:.4f}")
print(f"  Precision: {best_metrics['precision']:.4f}")
print(f"  Recall: {best_metrics['recall']:.4f}")
print(f"  ROC-AUC: {best_metrics['roc_auc']:.4f}")

# Detailed classification report
print(f"\n✓ Classification Report ({best_model_name}):")
print(classification_report(y_test, best_metrics["y_pred"], 
                          target_names=["No Churn", "Churn"]))

# Confusion matrix
cm = confusion_matrix(y_test, best_metrics["y_pred"])
print(f"✓ Confusion Matrix:")
print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
print(f"  FN={cm[1,0]}, TP={cm[1,1]}")

# Save best model
model_path = ARTIFACTS_DIR / "churn_model.pkl"
joblib.dump(best_model, model_path)
print(f"\n✓ Saved best model to {model_path}")

# Save model metadata
metadata = {
    "model_name": best_model_name,
    "train_size": len(X_train),
    "test_size": len(X_test),
    "num_features": X_train.shape[1],
    "metrics": {
        "f1_score": float(best_metrics["f1"]),
        "precision": float(best_metrics["precision"]),
        "recall": float(best_metrics["recall"]),
        "roc_auc": float(best_metrics["roc_auc"])
    }
}
metadata_path = ARTIFACTS_DIR / "model_metadata.pkl"
joblib.dump(metadata, metadata_path)
print(f"✓ Saved model metadata to {metadata_path}")

# Create comparison report
comparison_df = pd.DataFrame({
    "Model": list(results.keys()),
    "F1 Score": [results[m]["f1"] for m in results.keys()],
    "Precision": [results[m]["precision"] for m in results.keys()],
    "Recall": [results[m]["recall"] for m in results.keys()],
    "ROC-AUC": [results[m]["roc_auc"] for m in results.keys()]
})

comparison_path = ARTIFACTS_DIR / "model_comparison.csv"
comparison_df.to_csv(comparison_path, index=False)
print(f"✓ Saved model comparison to {comparison_path}")

print(f"\n✓ Model Comparison:")
print(comparison_df.to_string(index=False))

print(f"\n✅ Training complete! Model ready for deployment.")