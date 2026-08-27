# train_model.py

import pandas as pd
import numpy as np
import json
import joblib

from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. LOAD DEVELOPMENT DATASET
print("Loading development dataset...")
data = pd.read_csv("dataset/ml_development_dataset.csv")

print(f"Dataset loaded. Total samples: {len(data)}")
print(f"Public measured samples: {len(data[data['data_source'] == 'public_measured'])}")
print(f"Synthetic samples: {len(data[data['data_source'] == 'synthetic'])}")

features = ["Ra226", "Th232", "K40", "U235", "SoilPH", "SoilTexture", "MaterialType"]
X = data[features]

# 2. PIPELINE SETUP
numerical_cols = ["Ra226", "Th232", "K40", "U235", "SoilPH"]
categorical_cols = ["SoilTexture", "MaterialType"]

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, numerical_cols),
    ('cat', cat_transformer, categorical_cols)
])

# 3. TRAIN ANOMALY DETECTION MODEL (ISOLATION FOREST)
anomaly_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('detector', IsolationForest(
        n_estimators=150,
        contamination=0.05,
        random_state=42
    ))
])

print("\nTraining Isolation Forest Anomaly Detector...")
anomaly_pipe.fit(X)

# Predict outliers on the training set (-1 for outlier, 1 for inlier)
train_preds = anomaly_pipe.predict(X)
# Calculate decision scores
train_scores = anomaly_pipe.score_samples(X)

# Save the anomaly detection model
joblib.dump(anomaly_pipe, "model/anomaly_model.pkl")
# Also save as best_model.pkl for compatibility safeguards
joblib.dump(anomaly_pipe, "model/best_model.pkl")

# 4. COMPUTE RANDOM FOREST FEATURE IMPORTANCE ON DOSE RATE (FOR DASHBOARD CHART COMPATIBILITY)
print("Training Random Forest Regressor for feature importance mapping...")
rf_imp_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42
    ))
])
rf_imp_pipe.fit(X, data["DoseRate"])

rf_regressor = rf_imp_pipe.named_steps['regressor']
cat_encoder = rf_imp_pipe.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
encoded_cats = list(cat_encoder.get_feature_names_out(categorical_cols))
all_features_encoded = numerical_cols + encoded_cats

importances = rf_regressor.feature_importances_
radionuclide_mapping = {
    "Ra-226": "Ra226",
    "Th-232": "Th232",
    "K-40": "K40",
    "U-235": "U235"
}
raw_importances = {}
for rad_name, col_name in radionuclide_mapping.items():
    idx = all_features_encoded.index(col_name)
    raw_importances[rad_name] = float(importances[idx])

total_imp = sum(raw_importances.values())
if total_imp > 0:
    for k in raw_importances:
        raw_importances[k] = raw_importances[k] / total_imp

feat_importances_list = sorted(raw_importances.items(), key=lambda x: x[1], reverse=True)

# 5. SAVE METADATA & METRICS
metadata = {
    "task": "anomaly_detection",
    "model_name": "Isolation Forest",
    "features": features,
    "feature_importances": feat_importances_list
}
with open("model/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

metrics_data = {
    "task": "anomaly_detection",
    "target": "None (Unsupervised)",
    "model": "Isolation Forest",
    "features": features,
    "CV folds": None,
    "CV MAE": None,
    "CV MAE std": None,
    "CV RMSE": None,
    "CV RMSE std": None,
    "CV R²": None,
    "CV R² std": None,
    "training sample count": len(data),
    "public measured sample count": len(data[data['data_source'] == 'public_measured']),
    "synthetic sample count": len(data[data['data_source'] == 'synthetic']),
    "independent validation sample count": 20,
    "contamination": 0.05,
    "n_estimators": 150,
    "train_scores": train_scores.tolist()
}

with open("model/evaluation_metrics.json", "w") as f:
    json.dump(metrics_data, f, indent=4)

print("\nAnomaly model and metrics saved successfully!")