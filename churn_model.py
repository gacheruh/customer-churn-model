import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_CSV_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DEFAULT_MODEL_DIR = "models"
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "churn_pipeline.joblib")


def load_data(csv_path: str = DEFAULT_CSV_PATH) -> pd.DataFrame:
	dataframe = pd.read_csv(csv_path)
	# Coerce TotalCharges to numeric because the dataset may have blanks
	if "TotalCharges" in dataframe.columns:
		dataframe["TotalCharges"] = pd.to_numeric(dataframe["TotalCharges"], errors="coerce")
	return dataframe


def split_features_target(dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
	if "Churn" not in dataframe.columns:
		raise ValueError("Expected 'Churn' column in dataset")
	features = dataframe.copy()
	if "customerID" in features.columns:
		features = features.drop(columns=["customerID"])  # ID is not predictive
	target = (features.pop("Churn").astype(str).str.strip().str.title() == "Yes").astype(int)
	return features, target


def infer_feature_types(features: pd.DataFrame) -> Tuple[List[str], List[str]]:
	categorical_columns: List[str] = []
	numeric_columns: List[str] = []
	for column_name in features.columns:
		if pd.api.types.is_numeric_dtype(features[column_name]):
			numeric_columns.append(column_name)
		else:
			categorical_columns.append(column_name)
	return categorical_columns, numeric_columns


def build_preprocessor(categorical_columns: List[str], numeric_columns: List[str]) -> ColumnTransformer:
	categorical_transformer = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="most_frequent")),
			("onehot", OneHotEncoder(handle_unknown="ignore", drop=None, sparse_output=False)),
		]
	)
	numeric_transformer = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="median")),
			("scaler", StandardScaler(with_mean=True, with_std=True)),
		]
	)
	preprocessor = ColumnTransformer(
		transformers=[
			("categorical", categorical_transformer, categorical_columns),
			("numeric", numeric_transformer, numeric_columns),
		]
	)
	return preprocessor


def build_pipeline(features: pd.DataFrame) -> Pipeline:
	categorical_columns, numeric_columns = infer_feature_types(features)
	preprocessor = build_preprocessor(categorical_columns, numeric_columns)
	model = LogisticRegression(max_iter=200, n_jobs=None, solver="liblinear")
	pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
	return pipeline


def train_and_save(csv_path: str = DEFAULT_CSV_PATH, model_path: str = DEFAULT_MODEL_PATH) -> Dict[str, float]:
	dataframe = load_data(csv_path)
	features, target = split_features_target(dataframe)
	pipeline = build_pipeline(features)
	pipeline.fit(features, target)
	# Simple holdout-like AUC using in-sample for quick feedback (not true validation)
	try:
		probabilities = pipeline.predict_proba(features)[:, 1]
		auc = roc_auc_score(target, probabilities)
	except Exception:
		auc = float("nan")
	os.makedirs(os.path.dirname(model_path), exist_ok=True)
	joblib.dump({"pipeline": pipeline, "feature_columns": list(features.columns)}, model_path)
	return {"auc_in_sample": float(auc)}


def load_model(model_path: str = DEFAULT_MODEL_PATH) -> Dict:
	if not os.path.exists(model_path):
		# Train lazily if missing
		train_and_save(DEFAULT_CSV_PATH, model_path)
	return joblib.load(model_path)


def predict_single(feature_dict: Dict, model_path: str = DEFAULT_MODEL_PATH) -> Dict:
	bundle = load_model(model_path)
	pipeline: Pipeline = bundle["pipeline"]
	feature_columns: List[str] = bundle["feature_columns"]
	input_frame = pd.DataFrame([{k: feature_dict.get(k, np.nan) for k in feature_columns}])
	probability = float(pipeline.predict_proba(input_frame)[0, 1])
	label = int(probability >= 0.5)
	return {"churn_probability": probability, "churn_label": label}


def predict_batch(input_dataframe: pd.DataFrame, model_path: str = DEFAULT_MODEL_PATH) -> pd.DataFrame:
	bundle = load_model(model_path)
	pipeline: Pipeline = bundle["pipeline"]
	# Ensure TotalCharges numeric
	if "TotalCharges" in input_dataframe.columns:
		input_dataframe = input_dataframe.copy()
		input_dataframe["TotalCharges"] = pd.to_numeric(input_dataframe["TotalCharges"], errors="coerce")
	probabilities = pipeline.predict_proba(input_dataframe)[:, 1]
	predictions = (probabilities >= 0.5).astype(int)
	result = input_dataframe.copy()
	result["churn_probability"] = probabilities
	result["churn_label"] = predictions
	return result
