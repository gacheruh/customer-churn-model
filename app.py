import os
from typing import Dict

import pandas as pd
import streamlit as st

from churn_model import (
	DEFAULT_CSV_PATH,
	DEFAULT_MODEL_PATH,
	load_data,
	predict_batch,
	predict_single,
	train_and_save,
)

st.set_page_config(page_title="Telco Customer Churn", page_icon="📉", layout="wide")


st.markdown(
	"""
	<style>
		.main .block-container{padding-top:2rem;}
		.badge{display:inline-block;padding:4px 10px;border-radius:999px;font-weight:600;font-size:12px}
		.badge-green{background:#e8f7ef;color:#0a6b3a;border:1px solid #bfe7cf}
		.badge-red{background:#ffecec;color:#a00000;border:1px solid #ffb3b3}
		.card{border:1px solid #eaeaea;border-radius:12px;padding:16px;background:#ffffff}
		.small{color:#6b7280;font-size:12px}
	</style>
	""",
	unsafe_allow_html=True,
)

col_a, col_b = st.columns([0.7, 0.3])
with col_a:
	st.markdown("### Customer Churn Predictor")
	st.markdown(
		"Predict the likelihood a customer will churn. Use the form for a single prediction or upload a CSV for batch scoring."
	)
with col_b:
	st.markdown('<span class="badge badge-green">Logistic Regression • One-Hot + Scaling</span>', unsafe_allow_html=True)

st.divider()

# sidebar
with st.sidebar:
	st.header("Model Controls")
	csv_path = st.text_input("Training CSV path", value=DEFAULT_CSV_PATH, help="Path to Telco dataset")
	model_path = st.text_input("Model path", value=DEFAULT_MODEL_PATH)
	if st.button("Train / Retrain Model", type="primary"):
		with st.spinner("Training model..."):
			metrics = train_and_save(csv_path, model_path)
			auc_val = metrics.get("auc_in_sample")
			if auc_val == auc_val:  # not NaN
				st.success(f"Training complete • AUC (in-sample): {auc_val:.3f}")
			else:
				st.success("Training complete")

	st.markdown("---")
	st.subheader("About")
	st.caption(
		"This app uses a scikit-learn pipeline with preprocessing and Logistic Regression. "
		"It will auto-train if the model is missing."
	)

# tabs
predict_tab, batch_tab = st.tabs([" Single Prediction", " Batch Prediction"])

def show_probability(prob: float, threshold: float) -> None:
	pct = max(0.0, min(1.0, float(prob)))
	st.metric("Churn probability", f"{pct*100:.1f}%")
	st.progress(pct)
	label = int(pct >= threshold)
	badge = (
		"<span class='badge badge-red'>Predicted: Churn</span>"
		if label == 1
		else "<span class='badge badge-green'>Predicted: Stay</span>"
	)
	st.markdown(badge, unsafe_allow_html=True)

# single prediction
with predict_tab:
	st.subheader("Enter Customer Details")
	with st.container(border=True):
		threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01, help="Probability above which a customer is labeled as churn.\n Lower threshold → more customers flagged as churn (higher recall, more false positives).\n Higher threshold → fewer flagged (higher precision, more false negatives).")
		col_left, col_right = st.columns(2)
		with col_left:
			gender = st.selectbox("gender", ["Female", "Male"], index=0)
			senior = st.selectbox("SeniorCitizen", ["0", "1"], index=0, help="0 for No, 1 for Yes")
			partner = st.selectbox("Partner", ["No", "Yes"], index=0)
			dependents = st.selectbox("Dependents", ["No", "Yes"], index=0)
			phone = st.selectbox("PhoneService", ["No", "Yes"], index=1)
			multi = st.selectbox("MultipleLines", ["No", "Yes", "No phone service"], index=0)
			internet = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"], index=1)
			security = st.selectbox("OnlineSecurity", ["No", "Yes", "No internet service"], index=0)
			backup = st.selectbox("OnlineBackup", ["No", "Yes", "No internet service"], index=0)
			device = st.selectbox("DeviceProtection", ["No", "Yes", "No internet service"], index=0)
		with col_right:
			tech = st.selectbox("TechSupport", ["No", "Yes", "No internet service"], index=0)
			stv = st.selectbox("StreamingTV", ["No", "Yes", "No internet service"], index=0)
			smov = st.selectbox("StreamingMovies", ["No", "Yes", "No internet service"], index=0)
			contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=0)
			paperless = st.selectbox("PaperlessBilling", ["No", "Yes"], index=1)
			payment = st.selectbox(
				"PaymentMethod",
				[
					"Electronic check",
					"Mailed check",
					"Bank transfer (automatic)",
					"Credit card (automatic)",
				],
				index=0,
			)
			tenure = st.number_input("tenure (months)", min_value=0, max_value=1000, value=1, step=1)
			monthly = st.number_input("MonthlyCharges", min_value=0.0, max_value=10000.0, value=70.0, step=1.0)
			total = st.number_input("TotalCharges", min_value=0.0, max_value=1000000.0, value=70.0, step=1.0)

		predict_btn = st.button("Predict", type="primary")
		if predict_btn:
			input_row = {
				"gender": gender,
				"SeniorCitizen": senior,
				"Partner": partner,
				"Dependents": dependents,
				"PhoneService": phone,
				"MultipleLines": multi,
				"InternetService": internet,
				"OnlineSecurity": security,
				"OnlineBackup": backup,
				"DeviceProtection": device,
				"TechSupport": tech,
				"StreamingTV": stv,
				"StreamingMovies": smov,
				"Contract": contract,
				"PaperlessBilling": paperless,
				"PaymentMethod": payment,
				"tenure": float(tenure),
				"MonthlyCharges": float(monthly),
				"TotalCharges": float(total),
			}
			with st.spinner("Scoring..."):
				result = predict_single(input_row, model_path)
				show_probability(result["churn_probability"], threshold)

# batch prediction
with batch_tab:
	st.subheader("Upload CSV for Batch Scoring")
	with st.container(border=True):
		threshold_b = st.slider("Decision threshold (batch)", 0.05, 0.95, 0.50, 0.01, key="thr_batch")
		uploaded = st.file_uploader("CSV file", type=["csv"], accept_multiple_files=False)
		if uploaded is not None:
			try:
				input_df = pd.read_csv(uploaded)
				if "TotalCharges" in input_df.columns:
					input_df["TotalCharges"] = pd.to_numeric(input_df["TotalCharges"], errors="coerce")
				with st.spinner("Scoring batch..."):
					results = predict_batch(input_df, model_path)
					results["churn_label"] = (results["churn_probability"] >= threshold_b).astype(int)
				st.success(f"Scored {len(results)} rows")
				st.dataframe(results.head(100), use_container_width=True)
				csv_bytes = results.to_csv(index=False).encode("utf-8")
				st.download_button("Download results CSV", data=csv_bytes, file_name="churn_predictions.csv", mime="text/csv")
			except Exception as exc:
				st.error(f"Failed to score file: {exc}")
		else:
			st.info("Please upload a CSV with the input columns.")

#footer-
st.markdown("---")
st.caption("© 2025 Churn Predictor • Built with Streamlit and scikit-learn")
