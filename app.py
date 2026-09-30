

"""Streamlit demo. Run:  streamlit run app/app.py   (from project root, after python train.py)"""
import json
from pathlib import Path
import joblib, pandas as pd, streamlit as st

HERE = Path(__file__).parent
model = joblib.load(HERE / "model.joblib")   # same pipeline (imputer + scaler + classifier) used in training
meta = json.load(open(HERE / "meta.json"))

# Hard physical limits used for validation (pH scale is 0-14; others cannot be negative)
HARD = {"ph": (0.0, 14.0), "Turbidity": (0.0, 100.0)}

st.title("Water Quality Risk Classifier")
st.caption("Educational prototype - not a substitute for laboratory testing.")
st.write(f"Model: **{meta['model']}**  |  Test ROC-AUC: **{meta['test_metrics']['roc_auc']}**")

values = {}
cols = st.columns(2)
for i, f in enumerate(meta["features"]):
    lo, hi, med = meta["ranges"][f]
    hmin, hmax = HARD.get(f, (0.0, None))
    values[f] = cols[i % 2].number_input(f, min_value=hmin, max_value=hmax, value=float(round(med, 3)),
                                         help=f"Typical range in training data: {lo:.2f} - {hi:.2f}")

if st.button("Predict"):
    warn = [f for f in meta["features"] if not meta["ranges"][f][0] <= values[f] <= meta["ranges"][f][1]]
    if warn:
        st.warning("Outside the typical training range (prediction less reliable): " + ", ".join(warn))
    X = pd.DataFrame([values])[meta["features"]]
    p_unsafe = float(model.predict_proba(X)[0, 1])
    label = "UNSAFE (higher risk)" if p_unsafe >= 0.5 else "SAFE (lower risk)"
    (st.error if p_unsafe >= 0.5 else st.success)(f"Prediction: {label}")
    st.metric("Probability of unsafe", f"{p_unsafe:.1%}")
    st.progress(p_unsafe)
