import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pymongo import MongoClient
from gridfs import GridFS
from bson.objectid import ObjectId

# -------------------------------------------------------------------
# Solar Voltage Predictor — Streamlit UI
# Built around the notebook's features:
# Light_Intensity_Lux + Temperature_C -> Voltage_V
# MongoDB database: solar_energy_db
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Solar Voltage Predictor",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MONGO_URI_DEFAULT = os.getenv("MONGO_URI")
DB_NAME = "solar_energy_db"
CLEAN_COLLECTION = "cleaned_solar_data"
MODEL_COLLECTION = "trained_models"
MODEL_FILENAME = "solar_voltage_model.sav"
CLEANED_CSV = "cleaned_solar_voltage_dataset.csv"

# ---- UI styling -----------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: linear-gradient(135deg, #f5fbf7 0%, #f6f8fc 55%, #eef7f4 100%); }
    [data-testid="stSidebar"] { background: #102b27; }
    [data-testid="stSidebar"] * { color: #f4faf6 !important; }
    .hero {
        padding: 1.65rem 1.8rem; border-radius: 22px;
        background: linear-gradient(115deg, #123c32 0%, #176c52 62%, #36a879 100%);
        color: white; margin-bottom: 1.25rem;
        box-shadow: 0 12px 32px rgba(18,60,50,.16);
    }
    .hero h1 { font-family: 'Space Grotesk', sans-serif; font-size: 2.25rem; margin: 0; color: white; }
    .hero p { color: #e0f4e9; margin: .45rem 0 0 0; font-size: 1.02rem; }
    .eyebrow { text-transform: uppercase; letter-spacing: .16em; font-size: .72rem; font-weight: 700; color: #b6ecd1; }
    .metric-card {
        padding: 1.05rem 1.2rem; border-radius: 16px; background: white;
        border: 1px solid #e3ece7; box-shadow: 0 4px 16px rgba(23,54,43,.045);
    }
    .metric-label { color: #60736b; font-size: .82rem; font-weight: 600; }
    .metric-value { color: #123c32; font-family: 'Space Grotesk', sans-serif; font-size: 1.55rem; font-weight: 700; margin-top: .2rem; }
    .section-title { color: #153c32; font-family: 'Space Grotesk', sans-serif; font-size: 1.3rem; font-weight: 700; margin: .3rem 0 .8rem 0; }
    .muted { color: #63776e; font-size: .9rem; }
    div.stButton > button[kind="primary"] {
        background: #167653; border: 0; border-radius: 11px; font-weight: 700;
        padding: .65rem 1rem; min-height: 2.8rem;
    }
    div.stButton > button[kind="primary"]:hover { background: #105b3e; border: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Sidebar / settings --------------------------------------------
st.sidebar.markdown("## ☀️ Solar Lab")
st.sidebar.caption("Solar voltage prediction dashboard")
mongo_uri = MONGO_URI_DEFAULT
use_mongodb = st.sidebar.toggle("Read model/data from MongoDB", value=True)
st.sidebar.markdown("---")
st.sidebar.markdown("**Model inputs**")
st.sidebar.caption("• Light intensity (lux)\n\n• Temperature (°C)")
st.sidebar.markdown("---")
st.sidebar.caption("Multiple Linear Regression · Scikit-learn")

# ---- Data and model loading ----------------------------------------
@st.cache_resource(show_spinner=False)
def get_mongo(uri):
    client = MongoClient(uri, serverSelectionTimeoutMS=2500)
    client.admin.command("ping")
    return client

def load_model_and_metadata(uri, use_db):
    model = None
    metadata = None
    source = "Local model file"
    if use_db:
        try:
            client = get_mongo(uri)
            db = client[DB_NAME]
            metadata = db[MODEL_COLLECTION].find_one(
                sort=[("created_at", -1)]
            )
            if metadata and metadata.get("model_file_id"):
                fs = GridFS(db, collection="model_files")
                file_doc = fs.get(ObjectId(metadata["model_file_id"])) if isinstance(metadata["model_file_id"], str) else fs.get(metadata["model_file_id"])
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".sav", delete=False) as temp:
                    temp.write(file_doc.read())
                    temp_path = temp.name
                try:
                    model = joblib.load(temp_path)
                finally:
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
                source = "MongoDB GridFS"
        except Exception:
            # MongoDB may be unavailable; use the local artifact below.
            pass
    if model is None and os.path.exists(MODEL_FILENAME):
        try:
            model = joblib.load(MODEL_FILENAME)
            source = "Local model file"
        except Exception:
            model = None
    return model, metadata, source

@st.cache_data(show_spinner=False)
def load_dataset(uri, use_db):
    if use_db:
        try:
            client = get_mongo(uri)
            docs = list(client[DB_NAME][CLEAN_COLLECTION].find({}, {"_id": 0}))
            if docs:
                return pd.DataFrame(docs), "MongoDB"
        except Exception:
            pass
    if os.path.exists(CLEANED_CSV):
        try:
            return pd.read_csv(CLEANED_CSV), "cleaned CSV"
        except Exception:
            pass
    return pd.DataFrame(), "not found"

model, model_metadata, model_source = load_model_and_metadata(mongo_uri, use_mongodb)
data, data_source = load_dataset(mongo_uri, use_mongodb)

# ---- Header ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Renewable energy · Predictive analytics</div>
      <h1>Solar Voltage Predictor</h1>
      <p>Estimate solar panel voltage from light intensity and ambient temperature.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Status messages ------------------------------------------------
if model is None:
    st.error(
        "No trained model was found. Run the model-training and `joblib.dump(model, "
        "'solar_voltage_model.sav')` cells in your notebook, then place the saved file "
        "in the same folder as this Streamlit app. If you want to load from MongoDB, "
        "make sure the latest model metadata and GridFS file have been saved."
    )
    st.stop()

# ---- Overview metrics ----------------------------------------------
valid_data = not data.empty
row_count = len(data) if valid_data else 0
metrics = model_metadata.get("evaluation_metrics", {}) if model_metadata else {}
metrics = metrics if isinstance(metrics, dict) else {}

# Support metric keys saved with either uppercase or lowercase names.
def _metric_value(metric_dict, key):
    value = metric_dict.get(key, metric_dict.get(key.lower()))
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None

r2_value = _metric_value(metrics, "R2")
mae_value = _metric_value(metrics, "MAE")
rmse_value = _metric_value(metrics, "RMSE")
metrics_source = "Saved test-set metrics"

# If the MongoDB model document does not include saved metrics, calculate a
# fallback using rows with measured voltage and both input features.
# This fallback uses the available dataset, so it is not a substitute for a
# properly held-out test-set evaluation.
if (r2_value is None or mae_value is None or rmse_value is None) and valid_data:
    required_columns = ["Light_Intensity_Lux", "Temperature_C", "Voltage_V"]
    if all(column in data.columns for column in required_columns):
        evaluation_data = data[required_columns].copy()
        for column in required_columns:
            evaluation_data[column] = pd.to_numeric(evaluation_data[column], errors="coerce")
        evaluation_data = evaluation_data.replace([np.inf, -np.inf], np.nan).dropna()
        if len(evaluation_data) >= 2:
            try:
                X_eval = evaluation_data[["Light_Intensity_Lux", "Temperature_C"]]
                y_eval = evaluation_data["Voltage_V"]
                y_eval_pred = model.predict(X_eval)
                if r2_value is None:
                    r2_value = float(r2_score(y_eval, y_eval_pred))
                if mae_value is None:
                    mae_value = float(mean_absolute_error(y_eval, y_eval_pred))
                if rmse_value is None:
                    rmse_value = float(np.sqrt(mean_squared_error(y_eval, y_eval_pred)))
                metrics_source = "Calculated from available dataset (not held-out test data)"
            except Exception as exc:
                st.warning(f"Could not calculate fallback model metrics: {exc}")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Dataset records</div><div class="metric-value">{row_count:,}</div><div class="muted">{data_source}</div></div>', unsafe_allow_html=True)
with col2:
    val = f"{r2_value:.3f}" if r2_value is not None else "Not available"
    st.markdown(f'<div class="metric-card"><div class="metric-label">R² score</div><div class="metric-value">{val}</div><div class="muted">Model evaluation</div></div>', unsafe_allow_html=True)
with col3:
    val = f"{mae_value:.3f} V" if mae_value is not None else "Not available"
    st.markdown(f'<div class="metric-card"><div class="metric-label">Mean absolute error</div><div class="metric-value">{val}</div><div class="muted">Lower is better</div></div>', unsafe_allow_html=True)
with col4:
    val = f"{rmse_value:.3f} V" if rmse_value is not None else "Not available"
    st.markdown(f'<div class="metric-card"><div class="metric-label">RMSE</div><div class="metric-value">{val}</div><div class="muted">Lower is better</div></div>', unsafe_allow_html=True)

if r2_value is not None or mae_value is not None or rmse_value is not None:
    st.caption(f"Metric source: {metrics_source}.")
else:
    st.info("Evaluation metrics are unavailable. Save them in MongoDB with the trained model, or provide a dataset containing Light_Intensity_Lux, Temperature_C, and measured Voltage_V.")

st.markdown("---")
predict_tab, explore_tab, about_tab = st.tabs(["⚡ Predict voltage", "📊 Explore dataset", "ℹ️ Model details"])

# ---- Prediction tab -------------------------------------------------
with predict_tab:
    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        st.markdown('<div class="section-title">Configure prediction</div>', unsafe_allow_html=True)
        st.markdown('<p class="muted">Enter current environmental measurements. The trained model uses the same two input features as your notebook.</p>', unsafe_allow_html=True)

        # Use sensible bounds from observed data when available.
        light_min, light_max, light_default = 0.0, 100000.0, 500.0
        temp_min, temp_max, temp_default = -20.0, 80.0, 25.0
        if valid_data:
            if "Light_Intensity_Lux" in data.columns:
                vals = pd.to_numeric(data["Light_Intensity_Lux"], errors="coerce").dropna()
                if not vals.empty:
                    light_min = min(0.0, float(vals.min()))
                    light_max = max(float(vals.max()) * 1.15, float(vals.min()) + 1)
                    light_default = float(vals.median())
            if "Temperature_C" in data.columns:
                vals = pd.to_numeric(data["Temperature_C"], errors="coerce").dropna()
                if not vals.empty:
                    temp_min = min(float(vals.min()) - 5, -20.0)
                    temp_max = max(float(vals.max()) + 5, 50.0)
                    temp_default = float(vals.median())

        with st.form("prediction_form"):
            light = st.number_input(
                "☀️ Light intensity (lux)", min_value=float(light_min),
                max_value=float(light_max), value=float(np.clip(light_default, light_min, light_max)),
                step=10.0, help="Measured light intensity in lux."
            )
            temperature = st.number_input(
                "🌡️ Temperature (°C)", min_value=float(temp_min),
                max_value=float(temp_max), value=float(np.clip(temp_default, temp_min, temp_max)),
                step=0.5, help="Ambient temperature in degrees Celsius."
            )
            submitted = st.form_submit_button("Predict voltage", type="primary", use_container_width=True)

        if submitted:
            try:
                input_frame = pd.DataFrame([{
                    "Light_Intensity_Lux": light,
                    "Temperature_C": temperature
                }])
                prediction = float(model.predict(input_frame)[0])
                st.session_state["last_prediction"] = prediction
                st.session_state["last_inputs"] = (light, temperature)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    with right:
        st.markdown('<div class="section-title">Prediction result</div>', unsafe_allow_html=True)
        if "last_prediction" in st.session_state:
            prediction = st.session_state["last_prediction"]
            last_light, last_temp = st.session_state["last_inputs"]
            st.markdown(
                f"""
                <div style="background:linear-gradient(145deg,#123c32,#1d8059);padding:1.5rem;border-radius:20px;color:white;box-shadow:0 12px 28px rgba(18,60,50,.15)">
                  <div style="color:#c6f4dc;text-transform:uppercase;letter-spacing:.12em;font-size:.75rem;font-weight:700">Estimated output voltage</div>
                  <div style="font-family:'Space Grotesk',sans-serif;font-size:3rem;font-weight:700;margin:.3rem 0">{prediction:.3f} <span style="font-size:1.2rem">V</span></div>
                  <div style="color:#e0f4e9;font-size:.9rem">Based on the measurements entered</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            a, b = st.columns(2)
            a.metric("Light intensity", f"{last_light:,.1f} lux")
            b.metric("Temperature", f"{last_temp:.1f} °C")
            st.caption("This is a model estimate, not a calibrated electrical safety measurement.")
        else:
            st.info("Enter light intensity and temperature, then select **Predict voltage** to see the estimate here.")
            st.markdown(
                """
                <div style="background:white;border:1px solid #e3ece7;border-radius:18px;padding:1.2rem">
                  <div style="font-weight:700;color:#153c32">How it works</div>
                  <p class="muted">The Multiple Linear Regression model combines both inputs to estimate Voltage_V.</p>
                  <div style="font-size:1.05rem;color:#176c52;font-weight:700">Voltage = intercept + (b₁ × light) + (b₂ × temperature)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ---- Dataset exploration tab ---------------------------------------
with explore_tab:
    st.markdown('<div class="section-title">Cleaned solar dataset</div>', unsafe_allow_html=True)
    if not valid_data:
        st.warning(f"No cleaned dataset found in MongoDB or `{CLEANED_CSV}`. Save your cleaned data from the notebook or enable MongoDB access.")
    else:
        k1, k2, k3 = st.columns(3)
        k1.metric("Rows", f"{len(data):,}")
        k2.metric("Columns", f"{len(data.columns):,}")
        missing_count = int(data.isna().sum().sum())
        k3.metric("Missing values", f"{missing_count:,}")
        search = st.text_input("Filter rows (search all columns)", placeholder="Type a value to filter the table…")
        display_data = data.copy()
        if search.strip():
            mask = display_data.astype(str).apply(
                lambda col: col.str.contains(search, case=False, na=False)
            ).any(axis=1)
            display_data = display_data.loc[mask]
        st.dataframe(display_data.head(1000), use_container_width=True, hide_index=True)
        st.download_button(
            "Download displayed data as CSV",
            data=display_data.to_csv(index=False).encode("utf-8"),
            file_name="solar_voltage_data.csv",
            mime="text/csv",
        )
        if "Voltage_V" in data.columns:
            plot_data = data.copy()
            for col in ["Voltage_V", "Light_Intensity_Lux", "Temperature_C"]:
                if col in plot_data.columns:
                    plot_data[col] = pd.to_numeric(plot_data[col], errors="coerce")
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                if "Light_Intensity_Lux" in plot_data.columns:
                    fig = px.scatter(
                        plot_data, x="Light_Intensity_Lux", y="Voltage_V",
                        color="Temperature_C" if "Temperature_C" in plot_data.columns else None,
                        title="Voltage vs light intensity",
                        labels={"Light_Intensity_Lux": "Light intensity (lux)", "Voltage_V": "Voltage (V)", "Temperature_C": "Temperature (°C)"},
                        color_continuous_scale="Tealgrn",
                    )
                    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=10,r=10,t=50,b=10))
                    st.plotly_chart(fig, use_container_width=True)
            with chart_col2:
                if "Temperature_C" in plot_data.columns:
                    fig = px.scatter(
                        plot_data, x="Temperature_C", y="Voltage_V",
                        title="Voltage vs temperature",
                        labels={"Temperature_C": "Temperature (°C)", "Voltage_V": "Voltage (V)"},
                        trendline="ols",
                    )
                    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=10,r=10,t=50,b=10))
                    st.plotly_chart(fig, use_container_width=True)

# ---- Model details tab ----------------------------------------------
with about_tab:
    st.markdown('<div class="section-title">Model information</div>', unsafe_allow_html=True)
    st.write("**Algorithm:** Multiple Linear Regression")
    st.write("**Target:** `Voltage_V`")
    st.write("**Input features:** `Light_Intensity_Lux`, `Temperature_C`")
    st.write(f"**Loaded from:** {model_source}")
    if model_metadata:
        coeffs = model_metadata.get("coefficients", {})
        st.write("**Intercept:**", model_metadata.get("intercept", "Not available"))
        st.write("**Coefficients:**", coeffs)
        created_at = model_metadata.get("created_at")
        if created_at:
            st.write("**Saved at:**", str(created_at))
        st.write("**Model document ID:**", str(model_metadata.get("_id", "")))
    else:
        st.info("Model metadata was not found in MongoDB. The local model file is being used, so saved coefficients and metrics may not be displayed.")
    st.markdown("---")
    st.markdown("**Model limitations**")
    st.caption(
        "Predictions are only as reliable as the training data and test evaluation. "
        "Use measurements within the range represented by your dataset where possible. "
        "The app estimates voltage; it does not replace real sensor readings or electrical protection systems."
    )

st.markdown(
    '<div style="text-align:center;color:#819188;font-size:.78rem;padding:1.3rem 0 0.5rem 0">Solar Voltage Predictor · Built with Streamlit, scikit-learn and MongoDB</div>',
    unsafe_allow_html=True,
)
