
# Solar Voltage Prediction System

A machine learning web application that predicts solar panel voltage using **Multiple Linear Regression** based on light intensity and temperature. Built with Python and Streamlit, the application provides an interactive dashboard for making predictions, exploring solar datasets, and reviewing model performance.

## Features

- **Voltage prediction:** Estimate solar panel voltage from light intensity (lux) and temperature (°C).
- **Interactive dashboard:** Use a Streamlit interface to enter measurements and view predictions.
- **Data exploration:** View, filter, analyze, and download solar energy datasets.
- **Data visualization:** Explore relationships between solar measurements using interactive charts.
- **Model evaluation:** Review Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score.
- **MongoDB integration:** Store cleaned solar data and trained-model metadata in MongoDB.
- **Model persistence:** Save and load the trained model for future predictions.
- **Environment configuration:** Read the MongoDB connection URI from an environment variable instead of hardcoding it.

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Plotly
- MongoDB and PyMongo
- Joblib

## How It Works

1. Load the solar dataset.
2. Clean the data and handle missing values and outliers.
3. Use `Light_Intensity_Lux` and `Temperature_C` as model features.
4. Train a Multiple Linear Regression model to predict `Voltage_V`.
5. Evaluate the model on held-out test data.
6. Store cleaned data and model metadata in MongoDB.
7. Enter new measurements in the dashboard to generate voltage predictions.

## Project Structure

Your files may vary, but a typical project structure is:

```text
project-folder/
├── solar_voltage_app.py
├── requirements_solar_app.txt
├── solar_voltage_model.sav
├── cleaned_solar_voltage_dataset.csv
├── .gitignore
└── README.md
```

The app may use the local model and dataset as fallbacks if the corresponding records are not available in MongoDB.

## Requirements

- Python 3.10 or a compatible Python version
- A running MongoDB server, either local or hosted with MongoDB Atlas
- The project files listed above

## Installation and Setup

### 1. Clone the repository

Replace the placeholders with your GitHub repository URL and folder name:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_PROJECT_FOLDER>
```

### 2. Create and activate a virtual environment (recommended)

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use a different terminal or run the project's Python executable directly.

### 3. Install dependencies

```powershell
python -m pip install -r requirements_solar_app.txt
```

### 4. Configure MongoDB

Set the connection string as an environment variable. For a local MongoDB server, run this in PowerShell:

```powershell
setx MONGO_URI "mongodb://127.0.0.1:27017/"
```

Close and reopen the terminal after running `setx`.

For MongoDB Atlas, set `MONGO_URI` to your Atlas connection string instead. Keep the username and password private. Do not put real credentials in source code or commit them to GitHub.

The application should read the variable in Python with:

```python
import os

MONGO_URI = os.getenv("MONGO_URI")
```

Ensure the app's MongoDB connection code checks that `MONGO_URI` is set before connecting.

### 5. Check required local files

If your app relies on local fallback files, make sure these exist in the project folder:

- `solar_voltage_model.sav`
- `cleaned_solar_voltage_dataset.csv`

Do not commit a dataset if it contains private, sensitive, or restricted information.

### 6. Run the application

```powershell
python -m streamlit run solar_voltage_app.py
```

Open the local URL shown in the terminal to use the dashboard.

## Model Inputs and Output

| Type | Column | Description |
|---|---|---|
| Input feature | `Light_Intensity_Lux` | Measured light intensity in lux |
| Input feature | `Temperature_C` | Temperature in degrees Celsius |
| Prediction target | `Voltage_V` | Solar voltage predicted by the model |

Predictions are estimates based on the data used to train the model. Their accuracy depends on data quality and how representative the training data is of real operating conditions.

## Model Evaluation

The application can report these regression metrics:

- **MAE:** Average absolute difference between actual and predicted voltage.
- **RMSE:** Square root of the average squared prediction error; larger errors are penalized more heavily.
- **R²:** Indicates how well the model explains variation in the target on the evaluation data.

Add your actual evaluation scores here after training the final model. Do not report example values as measured results.

## MongoDB Storage

The project is designed to use the `solar_energy_db` database and these collections:

- `cleaned_solar_data` — cleaned dataset records
- `trained_models` — model details, features, target, coefficients, and evaluation metrics
- `model_files` — GridFS storage for serialized model files, if configured by the application

MongoDB creates databases and collections as data is inserted, so an empty database or collection may not appear until it contains data.

## Security

Example `.gitignore` entries:

```gitignore
.env
.env.*
!.env.example
.venv/
venv/
__pycache__/
*.py[cod]
```


=======
# solar-energy-prediction
A machine learning web application that predicts solar panel voltage using Multiple Linear Regression based on light intensity and temperature. Built with Python, Streamlit, and MongoDB, it provides interactive predictions, data visualization, dataset exploration, and model performance evaluation.
>>>>>>> a0d1e0ffd4bf5a830d3cb685592bfcb0d15cf3eb
