# 🚗 Road Accident Severity Prediction Using Machine Learning

## Problem Statement

Road traffic accidents are a major cause of injury and death worldwide. Predicting the severity of an accident based on contributing factors can help emergency services, traffic authorities, and urban planners take proactive measures to reduce fatalities and serious injuries.

## Objective

Build a machine learning model that predicts the **severity of a road traffic accident** (Slight Injury, Serious Injury, or Fatal Injury) based on features such as weather conditions, road surface, light conditions, driver characteristics, vehicle type, and cause of accident.

## Dataset

The dataset is sourced from **Kaggle** and contains records of road traffic accidents with various contributing factors.

| File       | Rows  | Columns | Target Available |
|------------|-------|---------|-----------------|
| train.csv  | 8,210 | 32      | ✅ Yes           |
| test.csv   | 4,106 | 31      | ❌ No            |

**Target Column:** `Accident_severity`

**Target Classes:**
- Slight Injury
- Serious Injury
- Fatal Injury

## Features

| Feature | Type | Description |
|---------|------|-------------|
| Time | Object | Time of accident |
| Day_of_week | Object | Day of the week |
| Age_band_of_driver | Object | Driver's age band |
| Sex_of_driver | Object | Driver's sex |
| Educational_level | Object | Driver's education level |
| Vehicle_driver_relation | Object | Relationship to vehicle |
| Driving_experience | Object | Years of driving experience |
| Type_of_vehicle | Object | Type of vehicle involved |
| Owner_of_vehicle | Object | Vehicle ownership |
| Service_year_of_vehicle | Object | Vehicle service years |
| Defect_of_vehicle | Object | Known vehicle defects |
| Area_accident_occured | Object | Area type of accident |
| Lanes_or_Medians | Object | Road lane configuration |
| Road_allignment | Object | Road alignment |
| Types_of_Junction | Object | Junction type |
| Road_surface_type | Object | Road surface material |
| Road_surface_conditions | Object | Road surface condition |
| Light_conditions | Object | Lighting conditions |
| Weather_conditions | Object | Weather at time of accident |
| Type_of_collision | Object | Type of collision |
| Number_of_vehicles_involved | Numeric | Count of vehicles |
| Number_of_casualties | Numeric | Count of casualties |
| Vehicle_movement | Object | Vehicle movement at time |
| Casualty_class | Object | Casualty classification |
| Sex_of_casualty | Object | Casualty's sex |
| Age_band_of_casualty | Object | Casualty's age band |
| Casualty_severity | Object | Casualty severity |
| Work_of_casuality | Object | Casualty's occupation |
| Fitness_of_casuality | Object | Casualty's fitness |
| Pedestrian_movement | Object | Pedestrian movement |
| Cause_of_accident | Object | Cause of accident |

## Data Preprocessing

1. **Identifier removal** — Dropped `Num` column
2. **Duplicate detection** — Checked and removed duplicates
3. **Missing value handling** — Replaced string `na` with NaN; filled categoricals with "Unknown", numericals with median
4. **Whitespace stripping** — Cleaned inconsistent spacing
5. **Encoding** — `OneHotEncoder(handle_unknown="ignore")` for categorical features
6. **Scaling** — `StandardScaler` for numerical features
7. **Pipeline** — `ColumnTransformer` + `Pipeline` to prevent data leakage
8. **Fit on train only** — All transformers fitted exclusively on training data

## EDA (Exploratory Data Analysis)

Charts generated in `reports/`:

- Target (severity) distribution
- Severity vs Weather conditions
- Severity vs Road surface conditions
- Severity vs Light conditions
- Severity vs Road surface type
- Severity vs Type of collision
- Severity vs Day of week
- Severity vs Cause of accident
- Numerical feature distributions

## Machine Learning Algorithms

| Model | Description |
|-------|-------------|
| Logistic Regression | Linear model with L2 regularization, balanced class weights |
| Decision Tree | Non-linear tree-based classifier, balanced class weights |
| Random Forest | Ensemble of 200 decision trees, balanced class weights |

All models use `class_weight="balanced"` to handle the imbalanced target distribution.

## Evaluation Metrics

- **Accuracy** — Overall correct predictions
- **Precision (Macro)** — Average precision across all classes
- **Recall (Macro)** — Average recall across all classes
- **F1 Score (Macro)** — Harmonic mean of precision and recall (primary selection metric)
- **Classification Report** — Per-class precision, recall, F1
- **Confusion Matrix** — Visualization of predictions vs actuals

**Model selection criterion:** Macro F1 score (handles class imbalance fairly).

## Model Selection

The best model is selected automatically based on the **highest macro F1 score** on the validation set. No model is assumed to be best — the selection is entirely data-driven.

## Streamlit Application

The interactive web application has four pages:

1. **🏠 Home** — Project overview, objective, and ML workflow
2. **🔮 Predict Severity** — Input accident details and get severity prediction with probabilities
3. **📊 Model Performance** — Validation metrics, classification reports, and confusion matrices
4. **📈 Dataset Insights** — EDA visualizations

## Folder Structure

```
road-traffic-accidents/
│
├── train.csv                 # Training data (8,210 rows)
├── test.csv                  # Test data (4,106 rows, no target)
│
├── models/
│   ├── road_accident_severity_model.joblib   # Trained pipeline
│   └── model_metadata.json                   # Model metrics & config
│
├── reports/                  # EDA charts and confusion matrices
│
├── src/
│   ├── train_model.py        # Training script
│   └── predict.py            # Prediction utility
│
├── app.py                    # Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── .gitignore                # Git ignore rules
```

## Installation

```bash
# Clone or download the project
cd road-traffic-accidents

# Install dependencies
pip install -r requirements.txt
```

## Training

```bash
python src/train_model.py
```

This will:
- Load and clean the data
- Generate EDA charts in `reports/`
- Train and evaluate 3 models
- Save the best model pipeline to `models/`
- Save model metadata to `models/`

## Streamlit App

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

## Limitations

- The dataset is relatively small (~8,200 training samples)
- The target classes are highly imbalanced (Slight Injury dominates)
- The Kaggle test set has no target labels, so final unbiased evaluation is limited
- Some features contain many missing values or inconsistent categories
- The model is trained on a specific geographic region's accident data

## Future Scope

- Feature engineering (e.g., extracting hour from Time, combining related features)
- Advanced models (XGBoost, LightGBM, Neural Networks)
- Hyperparameter tuning with GridSearchCV or Optuna
- SMOTE or other oversampling techniques for class imbalance
- Deployment on cloud platforms (Heroku, AWS, GCP)
- Integration with real-time accident reporting systems
- Cross-validation for more robust model evaluation

---

**⚠️ Disclaimer:** This project is for **educational purposes only** and should not be used for real-world accident severity assessment.
