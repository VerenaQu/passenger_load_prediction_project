# Project: Passenger Load Prediction Project
This project aims to predict airline passenger load factors based on route characteristics and operational data. Using machine learning techniques such as regression models, cross-validation and hyperparameter optimization, the project identifies patterns that influence route occupancy and evaluates model performance for both existing and new routes.


# Table of Contents
1. Project Goals
2. Data
3. Requirements
4. Project Structure
5. How to run
6. Notes


# Project Goals
- Analyze historical flight and route data to understand key factors influencing passenger load factors
- Develop and benchmark multiple regression models to predict load factors
- Evaluate model performance under two conditions:
    - Existing routes (known patterns)
    - New routes (previously unseen routes)
- Perform hyperparameter tuning to explore model robustness and evaluate trade-offs between complexity and performance.
- Derive insights to support decision-making in airline route planning and capacity optimization


# Data
- The data used for the analysis has been downloaded from US Bureau of Transportation Statistics: "Air Carriers : T-100 Domestic Segment (All Carriers)", https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=GEE&QO_fu146_anzr=Nv4%20Pn44vr45
- Filters set:
    - Geography: All
    - Filter Year: 2024
    - Filter Period: All Month
    - Field Names:
        - DepScheduled
        - DepPerformed
        - Payload
        - Seats
        - Passengers
        - Freight
        - Mail
        - Distance
        - RampTime
        - AirTime
        - UniqueCarrier
        - UniqueCarrierName
        - CarrierGroup
        - OriginAirportID
        - OriginCityName
        - DestAirportID
        - Dest
        - DestCityName
        - AircraftType
        - AircraftConfig
        - Year
        - Month


# Requirements
You should have installed Python along with the following libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy
<br>
For detailed version specifications, please refer to the requirements.txt file.


# Project structure
```
project-root/
├── data/
│   └── aircraft_type_mapping.csv
│   └── config_mapping.csv
│   └── flight_data.csv
│   └── passenger_load_data.csv
├── notebooks/
│   └── 1_eda_and_cleaning.ipynb
│   └── 2_pax_load_factor_prediction.ipynb
├── README.md
├── requirements.txt
```



# How to Run
- Download the "Air Carriers: T-100 Domestic Segment (All Carriers)" dataset (or use provided sample)
- Open notebooks in JupyterLab or VS Code
- Run all cells step-by-step to reproduce the results



# Notes

### Datasets
- The notebook **2_pax_load_factor_prediction.ipynb** requires either the cleaned dataset from the first notebook or a custom dataset with at least the following recommended columns used in model training:


    | Column                   | Type          | Description |
    | -------------            | ------------- | -------------
    | `carrier_group`.         | categorical   | Airline group
    | `aircraft_type`          | categorical   | Aircraft model
    | `departures_performed`   | numeric       | Number of flights performed
    | `payload`                | numeric       | Total payload weight (kg)
    | `freight`                | numeric       | Freight load (kg)
    | `mail`                   | numeric       | Mail load (kg)
    | `distance`               | numeric       | Flight distance (km)
    | `air_time`               | numeric       | Average flight duration (minutes)
    | `pax_load_factor`        | numeric       | Target variable (percentage of seat occupancy)


- The model can be adapted to alternative or extended feature sets. To use different predictors, update the preprocessing pipeline and feature selection steps accordingly
- Aircraft type and configuration information were mapped using the provided reference files




### Models, Hyperparameter Tuning & Evaluation
- The analysis includes multiple regression approaches:
    - Baseline (Mean Predictor)
    - Linear Regression
    - Random Forest
    - Gradient Boosting 

- Both RandomizedSearchCV and GridSearchCV were applied for hyperparameter tuning of the best performing model
- SHAP (SHapley Additive exPlanations) was used to analyze feature importance and interpret model predictions
- Model evaluation included cross-validation, R², MAE, and RMSE metrics
- Route-based evaluation was performed using GroupShuffleSplit and GroupKFold to simulate predictions on unseen routes
- All experiments were conducted using scikit-learn pipelines to ensure reproducibility and clean preprocessing integration

