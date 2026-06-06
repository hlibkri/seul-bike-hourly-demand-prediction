## Contributors:
- Hlib Kriukov*
- Jun Naruse*

*Saarland University

## Problem Discription
The goal of this project is to predict the hourly demand for shared bicycles in Seoul using historical usage data
combined with weather and temporal information. The project is divided into two tasks:
1. Regression: Predict the exact number of bikes rented per hour. This is a supervised regression problem
where the target variable is Rented Bike Count. Accurate predictions can help optimize bike availability,
reduce shortages or over-supply, and improve the efficiency of urban mobility systems.
2. Classification: Predict the demand category for a given hour. This is a supervised multi-class classification problem where the target variable is Demand Category, which takes one of three values — Low,
Normal, or High — derived from the distribution of hourly rentals. This task focuses on coarser-grained
demand forecasting, which can be useful for operational planning under uncertainty.

## Dataset Discription
### Dataset Description
- Domain: Urban Mobility / Business Analytics
- Task: Regression and Classification
- Target Variable (Regression): Rented Bike Count
- Target Variable (Classification): Demand Category
- Observations: Approximately 10,000 hourly records (about 7,000 used for training)
- Features: 15 input variables (each task’s dataset excludes the other task’s target variable)
### Features
#### Temporal Features
- Date – Calendar date of observation (year-month-day)
- Hour – Hour of the day (0–23)
- Seasons – Season of the year (Winter, Spring, Summer, Autumn)
- Holiday – Indicates whether the day is a public holiday (Holiday / No holiday)
- Functioning Day – Indicates whether bike rental service was operational (Yes / No)
#### Weather & Environmental Features
- Temperature – Air temperature (°C)
- Humidity – Relative humidity (%)
- Wind Speed – Wind speed (m/s)
- Visibility – Visibility distance (10m scale)
- Dew Point Temperature – Dew point temperature (°C)
- Solar Radiation – Solar radiation (MJ/m2)
- Rainfall – Precipitation (mm)
- Snowfall – Snow accumulation (cm)
#### Other Features
- Kaggle ID – Unique identifier for each record (used for Kaggle submission mapping and evaluation)
- Record id – Identifier found in the dataset
### Target Variable
This project has two target variables, one per task:
- Rented Bike Count (Regression) – Number of bicycles rented per hour
- Demand Category (Classification) – A three-class categorical variable derived from Rented Bike Count, indicating whether hourly demand was Low, Normal, or High. We encoded the three options to {0,1,2} respectively.

The two target variables are mutually exclusive across task datasets: the regression files do not contain Demand
Category, and the classification files do not contain Rented Bike Count

### Dataset Characteristics
- Type: Multivariate time-series regression dataset
- Data Quality: The dataset has been intentionally modified to reflect realistic imperfections commonly found in real-world data.
-  Missing Values: Present in select features
---
## Installation guide
Install required packages:
```
pip install -r requirements.txt
```
In the root folder create file "kaggle_token.json" with kaggle account api token: **Example**
```
{
    "username" : "hlibkriukov",
    "key"  : "******"
}
```
---
# Project developed during participation in core lecture Machine Learning