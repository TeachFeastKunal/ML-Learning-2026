# 🚗 Used Car Price Prediction Engine

An end-to-end Exploratory Data Analysis and Machine Learning notebook built to analyze and predict the resale value of vehicles using marketplace data sourced directly from [Kaggle](https://kaggle.com).

---

## 📌 Problem Statement
Determining a fair market price for a used vehicle is complex due to non-linear variables like fast depreciation rates, fuel configurations, and mechanical transmission preferences. 

This project uses a data-driven approach to map out these trends and train predictive models, helping buyers and sellers estimate fair marketplace valuations.

---

## 📊 Dataset Specifications
The analysis utilizes the popular [Car price prediction(used cars) Dataset](https://kaggle.comdatasets/vijayaadithyanvg/car-price-predictionused-cars) uploaded by [vijayaadithyan V.G on Kaggle](https://kaggle.comdatasets/vijayaadithyanvg/car-price-predictionused-cars).

* **Dimensions**: 301 entries × 9 features (clean data with 0 missing values)
* **Target Variable**: `Selling_Price` (Resale value, denominated in Indian Lakhs. *e.g., 5.00 = ₹5,00,000*)
* **Key Features**: `Present_Price` (Showroom price when new), `Driven_kms` (Mileage), `Year` (Manufacture year), `Fuel_Type`, `Transmission`, `Selling_type`, and `Owner`.

---

## 🛠️ Notebook Workflow (`car_price_predictor.ipynb`)
The entire project is self-contained within a single, readable Jupyter Notebook file:

1. **Exploratory Data Analysis (EDA)**: Checking for missing values, running univariate distribution summaries, and plotting bivariate relationships against the target price.
2. **Feature Engineering**: Transforming the raw calendar `Year` column into a continuous `Car_Age` metric (`2026 - Year`) to improve regression learning.
3. **Data Preprocessing**: Safely dropping the inconsistent `Car_Name` text column and converting categorical options into binary variables via One-Hot Encoding (`pd.get_dummies`).
4. **Model Training**: Evaluating three distinct algorithm candidates (Linear Regression, Random Forest, and XGBoost).
5. **Hyperparameter Tuning**: Running `GridSearchCV` to optimize the decision tree parameters.

---

## 📈 Key Visual Insights From Our Plots
* **The Dominant Driver**: Our scatter plots revealed that `Present_Price` demonstrates a massive linear correlation with `Selling_Price`. Vehicles with a steep initial premium cleanly retain a higher baseline used valuation.
* **Premium Configs**: Bar charts highlighted that **Diesel** fuel systems, **Automatic** transmissions, and vehicles sold via authorized **Dealers** routinely clear significantly higher valuation margins.
* **Data Scarcity Edge Case**: Initial boxplots suggested a strange price bump for `Owner == 3`. Data inspection exposed this as a data mirage caused by a single premium luxury vehicle resting in a tiny sample size with zero general representation.

---

## 🤖 Model Evaluation Matrix

All configurations were tested against a hidden **20% testing split** using metrics from [Scikit-Learn Evaluation Metrics](https://scikit-learn.org).

### 📊 Performance Leaderboard

| Model Architecture | Hyperparameter Status | R² Score | MAE (Lakhs) | RMSE (Lakhs) |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression (Baseline)** | Default | 0.8489 | 1.2164 | 1.8658 |
| **XGBoost Regression** | Default | 0.9524 | 0.6199 | 1.0467 |
| **XGBoost Regression** | Tuned | 0.9478 | 0.6444 | 1.0970 |
| **Random Forest Regressor** | Default | 0.9595 | 0.6409 | 0.9656 |
| **Random Forest Regressor 🏆** | **Hypertuned** | **0.9609** | **0.6308** | **0.9495** |

### 🏆 Core Architecture Insights:
1. **Random Forest** captures peak performance, successfully mapping **96.1% of the total variance** in used vehicle resale valuations.
2. Ensembled decision tree paths handled the steep, localized drops of structural vehicle depreciation much more smoothly than basic global linear equations.

---

## 🚀 How to Open and Run This Notebook

### 1. Clone the Environment
```bash
git clone https://github.com
cd ML-Learning-2026
```

### 2. Install Required Packages
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
```

### 3. Launch Jupyter
Start your notebook server and open the main analysis file:
```bash
jupyter notebook
```
From the browser dashboard, click on **`car_price_predictor.ipynb`** and run all cells sequentially to reproduce the visualizations and model training results.
