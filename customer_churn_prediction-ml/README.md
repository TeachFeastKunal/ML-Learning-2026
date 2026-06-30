# 📞 Telco Customer Churn Predictive Modeling Pipeline

## 🎯 Problem Statement
Customer retention is one of the primary financial drivers in the telecommunications industry. Acquiring a new customer is significantly more expensive than retaining an existing one. The objective of this project is to build an end-to-end machine learning pipeline that **predicts which customers are likely to churn** based on their account profiles, services utilized, and demographic indicators. By accurately flagging high-risk accounts ahead of time, customer success teams can deploy highly targeted retention offers and operational adjustments to actively save recurring revenue.

## 📊 Dataset Description
*   **Source:** Kaggle (`blastchar/telco-customer-churn`).
*   **Data Footprint:** 7,043 rows and 21 columns.
*   **Target Column:** `Churn` (Indicates if the customer left within the last month).
*   **Class Imbalance:** **Heavily Imbalanced** at a ~1:3 ratio. Roughly **26.54%** of the dataset represents customers who churned (1,869 accounts), while **73.46%** represents retained accounts (5,174 accounts). This skew requires tracking metrics beyond accuracy, such as F1-Score, Recall, and ROC-AUC.

---

## 🛠️ Project Approach & Models
1.  **Exploratory Data Analysis (EDA):** Uncovered bimodal distributions in customer tenure, steep price premium floors for Fiber Optic lines, and major churn correlations with month-to-month contracts and Electronic Check payments.
2.  **Data Preprocessing & Cleaning:** Coerced broken string spaces in `TotalCharges` to float variables, handled missing rows via median imputation, and encoded pure binary features (`0`/`1`) without generating unnecessary column noise.
3.  **Feature Engineering:** Constructed a global `TotalServices` consumption metric and engineered a targeted **`HighCost_NoSupport`** indicator to isolate high-friction customer segments.
4.  **Framework Evaluation:** Trained, verified, and compared five distinct frameworks: Baseline Distance Models (Logistic Regression with `class_weight='balanced'`), Tree-based parallel ensembles (Default vs. Hypertuned Random Forest), and Sequential Tree Boosting (Default vs. Hypertuned XGBoost).
5.  **Probability Threshold Tuning:** Slid probability cutoffs past the standard 50% threshold limit (`0.38` for Random Forest; `0.60` for XGBoost) to optimize the precision-recall trade-off based on business cost realities.

---

## 📈 Master Performance Metrics Dashboard

| Model Framework | Overall Accuracy | Churn Precision | Churn Recall | Churn F1-Score | ROC-AUC Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Baseline)** | 79.28% | 64.34% | 49.20% | 0.5576 | 0.8274 |
| **XGBoost (Baseline)** | 78.00% | 59.64% | 52.94% | 0.5609 | 0.8203 |
| **Logistic Regression (Baseline)** | 73.95% | 50.60% | **78.88%** | 0.6165 | **0.8396** |
| **XGBoost (Tuned)** | 77.64% | 56.26% | 0.7086 | 0.6272 | 0.7548 |
| **Random Forest (Tuned)** | **78.92%** | **59.02%** | 0.6738 | **0.6292** | 0.7524 |

### 💡 Core Analytical Insights & Discoveries
*   **The Operational Winner — Random Forest (Tuned):** If corporate execution demands optimal budget efficiency, this model wins. It captures **252 out of 374 true churners** while containing false alarms to just 175, achieving our peak **F1-Score of 0.6292**.
*   **The Revenue-Catching Winner — Logistic Regression (Baseline):** If letting revenue walk out the door is the highest risk, this model wins. It blocks the maximum volume of churn with a massive **78.88% Recall**, backed by the highest raw ranking power (**0.8396 ROC-AUC**).
*   **The Critical Volatility Profile:** The combination of **short tenure + steep monthly charges ($80+) + Month-to-month contracts** represents the absolute highest risk segment. Permutation testing confirms that contract longevity is the ultimate anchor of subscriber loyalty.
*   **Support & Billing Friction:** Lacking Tech Support and paying via manual Electronic Check double the baseline risk of churn. Upfront support bundling and automated auto-pay conversions serve as immediate solutions.

---

## 💻 How to Run the Project in Google Colab

### 1. Launch via Google Colab
You can run this entire notebook instantly in the cloud without installing anything on your local computer:
1. Go to [Google Colab](https://google.com)
2. Click on **File > Upload Notebook**
3. Select the `.ipynb` file from this project repository

### 2. Execute the Pipeline
Once the notebook opens in Colab, the data will download automatically from the source URL. Simply click **Runtime > Run all** in the top menu bar to execute the entire predictive pipeline.
