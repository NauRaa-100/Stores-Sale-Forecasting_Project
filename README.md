
* Description
* Dataset Info
* Workflow & Pipeline
* EDA & Cleaning Steps
* Feature Engineering
* Model Building
* Results
* Future Work

---

##  **README.md – Store Sales Forecasting**


#  Store Sales Forecasting

This project aims to predict **store sales** using structured data.  
It is inspired by real-world retail datasets, such as the *Superstore Sales Forecasting* dataset on Kaggle.

---

##  Project Overview

The goal of this project is to **analyze and forecast sales** for a retail business using machine learning.  
By studying sales patterns, customer segments, and product categories, we aim to identify the key drivers of profit and optimize business decisions.

---

##  Dataset Description

| Feature | Description |
|----------|-------------|
| Order ID | Unique ID for each order |
| Order Date / Ship Date | Dates for order and shipment |
| Ship Mode | Type of shipping (Standard, First Class, etc.) |
| Segment | Customer segment (Consumer, Corporate, etc.) |
| Country / City / State / Region | Geographical information |
| Product ID / Product Name / Category / Sub-Category | Product details |
| Sales | Total sales amount (Target variable) |
| Quantity | Number of units sold |
| Discount | Discount applied |
| Profit | Profit from the sale |
| Postal Code | Store location postal code |

 **Rows:** ~2,100  
 **Columns:** 21  

---

##  Data Cleaning & Preprocessing

**Steps performed:**

1. **Data Type Optimization**
   - Converted numerical columns (`Quantity`, `Sales`, `Profit`) to smaller data types (`int16`, `float16`).
   - Converted text columns to `category` for memory efficiency.

2. **Datetime Conversion**
   - Parsed `Order Date` and `Ship Date` using `pd.to_datetime`.
   - Extracted new features:
     - `Order Date-Year`
     - `Order Date-Month`
     - `Ship Date-Year`
     - `Ship Date-Month`

3. **Missing & Duplicate Values**
   - Verified:  No missing or duplicate records.

4. **Outlier Detection**
   - Detected outliers in `Quantity`, `Profit`, and `Sales` using IQR method.
   - Visualized via boxplots (optional).

---

##  Feature Engineering

**Encodings:**
- **Label Encoding:** `City`, `State`, `Product ID`, `Product Name`
- **Manual Mapping:**  
  - `Ship Mode`, `Segment`, `Region`
- **One-Hot Encoding:**  
  - Applied to `Sub-Category`

**New Columns Created:**
- Encoded versions of text data for ML models.
- Extracted time-based features from date columns.

**Dropped Irrelevant Columns:**
`Category`, `Country`, `Customer Name`, `Customer ID`, `Row ID`, `Order ID`

---

##  Feature Selection

Used **SelectKBest** with `f_regression` (for regression task) to find the most influential features.

```python
from sklearn.feature_selection import SelectKBest, f_regression

features_col = [
    'Quantity', 'Discount', 'Profit', 'Postal Code',
    'Ship Mode', 'Segment', 'Region',
    'City_Encode', 'State_Encode', 'Product ID_Encode'
] + list(sub_oh.columns)

X = df[features_col].fillna(0).values
y = df['Sales'].values

selector = SelectKBest(score_func=f_regression, k=5)
selector.fit(X, y)

selected_features = np.array(features_col)[selector.get_support()]
print(selected_features)
````

---

##  Scaling

Applied **StandardScaler** to normalize numerical features for model stability and performance.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

##  Model Building (Planned)

Models to be tested include:

* **Linear Regression**
* **Random Forest Regressor**
* **Gradient Boosting (XGBoost / LightGBM)**

Evaluation metrics:

* **MAE (Mean Absolute Error)**
* **RMSE (Root Mean Square Error)**
* **R² Score**

---

##  Insights

* **Profit** and **Discount** have the strongest correlation with **Sales**.
* **Customer Segment** and **Region** significantly influence sales distribution.
* Time-based features (year, month) can help model **seasonality** in future versions.
* Data shows a few **outliers** in high sales values — possibly large bulk orders.

---

##  Future Improvements

* Add **time-series forecasting** models (Prophet, ARIMA).
* Perform **feature interaction analysis** between discounts and profit margins.
* Build an interactive **dashboard** using Streamlit for visualization.
* Deploy the model as a **REST API** (Flask / FastAPI).

---

##  Tech Stack

| Tool           | Purpose                   |
| -------------- | ------------------------- |
| Python         | Core language             |
| Pandas / NumPy | Data analysis             |
| Matplotlib     | Visualization             |
| Scikit-learn   | ML models & preprocessing |
| Kaggle Dataset | Data source               |

---

##  Project Structure

```
Store_Sales_Forecasting/
│
├── stores_sales_forecasting.csv
├── store_sales_forecasting.ipynb
├── README.md
├── requirements.txt
└── models/
    ├── linear_regression.pkl
    └── random_forest.pkl
```

---

##  Author

**Nau Raa**
Machine Learning & Data Science Enthusiast

---
