

"""

-- Analysis of Store Sales Forecasting Dataset

From Kaggle about Structured Data ..

"""
#============================================== 
#Import Libraries
#============================================== 
import math
import warnings
import numpy as np
import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
import joblib
#============================================== 
#load Data --
#============================================== 

df_original=pd.read_csv(r'E:\Rev-DataScience\AI-ML\store.csv',encoding='latin')
print(df_original.head())
df=df_original.copy()

print('-----------Seperate------------')
print("Shape of data .. ",df.shape)
#Rows = 2121 , columns = 21

print(df.describe())

print(df.dtypes)

print('-----------Seperate------------')

#============================================== 
#Optimization --
#============================================== 
print(df.memory_usage(deep=True))
print('-----------Seperate------------')

#-*- Fix Data types

#A/ Integer Numbers Data

df['Row ID']=df['Row ID'].astype('int16')
df['Postal Code']=df['Postal Code'].astype('int16')
df['Quantity']=df['Quantity'].astype('int16')

#B/ Float Numbers Data

df['Profit']=df['Profit'].astype('float16')
df['Discount']=df['Discount'].astype('float16')
df['Sales']=df['Sales'].astype('float16')

#C/ Categorical & Objects Data
for col in df:
    if (df[col].dtype != 'object') | (col == 'Order ID'):
        continue
    else:
        df[col]=df[col].astype('category')
        print(df[col])
print(df.dtypes)

#-*- Fix Date Data

#-- Order Date
df['Order Date']=pd.to_datetime(df['Order Date'],errors='coerce',format='%Y%m%d')
df['Order Date-Year']=df['Order Date'].dt.year
df['Order Date-Month']=df['Order Date'].dt.month


#-- Ship Date
df['Ship Date']=pd.to_datetime(df['Ship Date'],errors='coerce',format='%Y%m%d')
df['Ship Date-Year']=df['Ship Date'].dt.year 
df['Ship Date-Month']=df['Ship Date'].dt.month 
print(df.memory_usage(deep=True))
print('-----------Seperate------------')

#==============================================
#Missing Values and Duplicated Values --
#==============================================

print(df.isnull().sum())
print(df.duplicated().sum())

#Theres no missing values or duplicated values 

print('-----------Seperate------------')

#==============================================
#Outliers --
#==============================================


cols=['Row ID','Postal Code','Quantity','Profit','Discount','Sales']

def find_outliers(series):
    Q1=series.quantile(0.25)
    Q3=series.quantile(0.75)
    IQR= Q3 - Q1
    lower= Q1 - 1.5 * IQR
    upper= Q3 + 1.5 * IQR
    return series[(series < lower) | (series > upper)]

for k in cols:
   print(f'{k} => {find_outliers(df[k]).shape[0]} Outliers ')

#three features have outliers values

print('-----------Seperate------------') 
#=======================================
#Visualization Of Outliers--
#=======================================

#Postal Code
plt.boxplot(df['Postal Code'])
plt.title('Outliers in Postal Code')
plt.show()

#Quantity

plt.boxplot(df['Quantity'])
plt.title('Outliers in Quantity')
plt.show()

#Profit

plt.boxplot(df['Profit'])
plt.title('Outliers in Profit')
plt.show()

#Sales

plt.boxplot(df['Sales'])
plt.title('Outliers in Sales')
plt.show()

print('-----------Seperate------------') 

#===============================================
# Encoding --
#===============================================

#Label Encoder
le= LabelEncoder()
#df['Order ID_Encode']=le.fit_transform(df['Order ID'])
#df['Customer ID_Encode']=le.fit_transform(df['Customer ID'])
#df['Country_Encode']=le.fit_transform(df['Country'])
df['City_Encode']=le.fit_transform(df['City'])
df['State_Encode']=le.fit_transform(df['State'])
df['Product ID_Encode']=le.fit_transform(df['Product ID'])
df['Product Name_Encod']=le.fit_transform(df['Product Name'])


#One-Hot-Code (Map)

df['Ship Mode']=df['Ship Mode'].map({'Standard Class':0,'First Class':1,'Seconed Class':2})
df['Segment']=df['Segment'].map({'Consumer':2,'Corporate':1,'Home Office':0})
df['Region']=df['Region'].map({'North':0,'East':1,'West':2,'South':3,'Central':4})


#Get_Dummies
sub_oh=pd.get_dummies(df['Sub-Category'],prefix='Sub')
df=pd.concat([sub_oh,df],axis=1)


#Cleaning Columns not need 
df=df.drop(columns=['Category','Country','Customer Name','Customer ID','Row ID','Order ID'])


print('-----------Seperate------------') 
#==============================================
#Relations --
#==============================================
#ٌRelations
#most Product saled ! 
print("Sales of Books => ",df['Sub_Bookcases'].sum())
print("Sales of Chairs => ",df['Sub_Chairs'].sum())
print("Sales of Furnishings => ",df['Sub_Furnishings'].sum())
print("Sales of Tables => ",df['Sub_Tables'].sum())
#Furnishings Most saled Then Chaires then Tables ! 
#least saled is Books 
#Most Product has highest discount 

print('-----------Seperate------------') 

#==============================================
#Features Selection
#==============================================

features_col = [
    'Quantity', 'Discount', 'Profit', 'Postal Code',
    'Ship Mode', 'Segment', 'Region',
    'City_Encode', 'State_Encode', 'Product ID_Encode'
] + list(sub_oh.columns)

features_col = [c for c in features_col if c in df.columns]
print("Features considered:", features_col)

X_df = df[features_col].copy()

for col in X_df.columns:
    if X_df[col].dtype == 'object' or str(X_df[col].dtype).startswith('category'):
        X_df[col] = pd.to_numeric(X_df[col], errors='coerce')

print("NaN per column before fill:\n", X_df.isnull().sum())

for col in X_df.columns:
    if X_df[col].isnull().any():
        X_df[col] = X_df[col].fillna(X_df[col].median())

y = df['Sales'].copy()
if y.isnull().any():
    y = y.fillna(y.median())

print("NaN after fix (X):", X_df.isnull().sum().sum(), " NaN in y:", y.isnull().sum())

# ======= 3) Feature selection (f_regression) =======
X = X_df.values
selector = SelectKBest(score_func=f_regression, k=5)
selector.fit(X, y)

selected_mask = selector.get_support()
selected_features = list(np.array(features_col)[selected_mask])
scores = selector.scores_

print("Selected Features:", selected_features)
print("Feature scores (top):")
for f, s in sorted(zip(features_col, scores), key=lambda x: -np.nan_to_num(x[1]))[:10]:
    print(f, round(float(s), 4))
#================================================
#Model--
#================================================

# ======= 4) Prepare final X using selected features =======
X_sel_df = X_df[selected_features].copy()

# scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sel_df.values)

# split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values, test_size=0.2, random_state=42)

# ======= 5) Train a simple model (SVR) and evaluate =======
model = SVR(kernel='linear', gamma='scale')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse=math.sqrt(mean_absolute_error(y_test,y_pred))
print("RMSE:",round(rmse,2))
print("R2:", round(r2,4))
print("MAE:", round(mae,4))
'''
# ======= 6) Plot Actual vs Predicted =======
plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()
'''
# ======= 7) Example: Predicted probability/target as function of one feature (Sub_Bookcases) =======
feature = 'Sub_Bookcases'
if feature in X_sel_df.columns:
    # build grid over the feature
    grid = np.linspace(X_sel_df[feature].min(), X_sel_df[feature].max(), 100)
    # baseline: median of other features
    baseline = X_sel_df.median().to_dict()
    X_grid = []
    for val in grid:
        row = [baseline[f] if f != feature else val for f in X_sel_df.columns]
        X_grid.append(row)
    X_grid = np.array(X_grid)
    X_grid_scaled = scaler.transform(X_grid)  # use same scaler
    preds = model.predict(X_grid_scaled)
    plt.figure(figsize=(7,4))
    plt.plot(grid, preds)
    plt.xlabel(feature)
    plt.ylabel("Predicted Sales")
    plt.title(f"Predicted Sales vs {feature}")
    plt.show()
else:
    print(f"{feature} not in selected features.")

residual=y_test - y_pred
sns.histplot(residual,kde=True)
plt.title('Residuals Distribution')
plt.show()

#Correlation
plt.figure(figsize=(10,7))
plt.title("Correlation Of Sales")
sns.heatmap(df[features_col + ['Sales']].corr(),annot=True,cmap='coolwarm')
plt.show()

#===========Saving Model==============

joblib.dump(model,'model.pkl')
joblib.dump(scaler,'scaler')
