# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, f_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# -------------------------------
# 1. Load Data
# -------------------------------
a1 = pd.read_excel("case_study1.xlsx")
a2 = pd.read_excel("case_study2.xlsx")

df1 = a1.copy()
df2 = a2.copy()

# -------------------------------
# 2. Clean Data
# -------------------------------
# Remove null-like values
df1 = df1.loc[df1['Age_Oldest_TL'] != -99999]

columns_to_be_removed = []
for col in df2.columns:
    if df2.loc[df2[col] == -99999].shape[0] > 10000:
        columns_to_be_removed.append(col)

df2 = df2.drop(columns_to_be_removed, axis=1)

for col in df2.columns:
    df2 = df2.loc[df2[col] != -99999]

# -------------------------------
# 3. Merge Data
# -------------------------------
df = pd.merge(df1, df2, how='inner', on='PROSPECTID')

# -------------------------------
# 4. Feature Selection
# -------------------------------
# Chi-square test for categorical
for col in ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']:
    chi2, pval, _, _ = chi2_contingency(pd.crosstab(df[col], df['Approved_Flag']))
    print(f"{col} -> p-value: {pval}")

# VIF for numeric
numeric_columns = [c for c in df.columns if df[c].dtype != 'object' and c not in ['PROSPECTID','Approved_Flag']]
vif_data = df[numeric_columns].copy()
columns_to_be_kept = []
for i in range(len(numeric_columns)):
    vif_value = variance_inflation_factor(vif_data, i)
    if vif_value <= 6:
        columns_to_be_kept.append(numeric_columns[i])

# ANOVA for numeric
columns_to_be_kept_numerical = []
for col in columns_to_be_kept:
    a = df[col].tolist()
    b = df['Approved_Flag'].tolist()
    groups = {flag: [val for val, grp in zip(a, b) if grp == flag] for flag in set(b)}
    f_stat, p_val = f_oneway(*groups.values())
    if p_val <= 0.05:
        columns_to_be_kept_numerical.append(col)

# Final features
features = columns_to_be_kept_numerical + ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
df = df[features + ['Approved_Flag']]

# -------------------------------
# 5. Encoding
# -------------------------------
# Ordinal encoding for EDUCATION
edu_map = {
    'SSC': 1, '12TH': 2, 'GRADUATE': 3, 'UNDER GRADUATE': 3,
    'POST-GRADUATE': 4, 'OTHERS': 1, 'PROFESSIONAL': 3
}
df['EDUCATION'] = df['EDUCATION'].map(edu_map).astype(int)

# One-hot encode categorical
df_encoded = pd.get_dummies(df, columns=['MARITALSTATUS','GENDER','last_prod_enq2','first_prod_enq2'])

# -------------------------------
# 6. Machine Learning Models
# -------------------------------
X = df_encoded.drop('Approved_Flag', axis=1)
y = df_encoded['Approved_Flag']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("\nRandom Forest Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred)
print("Precision:", prec)
print("Recall:", rec)
print("F1:", f1)

# ---- XGBoost
le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=4)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

print("\nXGBoost Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred)
print("Precision:", prec)
print("Recall:", rec)
print("F1:", f1)

# ---- Decision Tree
dt = DecisionTreeClassifier(max_depth=20, min_samples_split=10)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

print("\nDecision Tree Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred)
print("Precision:", prec)
print("Recall:", rec)
print("F1:", f1)

