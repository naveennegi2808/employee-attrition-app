import pandas as pd
import numpy as np
df=pd.read_csv("Employee.csv")
(df.head(10))
(df.columns)
 

new_df=df[['OverTime',
    'JobSatisfaction',
    'EnvironmentSatisfaction',
    'MonthlyIncome',
    'Age',
    'YearsAtCompany',
    'TotalWorkingYears',
    'DistanceFromHome',
    'JobRole',
    'BusinessTravel']]
(new_df)

"""print(new_df[new_df[['OverTime',
    'JobSatisfaction',
    'EnvironmentSatisfaction',
    'MonthlyIncome',
    'Age',
    'YearsAtCompany',
    'TotalWorkingYears',
    'DistanceFromHome',
    'JobRole',
    'BusinessTravel']]].isnull())"""

#print(type(new_df.info()))
categorical_cols = ['OverTime', 'JobRole', 'BusinessTravel']

df_encoded = pd.get_dummies(new_df, columns=categorical_cols, drop_first=True)

#print(df_encoded.columns)  
#print(df["JobRole"].nunique())

#print(df_encoded.isnull().values.any())
df_encoded = df_encoded.astype(int)


df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
df_encoded["Attrition"] = df["Attrition"]
df_encoded.to_csv("df_encoded.csv", index=False)

#print(df_encoded.head())
from sklearn.model_selection import train_test_split
X=df_encoded.drop("Attrition",axis=1)
y=df_encoded["Attrition"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101, stratify=y)



from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import classification_report
rfc = BalancedRandomForestClassifier(n_estimators=100, random_state=101, class_weight='balanced')

rfc.fit(X_train, y_train)


pred=rfc.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))

import joblib
joblib.dump(rfc, 'model.pkl')

