# Importing Libraries
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
# Import train_test_split function
from sklearn.model_selection import train_test_split
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Reads CSV file
df = pd.read_csv(r"C:\Users\...") #Input location of 2019_APD_Arrests_DATA.csv

dataset = df[['APD_RACE_DESC','RACE_KNOWN','Reason for Stop', 'Person Search YN','Search Based On','Search Found']]

dataset['APD_RACE_DESC'].replace({'WHITE': 1,'BLACK': 2, 'HISPANIC OR LATINO': 3, 'ASIAN': 4, 'AMERICAN INDIAN/ALASKAN NATIVE': 5, 'MIDDLE EASTERN': 6, 'HAWAIIAN/PACIFIC ISLANDER': 7, 'UNKNOWN': 8, 'nan': 0}, inplace=True)
dataset['RACE_KNOWN'].replace({'YES - RACE OR ETHNICITY WAS KNOWN BEFORE STOP': 1,'NO - RACE OR ETHNICITY WAS NOT KNOWN BEFORE STOP': 2,}, inplace=True)
dataset['Reason for Stop'].replace({'Moving Traffic Violation': 1,'Pre-existing knowledge (i.e. warrant)': 2, 'Violation of law other than traffic': 3}, inplace=True)
dataset['Person Search YN'].replace({'YES = 1': 1,'NO = 2': 2,}, inplace=True)
dataset['Search Based On'].replace({'CONSENT': 1,'CONTRABAND/EVIDENCE IN PLAIN VIEW': 2, 'INCIDENTAL TO ARREST': 3, 'PROBABLE CAUSE':4, 'FRISK FOR SAFETY':5, 'INVENTORY OF TOWED VEHICLE': 6, 'nan':0}, inplace=True)
dataset['Search Found'].replace({'ALCOHOL': 1,'CASH': 2, 'DRUGS':3,'NOTHING':0, 'OTHER': 4, 'WEAPONS':5, 'nan':0}, inplace=True)

names = ['APD_RACE_DESC', 'Person Search YN', 'Reason for Stop – TCOLE form', 'Search Based On', 'Search Found', 'RACE_KNOWN' ]
dataset = dataset.dropna()


X = dataset['APD_RACE_DESC']
y = dataset['RACE_KNOWN']

array = dataset.values
X = dataset.iloc[:,0:-5].values

Y = dataset.iloc[:,1:6].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
num_trees = 30
kfold = model_selection.KFold(n_splits=10)
abc = AdaBoostClassifier(n_estimators=50,learning_rate=1)
model = abc.fit(X_train, y_train)
results = model_selection.cross_val_score(model, X, y, cv=kfold)
y_pred = model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))