# Importing Libraries
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

# Reads CSV file
df = pd.read_csv(r'C:\Users\...') #Input location of 2019_APD_Arrests_DATA.csv

dataset = df[['APD_RACE_DESC','RACE_KNOWN','Reason for Stop', 'Person Search YN','Search Based On','Search Found']]

dataset['APD_RACE_DESC'].replace({'WHITE': 1,'BLACK': 2, 'HISPANIC OR LATINO': 3, 'ASIAN': 4, 'AMERICAN INDIAN/ALASKAN NATIVE': 5, 'MIDDLE EASTERN': 6, 'HAWAIIAN/PACIFIC ISLANDER': 7, 'UNKNOWN': 8, 'nan': 0}, inplace=True)
dataset['RACE_KNOWN'].replace({'YES - RACE OR ETHNICITY WAS KNOWN BEFORE STOP': 1,'NO - RACE OR ETHNICITY WAS NOT KNOWN BEFORE STOP': 2,}, inplace=True)
dataset['Reason for Stop – TCOLE form'].replace({'Moving Traffic Violation': 1,'Pre-existing knowledge (i.e. warrant)': 2, 'Violation of law other than traffic': 3}, inplace=True)
dataset['Person Search YN'].replace({'YES = 1': 1,'NO = 2': 2,}, inplace=True)
dataset['Search Based On'].replace({'CONSENT': 1,'CONTRABAND/EVIDENCE IN PLAIN VIEW': 2, 'INCIDENTAL TO ARREST': 3, 'PROBABLE CAUSE':4, 'FRISK FOR SAFETY':5, 'INVENTORY OF TOWED VEHICLE': 6, 'nan':0}, inplace=True)
dataset['Search Found'].replace({'ALCOHOL': 1,'CASH': 2, 'DRUGS':3,'NOTHING':0, 'OTHER': 4, 'WEAPONS':5, 'nan':0}, inplace=True)


dataset['APD_RACE_DESC'] = dataset['APD_RACE_DESC'].replace(4, 0)
dataset['APD_RACE_DESC'] = dataset['APD_RACE_DESC'].replace(5, 3)
dataset['APD_RACE_DESC'] = dataset['APD_RACE_DESC'].replace(6, 3)
dataset['APD_RACE_DESC'] = dataset['APD_RACE_DESC'].replace(7, 3)
dataset['APD_RACE_DESC'] = dataset['APD_RACE_DESC'].replace(8, 0)
dataset['APD_RACE_DESC']= dataset['APD_RACE_DESC'].replace(0, 3)

dataset = dataset.dropna()

X = dataset['APD_RACE_DESC']
Y = dataset['RACE_KNOWN']

array = dataset.values

# Generating Matrix of Features
X = dataset.iloc[:,0:-5].values
# Generating Dependent Variable Vectors
Y = dataset.iloc[:,1:6].values
# Splitting dataset into training and testing dataset
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.21,random_state=0)

#Performing Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Initialising ANN
########################################################
# Define the keras model
model = Sequential()
# Adding Hidden Layers
model.add(Dense(6, activation='relu'))
model.add(tf.keras.layers.Dense(6, activation='relu'))
model.add(tf.keras.layers.Dense(6, activation='relu'))
# Adding Output Layer
model.add(Dense(5, activation='softmax'))
# Compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the keras model on the dataset
model.fit(X, Y, epochs=50, batch_size=100)
# Evaluate the keras model
_, accuracy = model.evaluate(X, Y)
print('Accuracy: %.2f' % (accuracy*100))
                                  

# Summarize the first 5 cases
Xnew, _ = make_blobs(n_samples=10)
scalar = MinMaxScaler()
scalar.fit(Xnew)
Xnew = scalar.transform(Xnew)

for i in range(len(Xnew)):
	print("X=%s, Predicted=%s" % (X[i], Y[i]))