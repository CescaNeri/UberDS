import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import svm

#Load Dataset
url = 'https://raw.githubusercontent.com/CescaNeri/UberDS/main/Dataset/uber_deliveryServiceDataset.csv'
df = pd.read_csv(url)

#Add Features
def features(df):
    df['TimeEfficiency'] = df['TimeOnTrip'] / df['SupplyHours']
    df['CompletionRate'] = df['Completes'] / df['Requests']
    df['ETADifference'] = df['aETA'] - df['pETA']
    df['ProductsDeliveredPercentage'] = (df['DeliveredProducts'] / df['TotalProductsAvailable']) * 100

features(df)

df['DayNum'] = df['Day'].map({'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7})

#CLASSIFICATION - TARGET 0-1

#Define the Target for Classification
average_completion_rate = df['CompletionRate'].mean()
df['Target'] = df['CompletionRate'].apply(lambda x: 1 if x > average_completion_rate else 0)

#Build the Train and Test Set
X = df.drop('Target', axis=1)
X = X.drop('Date', axis=1)
X = X.drop("Day", axis=1)
X = X.drop('Requests', axis=1)
X = X.drop('SupplyHours', axis=1)
X = X.drop('TotalProductsAvailable', axis=1)
X = X.drop('pETA', axis=1)
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Execute the Models
names = [
    "SVC",
    "Random Forest",
    "Ada Boost",
    "Decision Tree",
    "Quadratic Discrimination"
]

classifiers = [
    svm.SVC(gamma=0.001, C=100., kernel='rbf', verbose=False, probability=False),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    DecisionTreeClassifier(),
    QuadraticDiscriminantAnalysis()
]

for name, clf in zip(names, classifiers):
  score = clf.fit(X_train, y_train)

  pred_y = clf.predict(X_test)
  print('Final Accuracy: {}, {:.3f}'.format((name),accuracy_score(y_test, pred_y)))

#LINEAR REGRESSION - PREDICT REQUESTS

Xlr = df.drop('Date', axis=1)
Xlr = Xlr.drop('Day', axis=1)
#Xlr = Xlr.drop('Requests', axis=1)
Xlr = Xlr.drop('SupplyHours', axis=1)
Xlr = Xlr.drop('TotalProductsAvailable', axis=1)
Xlr = Xlr.drop('pETA', axis=1)
ylr = df['Requests']

X_train, X_test, y_train, y_test = train_test_split(Xlr, ylr, test_size=0.2, random_state=42)

linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

y_pred = linear_regression.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

#RECURRENT NEURAL NETWORK - PREDICT REQUESTS

Xrnn = df.drop('Date', axis=1)
Xrnn = Xrnn.drop('Day', axis=1)
yrnn = df['Requests']

X_train, X_test, y_train, y_test = train_test_split(Xrnn, yrnn, test_size=0.2, random_state=42)

#Scale and Reshape
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

model = Sequential()
model.add(SimpleRNN(units=64, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=400, batch_size=32)

loss = model.evaluate(X_test, y_test)
print(f"Loss: {loss}")






