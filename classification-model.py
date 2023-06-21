import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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

#Define the Target for Classification
average_completion_rate = df['CompletionRate'].mean()
df['Target'] = df['CompletionRate'].apply(lambda x: 1 if x > average_completion_rate else 0)

#Build the Train and Test Set
X = df.drop('Target', axis=1)
X = X.drop('Date', axis=1)
X = X.drop("Day", axis=1)
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Execute the Model
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