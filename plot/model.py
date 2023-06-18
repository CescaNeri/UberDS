import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Load Dataset
url = 'https://raw.githubusercontent.com/CescaNeri/UberDS/main/Dataset/uber_deliveryServiceDataset.csv'
df = pd.read_csv(url)

def features(df):
    df['TimeEfficiency'] = df['TimeOnTrip'] / df['SupplyHours']
    df['CompletionRate'] = df['Completes'] / df['Requests']
    df['ETADifference'] = df['aETA'] - df['pETA']
    df['ProductsDeliveredPercentage'] = (df['DeliveredProducts'] / df['TotalProductsAvailable']) * 100

#Add Features
features(df)

#Create Target for Classification Model
average_completion_rate = df['CompletionRate'].mean()
df['Target'] = df['CompletionRate'].apply(lambda x: 1 if x > average_completion_rate else 0)


