import pandas as pd

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




