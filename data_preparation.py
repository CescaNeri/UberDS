#Import Packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load Dataset
url = 'https://raw.githubusercontent.com/CescaNeri/UberDS/main/Dataset/uber_deliveryServiceDataset.csv'
df = pd.read_csv(url)

def scatter(vx, vy, vhue, vtitle):
    sns.relplot(data=df, x=vx, y=vy, hue=vhue)
    return plt.savefig(vtitle)

#Get Scatterplot
x = ['Requests', 'Requests', 'DeliveredProducts', 'SupplyHours', 'pETA']
y = ['Completes', 'Completes', 'TotalProductsAvailable', 'TimeOnTrip', 'aETA']
hue = ['Hour', 'Day', 'Hour', 'Hour', 'Hour']

for i in range(len(x)):
    vtitle = f'plot/scatter{i}.png'
    scatter(x[i], y[i], hue[i], vtitle)
    plt.close()

#Get Histogram
sum_completes_day = df.groupby('Day')['Completes'].sum().reset_index()
sns.barplot(data=sum_completes_day, x='Day', y='Completes')
plt.xlabel('Day')
plt.ylabel('Sum of Completes')
plt.savefig('plot\histo0')
plt.close()

sum_completes_hour = df.groupby('Hour')['Completes'].sum().reset_index()
sns.barplot(data=sum_completes_hour, x='Hour', y='Completes')
plt.xlabel('Hour')
plt.ylabel('Sum of Completes')
plt.savefig('plot\histo1')
plt.close()

#Get Correlation
columns = ['Hour', 'Requests', 'Completes', 'SupplyHours', 'TimeOnTrip', 'pETA', 'aETA', 'DeliveredProducts', 'TotalProductsAvailable']
data_included = df[columns]
sns.heatmap(data_included.corr(), annot=True, cmap='coolwarm')
plt.savefig('plot\correlation')

#Average Completion Rate
rate = df['Completes'].sum() / df['Requests'].sum()
print(f'Average Completion Rate: {rate}')




