import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sn
from datetime import datetime
from tabulate import tabulate as tb
mpl.use('tkagg')

dataset_name = 'sao_paulo.csv'
dataset_dir = '..' + os.sep + 'dataset'
dataset_location = os.path.join(dataset_dir, dataset_name)

cur_dir, _ = os.path.split(os.path.abspath(__file__))
dataset = os.path.join(cur_dir, dataset_location)

df = pd.read_csv(dataset, sep=';')

df.rename({'Swimming Pool': 'Swimming_Pool'}, axis=1, inplace=True)
df.rename({'Negotiation Type': 'Negotiation_Type'}, axis=1, inplace=True)
df.rename({'Property Type': 'Property_Type'}, axis=1, inplace=True)

# print(df.head())
print(tb(df.head(), headers = 'keys', tablefmt = 'grid'))

print(df.shape)

print(df.dtypes)

df.drop(['Latitude','Longitude'], axis = 1, inplace = True)

print(df['District'].unique())
# 96 unic Districts
print(len(df['District'].unique()))

# district = pd.get_dummies(df['District'])
df.drop(['District'], axis = 1, inplace = True)

# df_concat = pd.concat([df, district], axis = 1)
# df = df_concat

# print(df['Negotiation_Type'].unique())
# # 2 unic Negotiation Types
# print(len(df['Negotiation_Type'].unique()))

# df['Negotiation_Type'] = df['Negotiation_Type'].map({'sale' : 0, 'rent' : 1})
# df['Negotiation_Type'] = df['Negotiation_Type'].astype(int)

df = df.loc[df['Negotiation_Type']=='sale']

print(df['Property_Type'].unique())
# 1 unic Property Types
print(len(df['Property_Type'].unique()))

df['Property_Type'] = df['Property_Type'].str.replace('apartment','0')
df['Property_Type'] = df['Property_Type'].astype(int)

print(tb(df.head(), headers = 'keys', tablefmt = 'grid'))

print(df.dtypes)

def getMissingInfo( df ):
    # get amount missing data
    missing = df.isnull().sum()
    print( missing )
    # get percent missing data
    missing_percent = ( missing / len( df[df.columns[0]] ) ) * 100
    print( missing_percent )

getMissingInfo( df )

def getIRQ( column ):
    sort_column = column.sort_values(ascending=True)

    q1, q3 = sort_column.quantile([0.25, 0.75])

    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr) 
    upper_bound = q3 + (1.5 * iqr) 

    return lower_bound, upper_bound

lb, ub = getIRQ(df['Price'])

print("Lower bound: ", lb, " Upper bound: ", ub)

print("Total Uppers Outliers: ", df[df['Price'] > ub].count()['Price'])
print("Total Lowers Outliers: ", df[df['Price'] < lb].count()['Price'])

_, ax = plt.subplots()
ax.ticklabel_format(style='plain', useOffset=False, axis='both')
ax.boxplot(df['Price'])
plt.title("Price Boxplot")
plt.show()

correlation = df.corr()
plot = sn.heatmap(correlation, annot = True, fmt=".1f", linewidths=.6)
plot.plot()

def get_summary_statistics(column):
    mean = np.round(np.mean(column))
    median = np.round(np.median(column))
    min_value = np.round(column.min())
    max_value = np.round(column.max())
    quartile_1 = np.round(column.quantile(0.25))
    quartile_3 = np.round(column.quantile(0.75))
    # Interquartile range
    iqr = np.round(quartile_3 - quartile_1)
    print('Min: %s' % min_value)
    print('Mean: %s' % mean)
    print('Max: %s' % max_value)
    print('25th percentile: %s' % quartile_1)
    print('Median: %s' % median)
    print('75th percentile: %s' % quartile_3)
    print('Interquartile range (IQR): %s' % iqr)

get_summary_statistics(df['Price'])

now = datetime.now()
dt_str = now.strftime("%Y-%m-%d-%H-%M-%S")


df.drop(['Condo'], axis = 1, inplace = True)
df.drop(['Elevator'], axis = 1, inplace = True)
df.drop(['Furnished'], axis = 1, inplace = True)
df.drop(['Swimming_Pool'], axis = 1, inplace = True)
df.drop(['Negotiation_Type'], axis = 1, inplace = True)
df.drop(['Property_Type'], axis = 1, inplace = True)
df.drop(['New'], axis = 1, inplace = True)


pp_dataset_name = 'dataset_imoveis_' + dt_str + '-' + dataset_name
pp_dataset_location = os.path.join(dataset_dir, pp_dataset_name)
pp_dataset = os.path.join(cur_dir, pp_dataset_location)


df.to_csv(pp_dataset , index = False)
