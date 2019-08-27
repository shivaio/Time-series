# importing required libraries
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

# reading the original data and storing it into pandas Data frame
data = pd.read_csv("Data.csv")
# variable declarations
row_count, row_list = [], []
dict_row_count, dictionary = {}, {}

# checking for columns with constant data and eliminating the row(reducing dimension of the data)
if len(data.stdorToU.unique()) == 1:
    data.drop(columns="stdorToU", axis=1, inplace=True)
if len(data.Acorn.unique()) == 1:
    data.drop(columns="Acorn", axis=1, inplace=True)
if len(data.Acorn_grouped.unique()) == 1:
    data.drop(columns="Acorn_grouped", axis=1, inplace=True)

# removing redundant rows in the data
pd.DataFrame.drop_duplicates(data, inplace=True)
# changing the string format of 'DateTime' column to Timestamp
data['DateTime'] = pd.to_datetime(data['DateTime'])

data = data.reset_index(drop=True)

# grouping the data together using 'LCLid' so that individual household can be accessed
grouped_items = data.groupby(['LCLid'])
list_of_grouped = list(grouped_items)

# counting the number of samples for each household
for i in range(len(data['LCLid'].unique())):
    dict_row_count[list_of_grouped[i][1].iloc[0][0]] = list_of_grouped[i][1].shape[0]

# sorting in descending order based on number of samples to get top3
sorted_dict = list(sorted(dict_row_count.items(), key=lambda x: x[1], reverse=True))

# printing top 3 households
print("-------------------------------------------TOP 3 HOUSEHOLDS-------------------------------------------\n")
print("1st :", sorted_dict[0][0])
print("2nd :", sorted_dict[1][0])
print("3rd :", sorted_dict[2][0], "\n")

# storing row indices which doesn't belong to top 3
indexNames = data[~((data['LCLid'] == sorted_dict[0][0]) | (data['LCLid'] == sorted_dict[1][0]) |
                    (data['LCLid'] == sorted_dict[2][0]))].index
data.drop(indexNames, inplace=True)
data.reset_index(drop=True, inplace=True)
print("Checking for missing data.....Please wait")
list1, list2 = [], []
count, count1, count2 = 0, 0, 0
# checking for missing rows and storing them into a file
for i in range(len(data)):
    if i+1 >= len(data):
        break
    diff = data.iloc[i+1][1] - data.iloc[i][1]
    if (int(diff.seconds/60)) == 30:
        if data.iloc[i][0] == data.iloc[i + 1][0]:
            count = count + 1
    elif (int(diff.seconds/60)) == 60:
        if data.iloc[i][0] == data.iloc[i+1][0]:
            count1 = count1 + 1
            list1.append(i)
    elif (int(diff.seconds/60)) > 60:
        if data.iloc[i][0] == data.iloc[i + 1][0]:
            count2 = count2 + 1
            list2.append(i)
file = open('results', 'w')
file.write("ANALYSIS OF DATETIME DIFFERENCE BETWEEN CONSECUTIVE ROWS AMONG TOP 3 POWER CONSUMERS :\n\n" + "30 min"
           " DateTime difference count : " + str(count) + "\n1 hour DateTime difference count : "
           + str(count1) + "\nMore than 1 hour DateTime difference count : "
           + str(count2) + "\n\nList of rows with DateTime difference 1 hour :\n\n"
           + str(list1) + '\n\nList of rows with DateTime difference more then 1 hour :\n\n' + str(list2))
file.close()
print("Details of missing data are stored in \"results.txt\" file\n")

# adding missing values
'''
    code for adding missing values will be added here
    
    after adding missing values it will be stored in "Cleaned.csv" file
'''

data = pd.read_csv('Cleaned.csv')
data['DateTime'] = pd.to_datetime(data['DateTime'])
grouped = data.groupby(['LCLid'])
l_grouped = list(grouped)

# normalizing the data by min-max normalization method
# calculating mean and variance of power consumption by individual households before aggregation
print("-------------------------------------------Mean of Normalized data before aggregation-----------------------"
      "--------------------\n")
for i in range(len(data['LCLid'].unique())):
    X = l_grouped[i][1]['KWh'].values
    ts = (l_grouped[i][1]['KWh'] - X.min()) / (X.max() - X.min())
    l_grouped[i][1]["KWh"] = ts
    print("Mean of", l_grouped[i][1]['LCLid'].unique()[0], "is", ts.mean())
    print("Variance of", l_grouped[i][1]['LCLid'].unique()[0], "is", ts.var())

normalized = l_grouped[0][1]
normalized = normalized.append(l_grouped[1][1], ignore_index=True)
normalized = normalized.append(l_grouped[2][1], ignore_index=True)
print("\nData aggregation is being done....Please wait")
i = 0
# aggregating data to 1 hour interval
while True:
    if i + 1 > normalized.shape[0] - 1:
        break
    elif (int((normalized.iloc[i + 1][1] - normalized.iloc[i][1]).seconds / 60) == 30 and (
            normalized.iloc[i][0] == normalized.iloc[i + 1][0])):
        dictionary['LCLid'] = normalized.iloc[i][0]
        dictionary['DateTime'] = normalized.iloc[i + 1][1]
        dictionary['KWh'] = (normalized.iloc[i][2] + normalized.iloc[i + 1][2]) / 2
        dictionary['Acorn'] = normalized.iloc[i][3]
        dictionary['Acorn_grouped'] = normalized.iloc[i][4]
        row_list.append(dictionary)
        i = i + 2
        dictionary = {}
    elif (int((normalized.iloc[i + 1][1] - normalized.iloc[i][1]).seconds / 60) == 60 and (
            normalized.iloc[i][0] == normalized.iloc[i + 1][0])):
        dictionary['LCLid'] = normalized.iloc[i][0]
        dictionary['DateTime'] = normalized.iloc[i + 1][1]
        dictionary['KWh'] = normalized.iloc[i + 1][2]
        dictionary['Acorn'] = normalized.iloc[i][3]
        dictionary['Acorn_grouped'] = normalized.iloc[i][4]
        row_list.append(dictionary)
        print("Found a 1 hr interval")
        print(normalized.iloc[i])
        dictionary = {}
        i = i + 2
    else:
        i = i + 1
        print("Found THE END of a house")
merged_data = pd.DataFrame(row_list)
merged_data.reset_index(drop=True, inplace=True)
merged_data['DateTime'] = pd.to_datetime(merged_data['DateTime'])
merged_data.to_csv("Merged_top3.csv", index=False)
print("Aggregated data is stored in \"Merged_top3.csv\" file\n")

# calculating mean and variance of power consumption by individual households after aggregation
print("-------------------------------------------Mean of Normalized data after aggregation-----------------------"
      "--------------------\n")
data = pd.read_csv('Merged_top3.csv')
data['DateTime'] = pd.to_datetime(data['DateTime'])
grouped = data.groupby(['LCLid'])
l_grouped = list(grouped)
for i in range(len(data['LCLid'].unique())):
    ts = l_grouped[i][1]["KWh"]
    print("Mean of", l_grouped[i][1]['LCLid'].unique()[0], "is", ts.mean())
    print("Variance of", l_grouped[i][1]['LCLid'].unique()[0], "is", ts.var())

# data visualization
# plotting graph of normalized data
plt.plot(l_grouped[1][1].DateTime, l_grouped[1][1].KWh)
plt.xlabel("Date and Time")
plt.ylabel("KWh")
plt.title("Energy Consumption")
plt.show()

# calculating rolling mean and standard deviation and visualizing them
data1 = data.set_index('DateTime')
ts = l_grouped[1][1]['KWh']

roll_mean = ts.rolling(24).mean()
roll_std = ts.rolling(24).std()

# Plot rolling statistics:
orig = plt.plot(ts, color='blue', label='Original')
mean = plt.plot(roll_mean, color='red', label='Rolling Mean')
std = plt.plot(roll_std, color='black', label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show()

# test for seasonality and trend
result = seasonal_decompose(ts, model='additive', freq=24)
result.plot()
plt.show()

# Augmented Dickey-fuller test
print('\nResults of Augmented Dickey-Fuller Test:\n')

X = ts.values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
