# importing required libraries
import pandas as pd
import numpy as np
from Echo_state import ESN
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings

warnings.filterwarnings('ignore')

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
    if i + 1 >= len(data):
        break
    diff = data.iloc[i + 1][1] - data.iloc[i][1]
    if (int(diff.seconds / 60)) == 30:
        if data.iloc[i][0] == data.iloc[i + 1][0]:
            count = count + 1
    elif (int(diff.seconds / 60)) == 60:
        if data.iloc[i][0] == data.iloc[i + 1][0]:
            count1 = count1 + 1
            list1.append(i)
    elif (int(diff.seconds / 60)) > 60:
        if data.iloc[i][0] == data.iloc[i + 1][0]:
            count2 = count2 + 1
            list2.append(i)
file = open('results', 'w')
file.write("ANALYSIS OF DATETIME DIFFERENCE BETWEEN CONSECUTIVE ROWS AMONG TOP 3 POWER CONSUMERS :\n\n" + "30 min"
                                                                                                          " DateTime difference count : " + str(
    count) + "\n1 hour DateTime difference count : "
           + str(count1) + "\nMore than 1 hour DateTime difference count : "
           + str(count2) + "\n\nList of rows with DateTime difference 1 hour :\n\n"
           + str(list1) + '\n\nList of rows with DateTime difference more then 1 hour :\n\n' + str(list2))
file.close()
print("Details of missing data are stored in \"results.txt\" file\n")

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
for i in range(3):
    plt.plot(l_grouped[i][1].DateTime, l_grouped[i][1].KWh)
    plt.xlabel("Date and Time")
    plt.ylabel("KWh")
    plt.title("Energy Consumption")
    plt.show()
    data1 = data.set_index('DateTime')
    # calculating rolling mean and standard deviation and visualizing them
    ts = l_grouped[i][1]['KWh']
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

# Echo state network
# storing aggregated data into a data frame
data = pd.read_csv('Merged_top3.csv')
data['DateTime'] = pd.to_datetime(data['DateTime'])
data = data.set_index(data['DateTime'])
grouped = data.groupby(['LCLid'])
l_grouped = list(grouped)
data = l_grouped[0][1]['KWh']
data1 = l_grouped[1][1]['KWh']
data2 = l_grouped[2][1]['KWh']
# storing energy consumption values of each house in numpy array
data = np.array(data).astype('float64')
data1 = np.array(data1).astype('float64')
data2 = np.array(data2).astype('float64')
# parameters for echo state network
n_reservoir = 500
sparsity = 0.2
rand_seed = 23
spectral_radius = 1
noise = 0.0003
# creating ESN objects
esn = ESN(n_inputs=1,
          n_outputs=1,
          n_reservoir=n_reservoir,
          sparsity=sparsity,
          random_state=rand_seed,
          spectral_radius=spectral_radius,
          noise=noise
          )
esn1 = ESN(n_inputs=1,
           n_outputs=1,
           n_reservoir=n_reservoir,
           sparsity=sparsity,
           random_state=rand_seed,
           spectral_radius=spectral_radius,
           noise=noise
           )
esn2 = ESN(n_inputs=1,
           n_outputs=1,
           n_reservoir=n_reservoir,
           sparsity=sparsity,
           random_state=rand_seed,
           spectral_radius=spectral_radius,
           noise=noise
           )

# method for calculating root mean square error(RMSE)
def rmse(yhat, y):
    return np.sqrt(np.mean((yhat.flatten() - y) ** 2))

# assigning training data length, number of future predictions for each epoch
trainlen = 15000
future = 1
futureTotal = 100

print("\n-------------------------Forecasting energy consumption of household MAC000018-------------------------\n")
pred_tot = np.zeros(futureTotal)
# each for loop iteration is an epoch
for i in range(0, futureTotal, future):
    print("Epoch :", i+1)
    # fitting the model
    pred_training = esn.fit(np.ones(trainlen), data[i:trainlen + i])
    # predicting future values
    prediction = esn.predict(np.ones(future))
    pred_tot[i:i + future] = prediction[:, 0]

# plotting original data with forecast data for MAC000018
plt.plot(range(0, trainlen + futureTotal), data[:trainlen + futureTotal], 'b', label="Data")
plt.plot(range(trainlen, trainlen + futureTotal), pred_tot, 'k', label='Free Running ESN')

lo, hi = plt.ylim()
plt.plot([trainlen, trainlen], [lo + np.spacing(1), hi - np.spacing(1)], 'k:', linewidth=4)

plt.title(r'Energy forecasting for household MAC000018', fontsize=25)
plt.xlabel(r'DateTime', fontsize=20, labelpad=10)
plt.ylabel(r'KWh', fontsize=20, labelpad=10)
plt.legend(fontsize='xx-large', loc='best')
plt.show()

# calculating RMSE
loss = rmse(pred_tot, data[trainlen:trainlen + futureTotal])
print("\nRoot mean square value for household", l_grouped[0][1]['LCLid'].unique()[0], "is :", loss)

print("\n-------------------------Forecasting energy consumption of household MAC000020-------------------------\n")
for i in range(0, futureTotal, future):
    print("Epoch :", i+1)
    pred_training = esn1.fit(np.ones(trainlen), data1[i:trainlen + i])
    prediction = esn1.predict(np.ones(future))
    pred_tot[i:i + future] = prediction[:, 0]

# plotting original data with forecast data for MAC000020
plt.plot(range(0, trainlen + futureTotal), data1[:trainlen + futureTotal], 'b', label="Data")
plt.plot(range(trainlen, trainlen + futureTotal), pred_tot, 'k', label='Free Running ESN')

lo, hi = plt.ylim()
plt.plot([trainlen, trainlen], [lo + np.spacing(1), hi - np.spacing(1)], 'k:', linewidth=4)

plt.title(r'Energy forecasting for household MAC000020', fontsize=25)
plt.xlabel(r'DateTime', fontsize=20, labelpad=10)
plt.ylabel(r'KWh', fontsize=20, labelpad=10)
plt.legend(fontsize='xx-large', loc='best')
plt.show()

# calculating RMSE
loss = rmse(pred_tot, data1[trainlen:trainlen + futureTotal])
print("\nRoot mean square value for household", l_grouped[1][1]['LCLid'].unique()[0], "is :", loss)

print("\n-------------------------Forecasting energy consumption of household MAC000021-------------------------\n")
for i in range(0, futureTotal, future):
    print("Epoch :", i+1)
    pred_training = esn2.fit(np.ones(trainlen), data2[i:trainlen + i])
    prediction = esn2.predict(np.ones(future))
    pred_tot[i:i + future] = prediction[:, 0]

# plotting original data with forecast data for MAC000021
plt.plot(range(0, trainlen + futureTotal), data2[:trainlen + futureTotal], 'b', label="Data")
plt.plot(range(trainlen, trainlen + futureTotal), pred_tot, 'k', label='Free Running ESN')

lo, hi = plt.ylim()
plt.plot([trainlen, trainlen], [lo + np.spacing(1), hi - np.spacing(1)], 'k:', linewidth=4)

plt.title(r'Energy forecasting for household MAC000021', fontsize=25)
plt.xlabel(r'DateTime', fontsize=20, labelpad=10)
plt.ylabel(r'KWh', fontsize=20, labelpad=10)
plt.legend(fontsize='xx-large', loc='best')
plt.show()

# calculating RMSE
loss = rmse(pred_tot, data2[trainlen:trainlen + futureTotal])
print("\nRoot mean square value for household", l_grouped[2][1]['LCLid'].unique()[0], "is :", loss)
