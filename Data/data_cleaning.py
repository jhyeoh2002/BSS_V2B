import pandas as pd
import os

# Define the paths
rawpath = './Raw Data'

# Read the files from the raw data directory
filenames = os.listdir(rawpath)
filenames.sort()

# Initialize DataFrames for each station
stationA = pd.DataFrame(columns=['time','batt_num'])
stationB = pd.DataFrame(columns=['time','batt_num'])

# Concatenate data from each station's files into the corresponding DataFrame
for f in filenames:
    if 'B' in f:
        raw = pd.read_csv(os.path.join(rawpath, f))
        stationB = pd.concat([stationB, raw])
    elif 'A' in f:
        raw = pd.read_csv(os.path.join(rawpath, f))
        stationA = pd.concat([stationA, raw])
    else:
        continue

# Ensure 'time' column is datetime and remove duplicates
stationA['time'] = pd.to_datetime(stationA['time'])
stationA = stationA.drop_duplicates(subset='time')
stationA.set_index('time', inplace=True)

stationB['time'] = pd.to_datetime(stationB['time'])
stationB = stationB.drop_duplicates(subset='time')
stationB.set_index('time', inplace=True)

# Resample both stations to hourly frequency, using nearest values within a limit
stationA_hourly = stationA.resample('h').nearest(limit=5)
stationB_hourly = stationB.resample('h').nearest(limit=5)

# Create a new DataFrame "Battery_Data" combining stationA and stationB columns
Battery_Data = pd.DataFrame({
    'datetime': stationA_hourly.index[:14960],
    'stationA': stationA_hourly['batt_num'].values[:14960],
    'stationB': stationB_hourly['batt_num'].values[:14960],
    'total': stationA_hourly['batt_num'].values[:14960] + stationB_hourly['batt_num'].values[:14960]
})

# Define start and end for slicing the data
start, end = 9432, 12360

# Save the complete Battery_Data to a CSV file
Battery_Data.to_csv('Battery_Data.csv', index=False)
Battery_Data.iloc[start:end].to_csv('Battery_Data_0129_0529.csv', index=False)
