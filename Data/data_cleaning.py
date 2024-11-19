import pandas as pd
import os

# Define the paths
rawpath = './Raw Data'
cleanedpath = './Cleaned Data'

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
Battery_Data.to_csv('./Cleaned Data/Battery_Data.csv', index=False)
Battery_Data.iloc[start:end].to_csv('./Cleaned Data/Battery_Data_0129_0529.csv', index=False)

carbon = pd.read_csv('./Cleaned Data/Carbon Intensity Data 2020.csv')
elecrate = pd.read_csv('./Cleaned Data/ElectricityRate2020.csv')
weather = pd.read_csv('./Cleaned Data/Weather Data.csv')
battery = pd.read_csv('./Cleaned Data/Battery_Data_0129_0529.csv')

date_times = pd.date_range(start="2024-01-01 00:00:00", end="2024-12-31 23:00:00", freq="h")

battery['datetime'] = pd.to_datetime(battery["datetime"])
battery.set_index('datetime')

Full_data = pd.DataFrame({
    "datetime": date_times,
    "Carbon Intensity (kgC02eq/kWh)": carbon['kg CO2e'].tolist(), 
    "Electricity Rate (NT$/kWh)": elecrate['Electricity_Rate_NTD/kWh'].tolist(),
    "Radiation Intensity, I": weather['Radiation, I'].tolist(),
    "Temperature,T (deg)":weather['Temp_average'].tolist()
})

Full_data = Full_data.merge(battery, on="datetime", how="left", suffixes=('', '_new'))

Full_data.to_csv('./Full_Data.csv')
