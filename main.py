import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import pandas as pd

from scipy.optimize import minimize, basinhopping
from tqdm import tqdm
import matplotlib.pyplot as plt

from functions.generatecost import generate_building_cost, generate_ev_cost
# from functions.immediatecharging import get_EVimmediateCharge
from functions.optimization import Optimization
from functions.readcsv import readData

from scipy.stats import truncnorm

# TODO Set parameters (Modifiable)

#Set a name for this run
projName = 'Full'

#Charging Rate and Efficiency
charging_rate = 14       #kW   #Charging rate for gogoro station
charging_eff = 0.9

#Boundaries for Batteries
EVLowerBoundSoC = 0.2
EVUpperBoundSoC = 0.9

#Set number of iteration for basinhopping
iteration = 1

c_Batt = 9071 # Battery cost per kilowatt-hour ($/kWh)
pi_Cap = 1.5 # Battery energy capacity (kWh)
pi_CL = 2020 # Battery lifetime in terms of cycle life
pi_DoD = 0.7 # DoD for a certain cycle life
n_station = 38*2 # Number of batteries at the station
SOC_thr = 0.7 # Required leaving SOC

# Choose tnum
tnum = int(168) # 24 for a day optimization, 168 for a week optimization
days = int(tnum/24)

data = pd.read_csv('./Data/Full_Data.csv')
batteryinfo = pd.read_csv('./Data/Battery_info.csv')

BldgEnCon = data['energy (kWh)']
Temp = data['Temperature,T (deg)']
Rad = data['Radiation Intensity, I']
CarbInt = data['Carbon Intensity (kgC02eq/kWh)']

# Choose the start day
start_date = '2024-02-01' 
end_date = '2024-02-07' 

chosen_dateli = []

date_range = pd.date_range(start=start_date, end=end_date)
for chosen_date in date_range:
    chosen_dateli.append(chosen_date.strftime('%Y-%m-%d'))

# global D_B_t, S_R_t, N_V_t, n_total, t_a_v, t_d_v, a_vt, SOC_a_v, SOC_d_v, pi_Cn_t, c_G2B_t, c_G2V_t

a_vt = batteryinfo['Availability']
t_a_v = batteryinfo['Arrival_hour']
t_d_v = batteryinfo['Departure_hour']
SOC_a_v = batteryinfo['Arrival_SOC']
SOC_d_v = batteryinfo['Departure_SOC']

S_R_t = []
roof_area = 120 #m^2
pv_eff = 0.2036 #efficiency of PV cells

for i in range(tnum): ##Calculate generation by PV cells (kW)
    S_R_t.append(pv_eff*roof_area*Rad[i]*(1-0.005*(Temp[i]-25))/1000)
    
c_G2B_t = []
c_G2V_t = []。
pi_Cn_t = data['Electricity Rate (NT$/kWh)']
D_B_t = data['energy (kWh)']

for chosen_date in chosen_dateli:
    c_G2B_t.extend(generate_building_cost('twohigh', chosen_date))
    c_G2V_t.extend(generate_ev_cost('evlow', chosen_date))

print(len(c_G2B_t),len(c_G2V_t))


V2B = Optimization(chargingRate = charging_rate, chargingEff = charging_eff , 
                   gammaPeak = 1/3, gammaCost = 1/3, gammaCarbon = 1/3, 
                   lowestSoC = EVLowerBoundSoC, highestSoC = EVUpperBoundSoC)

date = pd.date_range(start='2023-06-16', end='2023-06-22', freq='D')
date_strings = date.strftime('%Y-%m-%d').tolist()

def get_EVimmediateCharge(days):

    EVChargingImmediate = np.zeros(24*days) #Create empty array for immediate charging
    
    for i in range(len(t_a_v)):
        
        required_energy = (SOC_d_v[i] - SOC_a_v[i]) * 1.5  #kWh
        rate = 14
        time = int(t_a_v[i])
        
        while required_energy > 0:

            if EVChargingImmediate[time] > rate:
                time+=1
            
            elif ( EVChargingImmediate[time] + required_energy ) > rate:
                available = rate - EVChargingImmediate[time]
                EVChargingImmediate[time]+= available
                required_energy -= available
                time+=1

            else:
                EVChargingImmediate[time]+= required_energy
                required_energy = 0
    
    return EVChargingImmediate
    
    print('Calculated Immediate Charging Slots')


projPath = './results/' + projName

if not os.path.exists(projPath):
    os.makedirs(projPath + '/npyFiles')
    os.makedirs(projPath + '/csv')
    os.makedirs(projPath + '/figures')

####################### Immediate Charging Slots #######################

if os.path.isfile(projPath + '/npyFiles/EVChargingImmediate.npy'):
    EVChargingImmediate = np.load(projPath + '/npyFiles/EVChargingImmediate.npy')
    print('File found: Loaded Immediate Charging Results from ./npyFiles\n')
    
else:
    EVChargingImmediate = get_EVimmediateCharge(days=days)
    np.save(projPath + '/npyFiles/EVChargingImmediate.npy', EVChargingImmediate)
    print('Saving Immediate Charging Results to ./npyFiles\n')

####################### V2B Charging/Discharging Slots #######################


if os.path.isfile(projPath + '/npyFiles/EVchargingV2Bnew.npy'):
    EVChargingV2B = np.load(projPath + '/npyFiles/EVchargingV2Bnew.npy')
    print('File found: Loaded V2B Charging/Discharging Results from ./npyFiles\n')
    
else:
    EVChargingV2B = V2B.optimize_ev(days = days, length = 24, iteration = iteration)
    np.save(projPath + '/npyFiles/EVchargingV2Bnew.npy', EVChargingV2B)
    print('Saving V2B Charging/Discharging Results to ./npyFiles\n')
    
# if os.path.isfile(projPath + '/npyFiles/EVchargingV2Bnew_12.npy'):
#     EVChargingV2B = np.load(projPath + '/npyFiles/EVchargingV2Bnew_12.npy')
#     print('File found: Loaded V2B Charging_12/Discharging Results from ./npyFiles\n')
    
# else:
#     EVChargingV2B = V2B.optimize_ev(days = days, length = 12, iteration = iteration)
#     np.save(projPath + '/npyFiles/EVchargingV2Bnew_12.npy', EVChargingV2B)
#     print('Saving V2B Charging_12/Discharging Results to ./npyFiles\n')
    
if os.path.isfile(projPath + '/npyFiles/EVchargingV2Bnewnew_6.npy'):
    EVChargingV2B = np.load(projPath + '/npyFiles/EVchargingV2Bnewnew_6.npy')
    print('File found: Loaded V2B Charging_6/Discharging Results from ./npyFiles\n')
    
else:
    EVChargingV2B = V2B.optimize_ev(days = days, length = 6, iteration = iteration)
    np.save(projPath + '/npyFiles/EVchargingV2Bnewnew_6.npy', EVChargingV2B)
    print('Saving V2B Charging_6/Discharging Results to ./npyFiles\n')
    
    