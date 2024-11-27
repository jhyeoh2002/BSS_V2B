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
# from functions.optimization import Optimization
from functions.readcsv import readData

from scipy.stats import truncnorm

# TODO Set parameters (Modifiable)

#Set a name for this run
projName = 'GuangZhou'

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

BldgEnCon = data['Carbon Intensity (kgC02eq/kWh)']
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

global D_B_t, S_R_t, N_V_t, n_total, t_a_v, t_d_v, a_vt, SOC_a_v, SOC_d_v, pi_Cn_t, c_G2B_t, c_G2V_t

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
c_G2V_t = []
pi_Cn_t = data['Electricity Rate (NT$/kWh)']
D_B_t = data['energy (kWh)']

for chosen_date in chosen_dateli:
    c_G2B_t.extend(generate_building_cost('twohigh', chosen_date))
    c_G2V_t.extend(generate_ev_cost('evlow', chosen_date))

print(len(c_G2B_t),len(c_G2V_t))

class Optimization: 
    
    def __init__(self, chargingRate, chargingEff, gammaPeak, gammaCost, gammaCarbon, lowestSoC, highestSoC, ESSCapacity = None):
        
        self.gamma_peak = gammaPeak
        self.gamma_cost = gammaCost
        self.gamma_carbon = gammaCarbon
        self.chargingRate = chargingRate
        self.chargingEff = chargingEff
        self.lowest = lowestSoC
        self.highest = highestSoC
        self.ESSCapacity = ESSCapacity
        self.BattInfo = pd.read_csv("./Data/Battery_info.csv")

    def cal_stress(self, GD) -> np.ndarray:        
        return np.array([(x ** 2) if x > 0 else 0 for x in GD])

    def cal_carbon(self, GD, hour) -> np.ndarray:   
        return np.array([(x * pi_Cn_t[hour + i]) if x > 0 else 0 for i, x in enumerate(GD)])

    def cal_cost(self, GD, EVelec, hour) -> np.ndarray:   
        buildingcost = [(x * c_G2B_t[hour + i]) if x > 0 else 0 for i, x in enumerate(GD)]
        evcost = [(x * c_G2V_t[hour + i]) if x > 0 else (x * c_G2B_t[hour + i]) for i, x in enumerate(EVelec)]
        return np.array([buildingcost[i] + evcost[i] for i in range(len(GD))])
               
    def minmax_scaling(self, x_list, x_max):    
        x_min = np.min(x_list)
        return np.array([(x - x_min)/(x_max-x_min)for x in x_list])

    def get_SOC(self, EV_initial,capacity,x):        
        
        EV_SOC = [EV_initial/capacity + x[0]*self.chargingEff/capacity if x[0]>=0 else EV_initial/capacity + x[0]/(self.chargingEff*capacity)]
        
        for j in range(len(x)-1):
            
            if x[j+1]>=0:
                EV_SOC.append(EV_SOC[j] + x[j+1]*self.chargingEff/capacity)
            else:
                EV_SOC.append(EV_SOC[j] + x[j+1]/(self.chargingEff*capacity))
                
        return np.array(EV_SOC)-0
    
    
    def get_optimized_slots(self, GD, initial, energyPerSlot, capacity, hour:int, iteration:int, store_soc:float = 0, time_in = None, time_out = None, ev = True, current_state = [0]*24):
        
        scaled_state = [x * 100 for x in current_state]

        # Check if any value in scaled_state exceeds 14
        if any(value > 15 for value in scaled_state):
            print(scaled_state)
            
        def ob_func(x):
            
            stress = self.cal_stress(GD + x)
            cost = self.cal_cost(GD, x, hour)
            carbon = self.cal_carbon(GD + x, hour)
                        
            norm_stress = self.minmax_scaling(stress, np.max(self.cal_stress(GD)))
            norm_cost = self.minmax_scaling(cost, np.max(self.cal_cost(GD, [0]*len(GD), hour)))
            norm_carbon = self.minmax_scaling(carbon, np.max(self.cal_carbon(GD, hour)))

            return self.gamma_peak * np.average(norm_stress) + self.gamma_cost * np.average(norm_cost) + self.gamma_carbon * np.average(norm_carbon)
             
        constraints = []
        
        for j in range(len(GD)):
            
            def con2(x, j=j):
                SOC = self.get_SOC(initial ,capacity, x)
                if j == len(GD)-1:
                    return SOC[j] - max(self.lowest,store_soc)
                else:
                    return SOC[j]- self.lowest
            constraints.append({'type': 'ineq', 'fun': con2})
            
            def con3(x, j=j):
                SOC = self.get_SOC(initial, capacity, x)
                return self.highest - SOC[j]
            constraints.append({'type': 'ineq', 'fun': con3})
    
        bound = []
        for i in range(len(GD)): 
            if time_in <= i and i < time_out:
                bound.append(((-14/100 - current_state[i]), (14/100 - current_state[i]))) ##Change to scaling afterwards
            else:
                bound.append((0,0))
                
        x0 = [ energyPerSlot if time_in <= i and i < time_out else 0 for i in range(len(GD))]
            
        minimizer_kwargs = {"method":"SLSQP", "constraints":constraints,"bounds":bound, "options":{'eps': 1e-10, 'ftol': 1e-3}}
                
        ret = basinhopping(ob_func, x0, minimizer_kwargs=minimizer_kwargs,niter= iteration, seed=0)
        
        return np.array(ret.x), float(ret.fun)

    def optimize_ev(self, days: int, length: int, iteration: int, scaling_factor=100):
        projPath = './results/' + projName
        grid_demand_pv = [D_B_t[i] - S_R_t[i] for i in range(24 * days)]
        batt_charging = np.zeros(24 * days)
        
        hours = 24 * days
        batt_now = pd.DataFrame().reindex_like(self.BattInfo).dropna()
        
        rolling_win = range(0,length)
        
        new_batt = self.BattInfo.loc[self.BattInfo['Arrival_hour'].isin(rolling_win)]
        batt_now = pd.concat([batt_now, new_batt])
    
        plt.figure()

        battery_indi = [[0]*24*days]*len(self.BattInfo)  
        
        for hour in tqdm(range(hours), ncols=150, desc=f'Calculating V2B Charging Slots'):
            # print("hour",hour)
            np.save('temp_evcharging.npy',batt_charging)

            if a_vt[i][hour]==0:
                print()
                continue
            
            if not batt_now.empty:
                                        
                batt_now['Required Energy(kWh)'] = (batt_now['Departure_SOC']-batt_now['Arrival_SOC'])*1.5
                batt_now['Hours_in_lot'] = -batt_now['Arrival_hour'] + batt_now['Departure_hour']
                batt_now['energy_per_slot'] = batt_now['Required Energy(kWh)']/batt_now['Hours_in_lot']

            rolling_win = range(hour, hour+length)
            
            new_batt = self.BattInfo.loc[self.BattInfo['Arrival_hour'] == rolling_win[-1]]
            batt_now = pd.concat([batt_now, new_batt])
            
            leaving_vehicles = batt_now[batt_now['Departure_hour'] == (hour-1)].index
            batt_now.drop(leaving_vehicles, inplace=True)
            
            batt_now = batt_now.reset_index(drop=True)
            sorted_batt = batt_now.sort_values(by=['energy_per_slot', 'Hours_in_lot'], ascending=[False, True])

            if sorted_batt.empty:
                print("No Batteries")
                continue
            
            if (rolling_win[0] not in batt_now['Arrival_hour'].tolist()) and (hour < hours-length):
                print("Skipping Battery: Not in first hour")
                continue
            
            if hour <= hours-length:
                
                updated_grid_demand = np.array(grid_demand_pv[hour:hour+length].copy())/scaling_factor
                num = 0
                numbers = np.where(np.array(sorted_batt['Arrival_hour'].tolist()) == rolling_win[0])
                
                for index, row in sorted_batt.iterrows():
                    
                    # print("batt",index)

                    if rolling_win[0] == row['Arrival_hour']:
                        batt_now.loc[index, 'Arrival_hour'] += 1
                    
                    if (not len(numbers[0]) == 0 ) and (hour != hours-length):
                        if num == numbers[0][-1]:
                            break
                                        
                    # ev_required = row['Required Energy(kWh)']
                    ev_capacity = 1.5 / scaling_factor
                    ev_initial = row['Arrival_SOC'] * ev_capacity
                    ev_desired = row['Departure_SOC'] * ev_capacity
                    ev_initial_soc = row['Arrival_SOC']
                    ev_desired_soc = row['Departure_SOC']
                    
                    slot_in = rolling_win.index(row['Arrival_hour'])
                    
                    if row['Departure_hour'] not in rolling_win:
                        slot_out = length-1
                    else:
                        slot_out = rolling_win.index(row['Departure_hour'])
                    
                    if row['energy_per_slot']/self.chargingEff >= self.chargingRate:
                        # print('skip')
                        results = np.array([self.chargingRate if slot_in <= i and i < slot_out else 0 for i in range(length)])/scaling_factor
                    else:
                        results, _ = self.get_optimized_slots(np.array(updated_grid_demand), ev_initial, row['energy_per_slot']/scaling_factor, ev_capacity, hour, iteration, store_soc = ev_desired_soc, time_in=slot_in, time_out=slot_out, ev=True, current_state=batt_charging[hour:hour+ length])


                    if hour == hours-length:
                        batt_charging[hour:] += np.array(results)
                        battery_indi[index][hour:] = results
                    else:
                        batt_charging[hour] += np.array(results[0])
                        battery_indi[index][hour] = results[0]
                        
                    updated_grid_demand += np.array(results)
                    
                    batt_now.loc[index, 'Arrival_SOC'] = ev_initial_soc + (results[0]/ev_capacity)
                        
                    num += 1
                    # print(results * scaling_factor)
                    # print(batt_charging * scaling_factor, '\n\n')

                battery_indi_df = pd.DataFrame(battery_indi)
                battery_indi_df.to_csv(projPath + '/checkpoint/battery'+str(hour)+'_'+str(length)+'.csv')
                print('\n',"*"*50, f'\nHour {hour}\n {batt_charging * scaling_factor}')
                
                np.save(projPath + '/checkpoint/EVchargingV2B'+str(hour)+'_'+str(length)+'.npy', batt_charging * scaling_factor)
        
            else:
                break
        
        
        
        # print("Results for V2B: ", ev_charging[(day * 24):(day * 24) + 24])
        # np.save(projPath + f'/npyFiles/EVchargingV2B{date_strings[day]}.npy', (ev_charging * scaling_factor))

        # # Save battery_indi as CSV
        # battery_indi_df = pd.DataFrame(battery_indi)
        # # battery_indi_df.to_csv(projPath + f'/csv/BatteryIndi{date_strings[day]}.csv', index=False)

        return batt_charging * scaling_factor

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


# if os.path.isfile(projPath + '/npyFiles/EVchargingV2Bnew.npy'):
#     EVChargingV2B = np.load(projPath + '/npyFiles/EVchargingV2Bnew.npy')
#     print('File found: Loaded V2B Charging/Discharging Results from ./npyFiles\n')
    
# else:
#     EVChargingV2B = V2B.optimize_ev(days = days, length = 24, iteration = iteration)
#     np.save(projPath + '/npyFiles/EVchargingV2Bnew.npy', EVChargingV2B)
#     print('Saving V2B Charging/Discharging Results to ./npyFiles\n')
    
if os.path.isfile(projPath + '/npyFiles/EVchargingV2Bnewnew_12.npy'):
    EVChargingV2B = np.load(projPath + '/npyFiles/EVchargingV2Bnewnew_12.npy')
    print('File found: Loaded V2B Charging_12/Discharging Results from ./npyFiles\n')
    
else:
    EVChargingV2B = V2B.optimize_ev(days = days, length = 12, iteration = iteration)
    np.save(projPath + '/npyFiles/EVchargingV2Bnewnew_12.npy', EVChargingV2B)
    print('Saving V2B Charging_12/Discharging Results to ./npyFiles\n')
    
# if os.path.isfile(projPath + '/npyFiles/EVchargingV2Bnew_6.npy'):
#     EVChargingV2B = np.load(projPath + '/npyFiles/EVchargingV2Bnew_6.npy')
#     print('File found: Loaded V2B Charging_6/Discharging Results from ./npyFiles\n')
    
# else:
#     EVChargingV2B = V2B.optimize_ev(days = days, length = 6, iteration = iteration)
#     np.save(projPath + '/npyFiles/EVchargingV2Bnew_6.npy', EVChargingV2B)
#     print('Saving V2B Charging_6/Discharging Results to ./npyFiles\n')
    
    