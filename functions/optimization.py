import numpy as np
import pandas as pd
from scipy.optimize import minimize, basinhopping
from functions.immediatecharging import slotcal
import timeit
from tqdm import tqdm
import matplotlib.pyplot as plt

class Optimization: 
    
    BldgEnCon = pd.read_csv(r"./Data/Cleaned data/Building Energy Consumption.csv").set_index('Datetime')
    BldgEnCon.index = pd.to_datetime(BldgEnCon.index) 

    PVGen = pd.read_csv(r"./Data/Cleaned data/PV generation.csv").set_index('Datetime')
    PVGen.index = pd.to_datetime(PVGen.index) 

    CarbInt = pd.read_csv(r"./Data/Cleaned data/Carbon Intensity Data 2020.csv").set_index('Datetime')
    CarbInt.index = pd.to_datetime(CarbInt.index) 
    
    ElecRate = pd.read_csv(r"./Data/Cleaned data/ElectricityRate2020.csv").set_index('Datetime')
    ElecRate.index = pd.to_datetime(ElecRate.index) 
    
    def __init__(self, chargingRate, chargingEff, gammaPeak, gammaCost, gammaCarbon, lowestSoC, highestSoC, datestring, ESSCapacity = None):
        
        self.gamma_peak = gammaPeak
        self.gamma_cost = gammaCost
        self.gamma_carbon = gammaCarbon
        self.chargingRate = chargingRate
        self.chargingEff = chargingEff
        self.lowest = lowestSoC
        self.highest = highestSoC
        self.ESSCapacity = ESSCapacity
        self.ParkPatt = pd.read_csv(r"./Data/Cleaned data/EV Parking Pattern.csv")
        self.ParkPatt[['Required Energy(kWh)', 'No. of Slots']] = self.ParkPatt.apply(lambda row: pd.Series(slotcal(row, chargingRate)), axis=1)
        self.ParkPatt['Hours_in_lot']=-self.ParkPatt['Time_in'] + self.ParkPatt['Time_out']
        self.ParkPatt['energy_per_slot']=self.ParkPatt['Required Energy(kWh)']/self.ParkPatt['Hours_in_lot']
        self.ParkPatt['Datetime_in']= self.ParkPatt['Date'] + ' ' + self.ParkPatt['Time_in'].apply(lambda x: format(int(x),"02d")).astype(str)
        self.ParkPatt['Datetime_out']= self.ParkPatt['Date'] + ' ' + self.ParkPatt['Time_out'].apply(lambda x: format(int(x),"02d")).astype(str)
        self.date_string = datestring

    def cal_stress(self, GD) -> np.ndarray:        
        return np.array([(x ** 2) if x > 0 else 0 for x in GD])

    def cal_carbon(self, GD, hour) -> np.ndarray:   
        return np.array([(x * self.CarbInt.iloc[hour + i,0]) if x > 0 else 0 for i, x in enumerate(GD)])

    def cal_cost(self, GD, hour) -> np.ndarray:   
        return np.array([(x * self.ElecRate.iloc[hour + i,1]) if x > 0 else 0 for i, x in enumerate(GD)])
               
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
                
        return np.array(EV_SOC)
    
    def get_optimized_slots(self, GD, initial, energyPerSlot, capacity, hour:int, iteration:int, store_soc:float = 0, time_in = None, time_out = None, ev = True):
            
        def ob_func(x):
            
            stress = self.cal_stress(GD + x)
            cost = self.cal_cost(GD + x, hour)
            carbon = self.cal_carbon(GD + x, hour)
                        
            norm_stress = self.minmax_scaling(stress, np.max(self.cal_stress(GD)))
            norm_cost = self.minmax_scaling(cost, np.max(self.cal_cost(GD, hour)))
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
                bound.append((-self.chargingRate/100, self.chargingRate/100)) ##Change to scaling afterwards
            else:
                bound.append((0,0))
                
        x0 = [ energyPerSlot if time_in <= i and i < time_out else 0 for i in range(len(GD))]

        minimizer_kwargs = {"method":"SLSQP", "constraints":constraints,"bounds":bound, "options":{'eps': 1e-10, 'ftol': 1e-9}}
        
        ret = basinhopping(ob_func, x0, minimizer_kwargs=minimizer_kwargs,niter= iteration, seed=0)
        
        return np.array(ret.x), float(ret.fun)

    def optimize_ev(self, days: int, length: int, iteration: int, scaling_factor = 100):

        grid_demand_pv = [self.BldgEnCon.iloc[i, 4] - self.PVGen.iloc[i, 4] for i in range(24 * days)]
        ev_charging = np.zeros(24 * days)
        
        hours = days*24
        
        vehicles_now = pd.DataFrame().reindex_like(self.ParkPatt).dropna()
        
        rolling_win = self.date_string[0:length]
        
        new_vehicles = self.ParkPatt.loc[self.ParkPatt['Datetime_in'].isin(rolling_win)]
        vehicles_now = pd.concat([vehicles_now, new_vehicles])

        plt.figure()

        for hour in tqdm(range(hours), ncols=150, desc = f'Calculating V2B Charging Slots'):
                        
            np.save('temp_evcharging.npy',ev_charging)
            
            if not vehicles_now.empty:
                                        
                vehicles_now[['Required Energy(kWh)', 'No. of Slots']] = vehicles_now.apply(lambda row: pd.Series(slotcal(row, self.chargingRate)), axis=1)
                vehicles_now['Hours_in_lot'] = -vehicles_now['Time_in'] + vehicles_now['Time_out']
                vehicles_now['energy_per_slot'] = vehicles_now['Required Energy(kWh)']/vehicles_now['Hours_in_lot']
                vehicles_now['Datetime_in'] = vehicles_now['Date'] + ' ' + vehicles_now['Time_in'].apply(lambda x: format(int(x),"02d")).astype(str)
                vehicles_now['Datetime_out'] = vehicles_now['Date'] + ' ' + vehicles_now['Time_out'].apply(lambda x: format(int(x),"02d")).astype(str)
            
            rolling_win = self.date_string[hour:hour+length]
            
            new_vehicles = self.ParkPatt.loc[self.ParkPatt['Datetime_in'] == rolling_win[-1]]
            vehicles_now = pd.concat([vehicles_now, new_vehicles])
            
            leaving_vehicles = vehicles_now[vehicles_now['Datetime_out'] == rolling_win[0]].index
            vehicles_now.drop(leaving_vehicles, inplace=True)
            
            vehicles_now = vehicles_now.reset_index(drop=True)
            sorted_vehicles = vehicles_now.sort_values(by=['energy_per_slot', 'Hours_in_lot'], ascending=[False, True])

            if sorted_vehicles.empty:
                print("No Vehicles")
                continue
                
            if (rolling_win[0] not in vehicles_now['Datetime_in'].tolist()) and (hour < hours-length):
                continue
            
            if hour <= hours-length:
                
                updated_grid_demand = np.array(grid_demand_pv[hour:hour+length].copy())/scaling_factor
                num = 0
                numbers = np.where(np.array(sorted_vehicles['Datetime_in'].tolist()) == rolling_win[0])
                
                for index, row in sorted_vehicles.iterrows():
                    
                    if rolling_win[0] == row['Datetime_in']:
                        vehicles_now.loc[index, 'Time_in'] += 1
                        vehicles_now.loc[index, 'Datetime_in'] = rolling_win[1]
                    
                    if (not len(numbers[0]) == 0 ) and (hour != hours-length):
                        if num == numbers[0][-1]:
                            break
                                        
                    ev_capacity = row['Battery Capacity (kWh)']/scaling_factor
                    ev_initial = row['SOC_initial'] * ev_capacity
                    ev_desired = row['SOC_desired'] * ev_capacity
                    ev_required = row['Required Energy(kWh)']
                    ev_initial_soc = row['SOC_initial']
                    ev_desired_soc = row['SOC_desired']

                    slot_in = rolling_win.index(row['Datetime_in'])
                    
                    if row['Datetime_out'] not in rolling_win:
                        slot_out = length-1
                    else:
                        slot_out = rolling_win.index(row['Datetime_out'])
                    
                    if row['energy_per_slot']/self.chargingEff >= self.chargingRate:
                        # print('skip')
                        results = np.array([self.chargingRate if slot_in <= i and i < slot_out else 0 for i in range(length)])/scaling_factor
                    else:
                        results, func_score = self.get_optimized_slots(np.array(updated_grid_demand), ev_initial, row['energy_per_slot']/scaling_factor, ev_capacity, hour, iteration, store_soc = ev_desired_soc, time_in=slot_in, time_out=slot_out, ev=True)

                    if hour == hours-length:
                        ev_charging[hour:] += np.array(results)
                    else:
                        ev_charging[hour] += np.array(results[0])
                        
                    updated_grid_demand += np.array(results)
                    
                    vehicles_now.loc[index, 'SOC_initial'] = ev_initial_soc + (results[0]/ev_capacity)
                        
                    num += 1
                    
                print("*"*50, f'\nHour {hour}\n {ev_charging * scaling_factor}')

            else:
                break
                
        return ev_charging * scaling_factor

