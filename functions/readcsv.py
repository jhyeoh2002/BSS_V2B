import pandas as pd 
import numpy as np

def readData() -> tuple[np.array, np.array, np.array]:
    '''
    Read data from .csv files and return year data 2020 in array format
    
    Parameters  
    None
    
    Returns
    year data(tuple): Building Energy Consumption, PV Generation, Carbon Intensity 
    
    '''

    BldgEnCon = pd.read_csv(r"Data/Cleaned data/Building Energy Consumption.csv").set_index('Datetime')
    BldgEnCon = np.array(BldgEnCon.iloc[:,4])

    PVGen = pd.read_csv(r"Data/Cleaned data/PV generation.csv").set_index('Datetime')
    PVGen = np.array(PVGen.iloc[:,4])

    CarbInt = pd.read_csv(r"Data/Cleaned data/Carbon Intensity Data 2020.csv").set_index('Datetime')
    CarbInt = np.array(CarbInt.iloc[:,0])

    return BldgEnCon, PVGen, CarbInt