import matplotlib.pyplot as plt
import datetime
import numpy as np
import matplotlib.dates as mdates
from functions.readcsv import readData
import pandas as pd
from colour import Color
from matplotlib.colors import LinearSegmentedColormap

def exemplaryweek(projPath:str, days = 366, start_day = 1, days_to_show = 7, 
                  title_fontsize = 12, axis_label_fontsize = 9, tick_label_fontsize = 8, legend_fontsize = 10, linewidth = 0.7, figsize=(8, 7), dpi=300, 
                  BC = '#AEC6CF', EV = '#CDEAC0', PV2EV = '#CC9900', PV2BC= '#CC5500', EV2BC = '#004d00', ESS2BC = '#330033', ESS = '#EAD1DC', Line = 'k'):

        colors = {'BC': BC, 'EV': EV , 'PV2EV': PV2EV, 'PV2BC': PV2BC, 'EV2BC': EV2BC, 'ESS2BC': ESS2BC, 'ESS': ESS, 'line':Line}

        BldgEnCon, PVGen, CarbInt = readData()    #Read data into array

        EVChargingImmediate = np.load(projPath + '/npyFiles/EVChargingImmediate.npy')
        EVChargingV2B = np.load(projPath + '/npyFiles/EVchargingV2B.npy')
        ESSCharging = np.load(projPath + '/npyFiles/ESScharging.npy')

        date_range = pd.date_range(start='2020-01-01', end='2021-1-1 00:00', freq='h')

        GD_init = BldgEnCon[:days*24] + EVChargingImmediate[:days*24]
        GD_S1 = BldgEnCon[:days*24] - PVGen[:days*24] + EVChargingImmediate[:days*24]
        GD_S2 = BldgEnCon[:days*24] - PVGen[:days*24] + EVChargingV2B  
        GD_S3 = BldgEnCon[:days*24] - PVGen[:days*24] + EVChargingV2B + ESSCharging

        GD_S1_pos = np.maximum(GD_S1, 0)
        GD_S2_pos = np.maximum(GD_S2, 0)
        GD_S3_pos = np.maximum(GD_S3, 0)

        # EV charging with positive and negative split
        EV_charging_pos = np.maximum(EVChargingV2B, 0)
        EV_charging_neg = np.minimum(EVChargingV2B, 0)

        # ESS charging with positive and negative split
        ESS_charge_pos = np.maximum(ESSCharging, 0)
        ESS_charge_neg = np.minimum(ESSCharging, 0)

        S1PV2EVImm = np.minimum(PVGen[:days*24],EVChargingImmediate[:days*24])
        S1BC2EV = EVChargingImmediate[:days*24] - S1PV2EVImm    

        S2PV2V2B = np.minimum(PVGen[:days*24],EV_charging_pos[:days*24])
        S2PV2BC = np.maximum(PVGen[:days*24]-S2PV2V2B[:days*24],0)

        S3BC2V2B = EV_charging_pos[:days*24] - S2PV2V2B[:days*24] - ESS_charge_neg[:days*24]

        hours_to_show = 24 * (start_day + days_to_show-1) + 1

        # Date range for axis
        date_start = datetime.date(2020, 1, start_day)
        date_end = datetime.date(2020, 1, start_day + days_to_show)

        # Set up figure    
        fig, axs = plt.subplots(2, 2, figsize=figsize, dpi=dpi)

        ####################### Subplot 1: Scenario 0 #######################

        plt.subplot(2, 2, 1)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))  # Formats the x-axis labels to month-day format
        plt.grid(True, alpha=0.2)

        plt.fill_between(date_range[:hours_to_show], 
                        0, 
                        BldgEnCon[:hours_to_show], 
                        step='post', color = colors['BC'], label = 'Building Consumption', linewidth = 0)

        plt.fill_between(date_range[:hours_to_show], 
                        BldgEnCon[:hours_to_show] + EVChargingImmediate[:hours_to_show], 
                        BldgEnCon[:hours_to_show], 
                        step='post', color = colors['EV'], label = 'EVs Charging', linewidth = 0)

        plt.step(date_range[:hours_to_show], 
                GD_init[:hours_to_show], 
                where='post', linewidth=linewidth, color = 'k', label = None)

        plt.axis([date_start, date_end, 0, 650])
        plt.title('Scenario 0', fontsize=title_fontsize)
        plt.xlabel('Datetime (hr)', fontsize=axis_label_fontsize)
        plt.ylabel('Grid Demand (kW)', fontsize=axis_label_fontsize)
        plt.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)  # Sets the font size for both x and y ticks

        ####################### Subplot 2: Scenario 1 #######################

        plt.subplot(2, 2, 2)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))  # Formats the x-axis labels to month-day format
        plt.grid(True, alpha=0.2)

        plt.fill_between(date_range[:hours_to_show], 
                        GD_init[:hours_to_show] - S1PV2EVImm[:hours_to_show], 
                        GD_init[:hours_to_show], 
                        step='post', color=colors['PV2EV'], alpha = 0.3, hatch='///', label="PV to EVs", linewidth = 0)

        plt.fill_between(date_range[:hours_to_show], 
                        GD_S1_pos[:hours_to_show], 
                        GD_init[:hours_to_show] - S1PV2EVImm[:hours_to_show], 
                        step='post', color=colors['PV2BC'], alpha = 0.3, hatch='///', label="PV to Buildings", linewidth = 0)

        plt.fill_between(date_range[:hours_to_show], 
                        GD_S1_pos[:hours_to_show], 
                        GD_S1_pos[:hours_to_show] - S1BC2EV[:hours_to_show], 
                        step='post', color = colors['EV'], label = None, linewidth = 0)

        plt.fill_between(date_range[:hours_to_show], 
                        0, 
                        GD_S1_pos[:hours_to_show] - S1BC2EV[:hours_to_show], 
                        step='post', color = colors['BC'], label = None, linewidth = 0)

        plt.step(date_range[:hours_to_show], 
                GD_S1_pos[:hours_to_show], 
                where='post', linewidth = linewidth, color = colors['line'], label=None)

        plt.axis([date_start, date_end, 0, 650])
        plt.title('Scenario 1', fontsize=title_fontsize)
        plt.xlabel('Datetime (hr)', fontsize=axis_label_fontsize)
        plt.ylabel('Grid Demand (kW)', fontsize=axis_label_fontsize)
        plt.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)  # Sets the font size for both x and y ticks
        # plt.legend(fontsize=legend_fontsize, loc='upper center')

        ####################### Subplot 3: Scenario 2 #######################

        plt.subplot(2, 2, 3)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))  # Formats the x-axis labels to month-day format
        plt.grid(True, alpha=0.2)

        plt.fill_between(date_range[:hours_to_show], 
                        BldgEnCon[:hours_to_show] + EV_charging_pos[:hours_to_show] - S2PV2V2B[:hours_to_show], 
                        BldgEnCon[:hours_to_show] + EV_charging_pos[:hours_to_show], 
                        step='post', alpha=0.3, color= colors['PV2EV'], hatch='///', label=None, linewidth = 0)

        plt.fill_between(date_range[:hours_to_show], 
                        BldgEnCon[:hours_to_show] + EV_charging_pos[:hours_to_show] - S2PV2V2B[:hours_to_show], 
                        np.maximum(GD_S2_pos[:hours_to_show],BldgEnCon[:hours_to_show] - PVGen[:hours_to_show]), 
                        step='post', alpha=0.3, color=colors['PV2BC'], hatch='///', label=None, linewidth = 0)

        plt.fill_between(date_range[:hours_to_show], 
                        GD_S2_pos[:hours_to_show], 
                        GD_S2_pos[:hours_to_show] - EV_charging_neg[:hours_to_show], 
                        step='post', alpha=0.3, color=colors['EV2BC'], hatch='///', label='EVs Discharging', linewidth = 0)

        plt.fill_between(date_range[:hours_to_show], 
                        0, 
                        np.minimum(GD_S2_pos[:hours_to_show], BldgEnCon[:hours_to_show]),
                        step='post', color=colors['BC'], linewidth = 0)

        plt.fill_between(date_range[:hours_to_show], 
                        GD_S2_pos[:hours_to_show],
                        np.minimum(GD_S2_pos[:hours_to_show], BldgEnCon[:hours_to_show]),
                        step='post', color=colors['EV'], linewidth = 0)



        plt.step(date_range[:hours_to_show], 
                GD_S2[:hours_to_show], 
                where='post', linewidth=linewidth,color=colors['line'], label=None)

        plt.axis([date_start, date_end, 0, 650])
        plt.title('Scenario 2', fontsize=title_fontsize)
        plt.xlabel('Datetime (hr)', fontsize=axis_label_fontsize)
        plt.ylabel('Grid Demand (kW)', fontsize=axis_label_fontsize)
        plt.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)  # Sets the font size for both x and y ticks
        # plt.legend(fontsize=legend_fontsize, loc='upper left')

        ####################### Subplot 4: Scenario 3 #######################

        plt.subplot(2, 2, 4)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))  # Formats the x-axis labels to month-day format
        plt.grid(True, alpha=0.2)

        plt.fill_between(date_range[:hours_to_show], 
                        BldgEnCon[:hours_to_show] + EV_charging_pos[:hours_to_show] - S2PV2V2B[:hours_to_show], 
                        BldgEnCon[:hours_to_show] + EV_charging_pos[:hours_to_show], 
                        step='post', alpha=0.3, color= colors['PV2EV'], hatch='///', label=None, linewidth = 0)

        plt.fill_between(date_range[:hours_to_show], 
                        BldgEnCon[:hours_to_show] + EV_charging_pos[:hours_to_show] - S2PV2V2B[:hours_to_show], 
                        np.maximum(BldgEnCon[:hours_to_show] + EV_charging_pos[:hours_to_show] - S2PV2V2B[:hours_to_show] - S2PV2BC[:hours_to_show], GD_S3_pos[:hours_to_show]),
                        step='post', alpha=0.3, color=colors['PV2BC'], hatch='///', label=None, linewidth = 0)

        plt.fill_between(date_range[:hours_to_show], 
                        GD_S3_pos[:hours_to_show], 
                        GD_S3_pos[:hours_to_show] - ESS_charge_neg[:hours_to_show],
                        step='post', alpha=0.3, color=colors['ESS2BC'], hatch='///', label='ESS Discharging', linewidth = 0)

        plt.fill_between(date_range[:hours_to_show], 
                        np.maximum(GD_S2_pos[:hours_to_show],GD_S3_pos[:hours_to_show]), 
                        np.maximum(GD_S2_pos[:hours_to_show] - EV_charging_neg[:hours_to_show], GD_S3_pos[:hours_to_show]), 
                        step='post', alpha=0.3, color=colors['EV2BC'], hatch='///', label=None, linewidth = 0)

        plt.fill_between(date_range[:hours_to_show], 
                        0, 
                        np.minimum(GD_S3_pos[:hours_to_show], BldgEnCon[:hours_to_show]),
                        step='post', color=colors['BC'], linewidth = 0)

        plt.fill_between(date_range[:hours_to_show], 
                        GD_S3_pos[:hours_to_show],
                        np.minimum(GD_S3_pos[:hours_to_show], BldgEnCon[:hours_to_show]),
                        step='post', color=colors['EV'], linewidth = 0)

        plt.fill_between(date_range[:hours_to_show], 
                        GD_S3_pos[:hours_to_show],
                        GD_S3_pos[:hours_to_show] - ESS_charge_pos[:hours_to_show],
                        step='post', color=colors['ESS'],label='ESS Charging', linewidth = 0)

        plt.step(date_range[:hours_to_show], 
                GD_S3_pos[:hours_to_show], 
                where='post',color = colors['line'], linewidth=linewidth, label="Grid Demand")

        plt.axis([date_start, date_end, 0, 650])
        plt.title('Scenario 3', fontsize=title_fontsize)
        plt.xlabel('Datetime (hr)\n\n\n\n', fontsize=axis_label_fontsize)
        plt.ylabel('Grid Demand (kW)', fontsize=axis_label_fontsize)
        plt.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)  # Sets the font size for both x and y ticks

        fig.legend(fontsize=legend_fontsize, loc='lower center', ncol = 4)
        plt.tight_layout()
        plt.savefig(projPath + '/figures/exemplaryweek.png', format='png',bbox_inches = 'tight')

        pass

def heatmap(projPath:str, days = 366,  figsize=(10,4), dpi=300, colors = ['#7FFF8C','#FFFF99','#FF8080']):
        
        BldgEnCon, PVGen, CarbInt = readData()    #Read data into array

        EVChargingImmediate = np.load(projPath + '/npyFiles/EVChargingImmediate.npy')
        EVChargingV2B = np.load(projPath + '/npyFiles/EVchargingV2B.npy')
        ESSCharging = np.load(projPath + '/npyFiles/ESScharging.npy')

        date_range = pd.date_range(start='2020-01-01', end='2021-1-1 00:00', freq='h')

        GD_init = BldgEnCon[:days*24] + EVChargingImmediate[:days*24]
        GD_S1 = BldgEnCon[:days*24] - PVGen[:days*24] + EVChargingImmediate[:days*24]
        GD_S2 = BldgEnCon[:days*24] - PVGen[:days*24] + EVChargingV2B  
        GD_S3 = BldgEnCon[:days*24] - PVGen[:days*24] + EVChargingV2B + ESSCharging
        
        start_date = "2020-01-01 00:00:00"
        end_date = "2020-12-31 23:00:00"
        date_range = pd.date_range(start=start_date, end=end_date, freq='h')

        wastage = pd.DataFrame({'datetime': date_range,
                'GD_S1_wastage': np.minimum(GD_S1,0),
                'GD_S2_wastage': np.minimum(GD_S2,0),
                'GD_S3_wastage': np.minimum(GD_S3,0)})

        wastage.set_index('datetime', inplace=True)

        # Group DataFrame by month and sum the values
        monthly_sum = wastage.groupby(pd.Grouper(freq='M')).sum()

        wastage_nparr = np.array([-monthly_sum['GD_S1_wastage'],-monthly_sum['GD_S2_wastage'],-monthly_sum['GD_S3_wastage']  ])
                
        fig, ax = plt.subplots(figsize = figsize, dpi = dpi)
        custom_ramp = make_Ramp(colors) 
        im = ax.imshow(wastage_nparr, cmap=custom_ramp)

        months = np.arange(1,13,1)
        scenarios = [1,2,3]

        cbar = ax.figure.colorbar(im, ax=ax, cmap=custom_ramp, orientation='horizontal', fraction=0.08, aspect = 40)
        # cbar.ax.set_ylabel('\nExcess RES from PV GEneration', rotation=90)
        
        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(len(months)), labels=months)
        ax.set_yticks(np.arange(len(scenarios)), labels=scenarios)

        for i in range(len(scenarios)):
                for j in range(len(months)):
                        text = ax.text(j, i, wastage_nparr[i, j].astype(int),
                                ha="center", va="center", color="k")
                
        ax.set_ylabel('Scenarios')
        ax.set_xlabel('Months\n')
        plt.title('Monthy Excess PV Generation across Different Scenarios (kWh)\n')
        plt.savefig(projPath + '/figures/excess_energy_heatmap.png', format='png',bbox_inches = 'tight')
        
        pass

def make_Ramp(ramp_colors): 

        color_ramp = LinearSegmentedColormap.from_list( 'my_list', [ Color( c1 ).rgb for c1 in ramp_colors ] )
        return color_ramp


