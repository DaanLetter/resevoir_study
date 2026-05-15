# Function to Model Reservoirs based on Turner et al., 2021
# JCS
# 8-2-2023

# TODO
# ADD in water balance outlfow == inflow
# push it by adding big flood and big demand  (a point value)
# half year of inflow
# add in multiple years of data and mess with drought periods and flood periods


# check with other points 
# TODO ADD IN LONGTERM AVERAGE INFLOW

import logging
import os

import xarray as xr
import numpy as np
import pcraster as pcr
import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# create 2x2 grid of all the variables in pcraster maps and then debug the pcraster :)
# niko = np.array(inflow_dam['soswater_inflow'])[0]
# jen = pcr.numpy2pcr(pcr.Scalar, niko.repeat(4).reshape(2,2), -99)
def TurnerOutflow(inflow_val, avg_inflow, env_flow, dem_val, previous_storage, week, date_string, hydropower): # will have 
    # get all the inputs for flood and conservation
    cap_215 = 23.5*1e6
    date_val = date_string.day
    month_val = date_string.month
    #print(week)
    new_week_val = week -1
    
    logger.debug("week: %s", new_week_val)
    date_string = str('2000')+ "-" + str(month_val) +"-" + str(date_val)
    _rf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data', 'POINTDATA', '10_param_RF_bounds_final')
    variables = xr.open_dataset(os.path.join(_rf_dir, date_string + '.nc')).sel(latitude = latitude, longitude = longitude, method = 'nearest').to_dataframe().reset_index()
    flood = int(variables['flood'].iloc[0]/100*cap_215)
    logger.debug("flood: %s", flood)
    conservation = int(variables['conservation'].iloc[0]/100*cap_215)
    logger.debug("conservation: %s", conservation)
    
    # constants and calculated values
    bankfull = 2.3
    current_storage = previous_storage + max(inflow_val, 0)# ensures inflow is not negative
    reduction_factor = max((current_storage -conservation)/(flood - conservation)* bankfull, 0)
    demand_reduction_factor = (current_storage - 0.1*cap_215)/(conservation- 0.1*cap_215)
    
    release = reduction_factor * avg_inflow      # unit: m3             
    ## FLOOD
    if reduction_factor > bankfull and current_storage > flood: # flood conditions
        reduction_factor = bankfull
    #release = reduction_factor*avg_inflow
    if current_storage - release > cap_215:
        flood_difference = current_storage - release - cap_215
        release = release + flood_difference
        
    ## ACTIVE STORAGE FOR NON HYDROPOWER
    if release < dem_val and hydropower ==0: # active storage
        #release = dem_val*reduction_factor/bankfull
        release = dem_val * demand_reduction_factor
        logger.debug("non hydropower active storage")
    ## ACTIVE STORAGE FOR HYDROPOWER
    if release < dem_val and hydropower ==1:
        release = dem_val * reduction_factor/bankfull
        if current_storage - release < conservation:
            release = current_storage - conservation # check to make sure it stays at conservation
        logger.debug("hydropower active storage")
     #### NON NEGATIVE STORAGE CHECK
    test_storage_non_neg= current_storage - release
    if test_storage_non_neg <0:
        release = release/bankfull * (1-flood/cap_215)
        
    ### ENV FLOW CHECK
    test_storage = current_storage - env_flow
    #release = release + env_flow
    if release < env_flow and test_storage >0: # min_storage/conservation, double check this! 
        release = env_flow
    ## CATCH FOR DEAD STORAGE
    if current_storage < 0.1*cap_215:
        release = 0  
    ## CHECK STORAGE 
    new_storage = current_storage - release
    logger.debug("new_storage: %s", new_storage)
    return(release, flood, conservation, new_storage, reduction_factor)
## WILL HAVE TO CONVERT TO PCRRASTER MAPS.....



# PCRGLOBWBW FUNCTION

def getReservoirOutflow_Turner(self,avgChannelDischarge,length_of_time_step,downstreamDemand,\
                            irrGrossDemand, nonIrrGrossDemand, environmental_flow):
    #import xarray as xr 
    
    ##### GET OUTFLOW OF RESERVOIR TO START #####
    # avgOutflow (m3/s)
    avgOutflow = self.avgOutflow
    self.sos_outflow = avgOutflow
    # The following is needed when new lakes/reservoirs introduced (its avgOutflow is still zero).
    #~ # - alternative 1
    #~ avgOutflow = pcr.ifthenelse(\
                 #~ avgOutflow > 0.,\
                 #~ avgOutflow,
                 #~ pcr.max(avgChannelDischarge, self.avgInflow, 0.001))
    # - alternative 2
    avgOutflow = pcr.ifthenelse(\
                 avgOutflow > 0.,\
                 avgOutflow,
                 pcr.max(avgChannelDischarge, self.avgInflow))
    avgOutflow = pcr.ifthenelse(\
                 avgOutflow > 0.,\
                 avgOutflow, pcr.downstream(self.lddMap, avgOutflow))
    avgOutflow = pcr.areamaximum(avgOutflow,self.waterBodyIds)              
    
    
    resvOutflow = (avgOutflow * length_of_time_step) # units = m3
    
    #### CALCULATE THE FLOOD AND CONSERVATION POOLS #####
    flood_read = self.turnerFlood/100 *self.waterBodyCap
    #print('flood max is' + str(np.nanmax(pcr.pcr2numpy(self.turnerFlood, mv = np.nan))))
    conserve_read = self.turnerConservation/100 *self.waterBodyCap
    #print('enter get res and tried to cover')

    flood_final = pcr.min(flood_read, self.waterBodyCap) # fills in missing values in flood with waterbody cap
    conservation_final = pcr.max(conserve_read, self.waterBodyCap*0.1)
    #print(pcr.pcr2numpy(flood_final, mv = -99))
    
    self.turnerFloodF = flood_read
    self.turnerConserveF = conserve_read
    #print('cover worked')\
    
    ### INITIALIZE THE ENV FLOW, INFLOW AND CURRENT STORAGE
    self.env_flow = environmental_flow
    bankfull = 2.3
    
    inflow_res = pcr.ifthenelse(self.inflowInM3PerSec < 0. , pcr.scalar(0), self.inflowInM3PerSec)*86400
    self.sos_inflow = inflow_res
    # TODO IS WATER BODY STORAGE CUBED! CHECK UNITS
    current_storage = self.waterBodyStorage + pcr.max(inflow_res, pcr.scalar(0)) # this gives
    #self.turner_current_stor = current_storage
    ### CALCULATE REDUCTION FACTORS ####
    #max((current_storage -conservation)/(flood - conservation)* bankfull, 0)
    RF_bankfull = ((current_storage-conservation_final)/(flood_final - conservation_final))*pcr.scalar(bankfull)
    reduction_factor = pcr.max(RF_bankfull, pcr.scalar(0))
    self.turnerReduction = reduction_factor
    demand_reduction_factor = (current_storage - 0.1*self.waterBodyCap)/(conservation_final - 0.1*self.waterBodyCap)
    self.reductionFactorDemand = demand_reduction_factor
    
    ###### CALCULATE TOTAL DEMAND ######
    
    # Total demand = irrigation plus non irrigation
    demand_tot = irrGrossDemand + nonIrrGrossDemand # each is in m/day
   
    # demand tot = 0.00609 cell area at the point in question is68010392.0 so the multiplication should be 41445.5328848
    downstreamDemand= demand_tot*self.cellArea # DEMAND IS IN m/day after this step to get to m3/day it needs to be multiplied by cell area
    
    ## THIS IS FOR THE AREA
    #downstreamDemand = pcr.cover(pcr.areatotal(downstreamDemand, pcr.nominal(self.sos_command_area)),pcr.scalar(0)) # this aggregates all the demand over the things with the same number
    #pcr.report(downstreamDemand, '/scratch/steya001/test_demand.map') # writes out the map so I can make sure that it's right
    
    ## THIS IS FOR THE SINGLE POINT 
    downstreamDemand =  pcr.ifthen(pcr.scalar(self.waterBodyTyp)==2.0, downstreamDemand)
    downstreamDemand =  pcr.ifthen(pcr.scalar(self.waterBodyIds) >0., downstreamDemand)
    #downstreamDemand =  pcr.ifthen(pcr.scalar(self.waterBodyIds) >0., downstreamDemand)
   
    max_demand = downstreamDemand
    self.sosdemand =  max_demand
    
    ###### FLOOD VALUES ######
    reduction_factor = pcr.ifthenelse((reduction_factor > bankfull) & (current_storage > flood_final), bankfull, reduction_factor)
    
    flood_difference = current_storage - resvOutflow  - self.waterBodyCap
    resvOutflow  = pcr.ifthenelse((current_storage - resvOutflow ) > self.waterBodyCap, (resvOutflow +flood_difference), resvOutflow )
   
    hydropower_check = pcr.scalar(self.use)
    self.hydropower_check = hydropower_check
    ##### ACTIVE STORAGE FOR NON HYDROPOWER ######
    #resvOutflow  = pcr.ifthenelse((resvOutflow  < max_demand) &(hydropower_check ==pcr.scalar(0)), (max_demand*demand_reduction_factor), resvOutflow )        
    
    ### ACTIVE STORAGE FOR HYDROPOWER #### # TODO CHECK THIS ONE
    resvOutflow_hydropower =  pcr.ifthenelse(((resvOutflow  < max_demand) & (hydropower_check ==pcr.scalar(1))),(max_demand*reduction_factor/bankfull), resvOutflow )
    current_storage_hydropower = current_storage - resvOutflow_hydropower
    #hydropower_mask = pcr.ifthenelse(current_storage_hydropower < conservation_final , pcr.boolean(1), pcr.boolean(0))
    #hydropower_mask2 = pcr.ifthenelse(hydropower_check ==pcr.scalar(1.0), pcr.boolean(1), pcr.boolean(0))
    
    hydropower_release = resvOutflow_hydropower - conservation_final
    
    resvOutflow  = pcr.ifthenelse( (current_storage_hydropower < conservation_final),hydropower_release, resvOutflow )
    
    # TODO CHECK UNITS
    
    #### NON NEGATIVE STORAGE CHECK
    test_storage_non_neg = current_storage - resvOutflow 
    #resvOutflow  = pcr.ifthenelse(test_storage_non_neg <0, (resvOutflow /bankfull * (1-flood_final/self.waterBodyCap)), resvOutflow )
    
    ### ENV FLOW CHECK
    test_env_flow = current_storage - environmental_flow
    resvOutflow = pcr.ifthenelse((resvOutflow < environmental_flow) & (test_env_flow >0), environmental_flow, resvOutflow)
    
    ## CATCH FOR DEAD STORAGE
    resvOutflow = pcr.ifthenelse(current_storage < (0.1*self.waterBodyCap), pcr.scalar(0), resvOutflow) 
    #self.waterBodyStorage = current_storage - resvOutflow
    
    #TODO DOUBLE CHECK THAT WATERBODY STORGE GETS UPDATED 
    
    resvOutflow = pcr.ifthen(pcr.scalar(self.waterBodyIds) > 0., resvOutflow)
    resvOutflow = pcr.ifthen(pcr.scalar(self.waterBodyTyp) == 2, resvOutflow)
    
    
    
    ## CHECK STORAGE 
    #self.sos_outflow = resvOutflow
    self.turner_current_stor = pcr.cover(current_storage, pcr.scalar(0.0)) - pcr.cover(resvOutflow, pcr.scalar(0))
    
    #print('ready to return it all')
    return (resvOutflow) # unit: m3  
# function to convert to pcraster files
def convert2pcr(data, variable):
    new_data = np.array(data[variable])[0]
    final_data = pcr.numpy2pcr(pcr.Scalar, new_data.repeat(4).reshape(2,2), -99)
# PCR GLOBWB ENV FLOW

def estimate_discharge_for_environmental_flow(self, channelStorage): ## USES CHANNEL STORAGE
    # statistical assumptions:
    # - using z_score from the percentile 90
    z_score = 1.2816 
    #~ # - using z_score from the percentile 95
    #~ z_score = 1.645
    
    # long term variance and standard deviation of discharge values
    varDischarge = self.m2tDischarge / \
                   pcr.max(1.,\
                   pcr.min(self.maxTimestepsToAvgDischargeLong, self.timestepsToAvgDischarge)-1.)                             
                   # see: online algorithm on http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    stdDischarge = pcr.max(varDischarge**0.5, 0.0)
    
    # calculate minimum discharge for environmental flow (m3/s)
    minDischargeForEnvironmentalFlow = pcr.max(0.0, self.avgDischarge - z_score * stdDischarge)
    factor = 0.10 # to avoid flip flop
    minDischargeForEnvironmentalFlow = pcr.max(factor*self.avgDischarge, minDischargeForEnvironmentalFlow)   # unit: m3/s
    minDischargeForEnvironmentalFlow = pcr.max(0.0, minDischargeForEnvironmentalFlow)
    
    return minDischargeForEnvironmentalFlow
#minDischargeForEnvironmentalFlow = self.estimate_discharge_for_environmental_flow(channelStorage)

pcrglobwb_dam = 215
geodar = 21085
time_step = 84600
conversion = 1e6
glolakes = 112629
# 1. Read in PCR GLOBWB Inflow and pull out the correct dam
file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data', 'POINTDATA') + os.sep
pcr.setclone(file_dir + "clone_M40.map")

# These are monthly. We can re-run to get daily for a select region
discharge = xr.open_dataset(file_dir+"sos_resout_final_monthAvg_output.nc") 
storage = xr.open_dataset(file_dir + 'surfaceWaterStorage_monthAvg_output.nc')

dam_file = file_dir + '/lakes_and_reservoirs_05min_geodar_2023.nc'

inflow = xr.open_dataset(file_dir + "soswaterInflow_annuaTot_output.nc") # this is yearly 
demand_tot = xr.open_dataset(file_dir +'sosreduction_demand_dailyTot_output.nc')

latitude = 37.625072479248
longitude = -122.458351135254

#inflow = xr.open_dataset(inflow_file)
dams = xr.open_dataset(dam_file).to_dataframe().reset_index()
storage_pcr_test = xr.open_dataset(file_dir +'sosStor_check_dailyTot_output.nc').sel(lat = latitude, lon = longitude, method = 'nearest').to_dataframe().reset_index().drop(columns= {'lon', 'lat'})
# 'soscurrStor_dailyTot_output.nc'
# pull out the dam and the relevant data
# filtered_dams = dams['waterBodyIds']  == pcrglobwb_dam
# inflow_dam = inflow.where(filtered_dams == True, method = 'nearest')

#inflow_dam = inflow
 
inflow_dam = inflow.sel(lat = latitude, lon = longitude, method = 'nearest').to_dataframe().reset_index().drop(columns= {'lon', 'lat'})
discharge_dam = discharge.sel(lat = latitude, lon = longitude, method = 'nearest').to_dataframe().reset_index().drop(columns= {'lon', 'lat'})
demand_dam = demand_tot.sel(lat = latitude, lon = longitude, method = 'nearest').to_dataframe().reset_index().drop(columns= {'lon', 'lat'})
storage_dam = storage.sel(lat = latitude, lon = longitude, method = 'nearest').to_dataframe().reset_index().drop(columns= {'lon', 'lat'})
#model_flood = xr.open_dataset(file_dir + 'SOSflood_final_dailyTot_output.nc').sel(lat = latitude, lon = longitude, method = 'nearest').to_dataframe().reset_index().drop(columns= {'lon', 'lat'})
#model_conservation = xr.open_dataset(file_dir + 'SOSconserve_final_dailyTot_output.nc').sel(lat = latitude, lon = longitude, method = 'nearest').to_dataframe().reset_index().drop(columns= {'lon', 'lat'})
model_reduction_factor = xr.open_dataset(file_dir + 'sosreduction_monthAvg_output.nc').sel(lat = latitude, lon = longitude, method = 'nearest').to_dataframe().reset_index().drop(columns= {'lon', 'lat'})
usage_check = xr.open_dataset(file_dir + 'reservoir_use.nc').sel(lat = latitude, lon = longitude, method = 'nearest').to_dataframe().reset_index().drop(columns= {'lon', 'lat'})
demand_rf = xr.open_dataset(file_dir + 'sosreduction_demand_dailyTot_output.nc').sel(lat = latitude, lon = longitude, method = 'nearest').to_dataframe().reset_index().drop(columns= {'lon', 'lat'})

env_flow_all = xr.open_dataset(file_dir + 'soswater_env_flow_monthTot_output.nc').sel(lat = latitude, lon = longitude, method = 'nearest').to_dataframe().reset_index().drop(columns= {'lon', 'lat'})


output = pd.merge(inflow_dam, discharge_dam, on= 'time')
output = pd.merge(output, demand_dam, on = 'time')
output = pd.merge(output, storage_pcr_test, on = 'time')
output = pd.merge(output, env_flow_all, on = 'time')
avg_inflow = inflow_dam['soswater_inflow'].mean()
# output= output[output['time'] > '1999-12-31']
# output = output[output['time']< '2001-01-01']
output['modelled_storage'] = 0.0
output['model_release'] = 0.0
output['flood'] = 0.0
output['conservation'] = 0.0
output['model_current_storage'] = 0.0
output['reduction_factor_model'] = 0.0


#output['soswater_inflow'] = output['soswater_inflow'] +output['soswater_inflow'].mean()
# Add in flood point
#output.iloc[50, 1] = output.iloc[50,1]*10000
#output.iloc[50:100, 3] = output.iloc[50:100,3]*10
#output.iloc[50:100, 1] = output.iloc[50:100,1]*2 
#output.iloc[250:350, 1] = output.iloc[250:350,1]*100
### add in drought point
##output.iloc[100:110, 1] = 0
#output.iloc[51:150, 1] = output.iloc[51:150,1]*0  #put drought at beginning to see what happens

### Make 10 years of data
#output = pd.concat([output]*10)

# ### add in flood and drought
#output.iloc[400:600, 1] = output.iloc[400:600, 1]*10000
#output.iloc[800:1000,1] = 0

output['day'] = range(0,len(output))
cap_215 = 23.5*1e6
for index in range(0,len(output)):
    logger.debug("processing timestep %s", index)

    if index ==1:
       previous_storage = output.iloc[index-1, 4]
       #break
    else:
        previous_storage = output.iloc[index-1, 9]
    date_string = output.iloc[index, 0]
    inflow_val= output.iloc[index, 1]/365 #because I only have yearly data
    dem_val = output.iloc[index,3]
    week = date_string.isocalendar().week
    if week >52:
        week = 52
    #outflow, flood_val, conservation_val, current_stor = TurnerOutflow_old(inflow_val,  dem_val, previous_storage, week, date_string)
    hydropower = 1
    usage_check['use_check'].mean()
    env_flow = output.iloc[index, 5]*86400
    outflow, flood_val, conservation_val, current_stor, current_rf = TurnerOutflow(inflow_val, avg_inflow,env_flow, dem_val, previous_storage, week, date_string, hydropower)
    logger.debug("current_stor: %s", current_stor)
    if current_stor < 0:
        break
    output.iloc[index, 6] = previous_storage
    output.iloc[index, 7] = outflow
    output.iloc[index, 8] = flood_val
    output.iloc[index, 9] = conservation_val
    output.iloc[index, 10] = current_stor
    output.iloc[index,11] = current_rf
    water_balance = inflow_val - outflow + previous_storage
    stor_diff = current_stor - previous_storage
    if round(water_balance) != round(current_stor): #or round(stor_diff) != round(inflow_val):
        logger.error("water balance check failed at timestep %s: balance=%.2f, storage=%.2f", index, water_balance, current_stor)
        break





#variables = pd.read_csv(filepath)

#plt.plot(output['time'], output['soswater_current_storage'], color = 'pink', label = 'pcrglobwb storage')
#plt.plot(output['day'], output['surface_water_storage'], color = 'pink', label = 'pcrglobwb storage final')
plt.plot(output['day'], output['sos_storage_check']/cap_215, label = 'current storage pcr function')
plt.axhline(y=cap_215/cap_215, color='black', linestyle='-')
#plt.plot(output['day'], output['soswater_demand']/cap_215, color ='grey',label = 'demand')
plt.plot(output['day'], output['model_current_storage']/cap_215, color ='purple',label = 'modelled storage 1D')
plt.plot(output['day'], output['model_release']/cap_215, color ='green',label = 'modelled release')
#plt.plot(output['day'], discharge_dam['sos_reservoir_outflow_end']/cap_215, color = 'blue', label = 'reservoir Outflow pcr')
plt.plot(output['day'], output['soswater_inflow']/cap_215, color ='grey', linestyle = 'dashed',label = 'modelled inflow')
plt.plot(output['day'], output['flood']/cap_215, color = 'blue', linestyle = 'dashed', label = 'flood')
plt.plot(output['day'], output['conservation']/cap_215, color = 'red', linestyle = 'dashed', label = 'conservation')
plt.xticks(rotation = 90)
plt.legend()

#

## Different Checks you can do to make sure things align 
# conservation
#diff_conserve= model_conservation['soswater_conserve_final'] - output['conservation']
#print(str(diff_conserve.sum()) + " is conservation difference")
## flood
#diff_flood = model_flood['soswater_flood_final'] - output['flood']
#print(str(diff_flood.sum()) + " is flood difference")
#
## reduction factor
#diff_rf = model_reduction_factor['soswater_reduction_factor'] - output['reduction_factor_model']
#print(str(diff_rf.sum()) + " is RF difference")
#
## rf_demand 
#diff_demand_rf = model_reduction_factor['soswater_reduction_factor'] - demand_rf['soswater_RF_demand']
#print(str(diff_rf.sum()) + " is RF difference")
#
## usage_check
#diff_use = usage_check['sos_main_use'] - hydropower
#print(str(diff_use.sum()) + " is hydropower check difference")
#
## env flow
#diff_env = env_flow_all['soswater_env_flow']*86400 - output['soswater_inflow'].mean()
#print(str(diff_env.sum()) + " is environmental flow difference")


# diff check 
#diff_hydropower =  storage_pcr_test['soswater_current_storage'] - discharge_dam['sos_reservoir_outflow']
#plt.plot(output['day'], diff_hydropower, color = 'red')
#plt.plot(output['day'], output['conservation'], color = 'blue')
#plt.plot(output['day'], storage_pcr_test['soswater_current_storage'] , color ='purple',label = 'modelled storage 1D')

#print(diff_hydropower - output['conservation'])