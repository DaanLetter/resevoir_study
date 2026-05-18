# Function to Model Reservoirs based on Turner et al., 2021
# JCS
# 8-2-2023

import logging
import os

import xarray as xr
import numpy as np
import pcraster as pcr
import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def TurnerOutflow(inflow_val, avg_inflow, env_flow, dem_val, previous_storage, week, date_string, hydropower):
    # Hardcoded capacity for PCR-GLOBWB dam 215 (San Antonio reservoir)
    cap_215 = 23.5*1e6
    date_val = date_string.day
    month_val = date_string.month
    new_week_val = week - 1

    logger.debug("week: %s", new_week_val)
    date_string = str('2000') + "-" + str(month_val) + "-" + str(date_val)
    _rf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data', 'POINTDATA', '10_param_RF_bounds_final')
    variables = xr.open_dataset(os.path.join(_rf_dir, date_string + '.nc')).sel(latitude=latitude, longitude=longitude, method='nearest').to_dataframe().reset_index()
    flood = int(variables['flood'].iloc[0] / 100 * cap_215)
    logger.info("flood: %s", flood)
    conservation = int(variables['conservation'].iloc[0] / 100 * cap_215)
    logger.debug("conservation: %s", conservation)

    bankfull = 2.3  # maximum release multiplier; release = bankfull * avg_inflow at the flood pool
    current_storage = previous_storage + max(inflow_val, 0)

    # Reduction factor linearly scales from 0 at conservation pool to bankfull at flood pool
    reduction_factor = max((current_storage - conservation) / (flood - conservation) * bankfull, 0)
    # Demand reduction factor scales from 0 at dead storage (10% cap) to 1 at conservation pool
    demand_reduction_factor = (current_storage - 0.1 * cap_215) / (conservation - 0.1 * cap_215)

    release = reduction_factor * avg_inflow

    ## FLOOD: cap reduction factor once above flood pool, then spill any excess above full capacity
    if reduction_factor > bankfull and current_storage > flood:
        reduction_factor = bankfull
    if current_storage - release > cap_215:
        flood_difference = current_storage - release - cap_215
        release = release + flood_difference

    ## ACTIVE STORAGE FOR NON HYDROPOWER
    if release < dem_val and hydropower == 0:
        release = dem_val * demand_reduction_factor
        logger.debug("non hydropower active storage")
    ## ACTIVE STORAGE FOR HYDROPOWER
    if release < dem_val and hydropower == 1:
        release = dem_val * reduction_factor / bankfull
        if current_storage - release < conservation:
            release = current_storage - conservation  # don't draw below conservation pool
        logger.debug("hydropower active storage")

    #### NON NEGATIVE STORAGE CHECK
    test_storage_non_neg = current_storage - release
    if test_storage_non_neg < 0:
        release = release / bankfull * (1 - flood / cap_215)

    ### ENV FLOW CHECK
    test_storage = current_storage - env_flow
    if release < env_flow and test_storage > 0:
        release = env_flow
    ## CATCH FOR DEAD STORAGE
    if current_storage < 0.1 * cap_215:
        release = 0

    new_storage = current_storage - release
    logger.debug("new_storage: %s", new_storage)
    return (release, flood, conservation, new_storage, reduction_factor)


# PCR-GLOBWB version of the Turner outflow model operating on PCRaster maps
def getReservoirOutflow_Turner(self, avgChannelDischarge, length_of_time_step, downstreamDemand,
                               irrGrossDemand, nonIrrGrossDemand, environmental_flow):
    ##### GET OUTFLOW OF RESERVOIR TO START #####
    # avgOutflow (m3/s)
    avgOutflow = self.avgOutflow
    self.sos_outflow = avgOutflow
    # Fill zero avgOutflow for newly-introduced lakes/reservoirs using downstream or inflow values
    avgOutflow = pcr.ifthenelse(
                 avgOutflow > 0.,
                 avgOutflow,
                 pcr.max(avgChannelDischarge, self.avgInflow))
    avgOutflow = pcr.ifthenelse(
                 avgOutflow > 0.,
                 avgOutflow, pcr.downstream(self.lddMap, avgOutflow))
    avgOutflow = pcr.areamaximum(avgOutflow, self.waterBodyIds)

    resvOutflow = (avgOutflow * length_of_time_step)  # m3

    #### CALCULATE THE FLOOD AND CONSERVATION POOLS #####
    flood_read = self.turnerFlood / 100 * self.waterBodyCap
    conserve_read = self.turnerConservation / 100 * self.waterBodyCap

    # Clamp flood to capacity; floor conservation at 10% of capacity (dead storage boundary)
    flood_final = pcr.min(flood_read, self.waterBodyCap)
    conservation_final = pcr.max(conserve_read, self.waterBodyCap * 0.1)

    self.turnerFloodF = flood_read
    self.turnerConserveF = conserve_read

    ### INITIALIZE ENV FLOW, INFLOW AND CURRENT STORAGE
    self.env_flow = environmental_flow
    bankfull = 2.3

    inflow_res = pcr.ifthenelse(self.inflowInM3PerSec < 0., pcr.scalar(0), self.inflowInM3PerSec) * 86400
    self.sos_inflow = inflow_res
    current_storage = self.waterBodyStorage + pcr.max(inflow_res, pcr.scalar(0))

    ### CALCULATE REDUCTION FACTORS ####
    # RF linearly interpolates from 0 at conservation to bankfull at flood pool
    RF_bankfull = ((current_storage - conservation_final) / (flood_final - conservation_final)) * pcr.scalar(bankfull)
    reduction_factor = pcr.max(RF_bankfull, pcr.scalar(0))
    self.turnerReduction = reduction_factor
    demand_reduction_factor = (current_storage - 0.1 * self.waterBodyCap) / (conservation_final - 0.1 * self.waterBodyCap)
    self.reductionFactorDemand = demand_reduction_factor

    ###### CALCULATE TOTAL DEMAND ######
    demand_tot = irrGrossDemand + nonIrrGrossDemand  # m/day
    # Convert demand from m/day to m3/day using cell area
    downstreamDemand = demand_tot * self.cellArea

    # Restrict demand to reservoir cells only
    downstreamDemand = pcr.ifthen(pcr.scalar(self.waterBodyTyp) == 2.0, downstreamDemand)
    downstreamDemand = pcr.ifthen(pcr.scalar(self.waterBodyIds) > 0., downstreamDemand)

    max_demand = downstreamDemand
    self.sosdemand = max_demand

    ###### FLOOD VALUES ######
    # Cap reduction factor at bankfull once storage exceeds the flood pool
    reduction_factor = pcr.ifthenelse((reduction_factor > bankfull) & (current_storage > flood_final), bankfull, reduction_factor)

    # Release any storage above full capacity as flood spill
    flood_difference = current_storage - resvOutflow - self.waterBodyCap
    resvOutflow = pcr.ifthenelse((current_storage - resvOutflow) > self.waterBodyCap, (resvOutflow + flood_difference), resvOutflow)

    hydropower_check = pcr.scalar(self.use)
    self.hydropower_check = hydropower_check

    ### ACTIVE STORAGE FOR HYDROPOWER ####
    resvOutflow_hydropower = pcr.ifthenelse(
        ((resvOutflow < max_demand) & (hydropower_check == pcr.scalar(1))),
        (max_demand * reduction_factor / bankfull), resvOutflow)
    current_storage_hydropower = current_storage - resvOutflow_hydropower
    # If hydropower release would draw below conservation, release only down to conservation pool
    hydropower_release = resvOutflow_hydropower - conservation_final
    resvOutflow = pcr.ifthenelse(current_storage_hydropower < conservation_final, hydropower_release, resvOutflow)

    ### ENV FLOW CHECK
    test_env_flow = current_storage - environmental_flow
    resvOutflow = pcr.ifthenelse((resvOutflow < environmental_flow) & (test_env_flow > 0), environmental_flow, resvOutflow)

    ## CATCH FOR DEAD STORAGE
    resvOutflow = pcr.ifthenelse(current_storage < (0.1 * self.waterBodyCap), pcr.scalar(0), resvOutflow)

    # Mask output to reservoir cells only
    resvOutflow = pcr.ifthen(pcr.scalar(self.waterBodyIds) > 0., resvOutflow)
    resvOutflow = pcr.ifthen(pcr.scalar(self.waterBodyTyp) == 2, resvOutflow)

    ## CHECK STORAGE
    self.turner_current_stor = pcr.cover(current_storage, pcr.scalar(0.0)) - pcr.cover(resvOutflow, pcr.scalar(0))

    return (resvOutflow)  # unit: m3


def estimate_discharge_for_environmental_flow(self, channelStorage):
    # z-score for 90th percentile; sets the environmental flow threshold+
    z_score = 1.2816

    # Long-term variance of discharge using an online Welford algorithm
    varDischarge = self.m2tDischarge / \
                   pcr.max(1.,
                   pcr.min(self.maxTimestepsToAvgDischargeLong, self.timestepsToAvgDischarge) - 1.)
    stdDischarge = pcr.max(varDischarge**0.5, 0.0)

    # Floor at 10% of avg discharge to prevent flip-flop near the threshold
    minDischargeForEnvironmentalFlow = pcr.max(0.0, self.avgDischarge - z_score * stdDischarge)
    factor = 0.10
    minDischargeForEnvironmentalFlow = pcr.max(factor * self.avgDischarge, minDischargeForEnvironmentalFlow)
    minDischargeForEnvironmentalFlow = pcr.max(0.0, minDischargeForEnvironmentalFlow)

    return minDischargeForEnvironmentalFlow


# ── Data loading ──────────────────────────────────────────────────────────────
file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data', 'POINTDATA') + os.sep
pcr.setclone(file_dir + "clone_M40.map")

# Monthly averages; daily resolution can be re-run for a select region
discharge = xr.open_dataset(file_dir + "sos_resout_final_monthAvg_output.nc")
inflow = xr.open_dataset(file_dir + "soswaterInflow_annuaTot_output.nc")  # yearly totals
demand_tot = xr.open_dataset(file_dir + 'sosreduction_demand_dailyTot_output.nc')

latitude = 37.625072479248
longitude = -122.458351135254

storage_pcr_test = xr.open_dataset(file_dir + 'sosStor_check_dailyTot_output.nc').sel(lat=latitude, lon=longitude, method='nearest').to_dataframe().reset_index().drop(columns={'lon', 'lat'})

inflow_dam = inflow.sel(lat=latitude, lon=longitude, method='nearest').to_dataframe().reset_index().drop(columns={'lon', 'lat'})
discharge_dam = discharge.sel(lat=latitude, lon=longitude, method='nearest').to_dataframe().reset_index().drop(columns={'lon', 'lat'})
demand_dam = demand_tot.sel(lat=latitude, lon=longitude, method='nearest').to_dataframe().reset_index().drop(columns={'lon', 'lat'})
env_flow_all = xr.open_dataset(file_dir + 'soswater_env_flow_monthTot_output.nc').sel(lat=latitude, lon=longitude, method='nearest').to_dataframe().reset_index().drop(columns={'lon', 'lat'})

# ── Merge into a single time-indexed DataFrame ────────────────────────────────
output = pd.merge(inflow_dam, discharge_dam, on='time')
output = pd.merge(output, demand_dam, on='time')
output = pd.merge(output, storage_pcr_test, on='time')
output = pd.merge(output, env_flow_all, on='time')
avg_inflow = inflow_dam['soswater_inflow'].mean()

output['modelled_storage'] = 0.0
output['model_release'] = 0.0
output['flood'] = 0.0
output['conservation'] = 0.0
output['model_current_storage'] = 0.0
output['reduction_factor_model'] = 0.0

output['day'] = range(0, len(output))
cap_215 = 23.5 * 1e6
logger.info('length dataframe', len(output))
logger.debug(output.head())

# ── Main simulation loop ──────────────────────────────────────────────────────
for index in range(0, len(output)):
    logger.debug("processing timestep %s", index)

    if index == 1:
        previous_storage = output.iloc[index - 1, 4]
    else:
        previous_storage = output.iloc[index - 1, 9]

    date_string = output.iloc[index, 0]
    inflow_val = output.iloc[index, 1] / 365  # annual inflow converted to daily
    dem_val = output.iloc[index, 3]
    week = date_string.isocalendar().week
    if week > 52:
        week = 52  # ISO weeks can reach 53; RF lookup files only have 52 entries

    hydropower = 1
    env_flow = output.iloc[index, 5] * 86400  # convert m3/s to m3/day
    outflow, flood_val, conservation_val, current_stor, current_rf = TurnerOutflow(
        inflow_val, avg_inflow, env_flow, dem_val, previous_storage, week, date_string, hydropower)
    logger.debug("current_stor: %s", current_stor)
    if current_stor < 0:
        break

    output.iloc[index, 6] = previous_storage
    output.iloc[index, 7] = outflow
    output.iloc[index, 8] = flood_val
    output.iloc[index, 9] = conservation_val
    output.iloc[index, 10] = current_stor
    output.iloc[index, 11] = current_rf

    water_balance = inflow_val - outflow + previous_storage
    if round(water_balance) != round(current_stor):
        logger.error("water balance check failed at timestep %s: balance=%.2f, storage=%.2f", index, water_balance, current_stor)
        break


# ── Plotting ──────────────────────────────────────────────────────────────────
plt.plot(output['day'], output['sos_storage_check'] / cap_215, label='current storage pcr function')
plt.axhline(y=cap_215 / cap_215, color='black', linestyle='-')
plt.plot(output['day'], output['model_current_storage'] / cap_215, color='purple', label='modelled storage 1D')
# plt.plot(output['day'], output['model_release'] / cap_215, color='green', label='modelled release')
# plt.plot(output['day'], output['soswater_inflow'] / cap_215, color='grey', linestyle='dashed', label='modelled inflow')
# plt.plot(output['day'], output['flood'] / cap_215, color='blue', linestyle='dashed', label='flood')
# plt.plot(output['day'], output['conservation'] / cap_215, color='red', linestyle='dashed', label='conservation')
plt.xticks(rotation=90)
plt.legend()
plt.show()