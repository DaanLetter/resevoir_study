import logging

import numpy as np
import matplotlib.pyplot as plt

#based on Steyaert2025: Data derived reservoir operations simulated in a global hydrologic model

logger = logging.getLogger(__name__)

BANKFULL_NUMBER = 2.3 #ratio of bankfull discharge to the average discharge

def reduction_factor(current_storage, min_storage, max_storage):

    #equation 3

    if min_storage > max_storage:
        raise ValueError(f'Maximum Storage {max_storage:.2f} cannot Exceed Minimum Storage {min_storage:.2f}')
   
    rf = (current_storage-min_storage)/(max_storage-min_storage)
    if rf < 0:
        rf = 0
    elif rf > 1:
        rf = 1
    return rf

def initial_discharge(current_storage, max_storage, min_storage, avg_outflow, avg_discharge):

    #equation 1

    RF = reduction_factor(current_storage, min_storage, max_storage)
    Ri = avg_outflow*RF
    if Ri > BANKFULL_NUMBER:
        logger.warning("Ri (%.2f) exceeds bankfull discharge threshold of %s", Ri, BANKFULL_NUMBER * avg_discharge)
    return Ri

def new_storage(current_storage, max_storage, min_storage, avg_outflow, inflow=0, precipitation=0, evaporation=0,):

    #equation 2, I am not sure yet what needs to be done with Inflow, Precipitation and evaporation
    #This updated storage is used to define the reduction factor. I am still thinking of a way how to implement the timesteps 
    #I might need to change initial discharge to a release function based on the timestep. 

    release = initial_discharge(current_storage, max_storage, min_storage, avg_outflow) #The release should be calculated and not just be the initial release. Some Logic is still required for this.
    Sn = current_storage + inflow + precipitation - evaporation - release

    return Sn

def flood_release(current_storage, release, max_storage):

    #determines the extra flood release when required

    return max(current_storage-release-max_storage, 0)

def generic_release(current_storage, avg_discharge, storage_capacity, bankfull_discharge, bankfull_number=BANKFULL_NUMBER):
    
    #equation 4, performing the generic reservoir scheme in PCR-GLOBWB 2. Smin and Smax are set at 10 and 75 percent of storage capacity as the active zone of a resevoir.
    #in all instances, demand is set to 0. 
 
    min_storage = 0.1*storage_capacity
    max_storage = 0.75*storage_capacity
    if current_storage < min_storage:
        release = 0
    elif min_storage < current_storage < max_storage:
        release = reduction_factor(current_storage, min_storage, max_storage)*avg_discharge
    elif current_storage > max_storage:
        release = (current_storage-storage_capacity)/(max_storage-storage_capacity) * (bankfull_discharge-avg_discharge) + bankfull_number

    if current_storage-release > max_storage: #check for flood conditions.
        release = release + flood_release(current_storage, release, max_storage)
    return release


def starfit_release(current_storage, storage_capacity, current_release, max_storage, min_storage, avg_outflow, env_flow, demand, use='irrigation'):
   
    #equation 6. I am not sure how to model the current_release and if under the flood conditions the Ri means initial release or irrigation release
    # The environmental flow requirement is defined in PCR-GLOBWB 2

    release = 0

    ID = initial_discharge(current_storage, max_storage, min_storage, avg_outflow)
    Ri = irrigation_release(min_storage, max_storage, current_storage, current_release, demand, storage_capacity)
    Rh = hydropower_release(current_storage, min_storage, max_storage, current_release, demand)
    Rf = flood_release(current_storage, current_release, max_storage)

    if current_storage - current_release >= storage_capacity:
        release = ID  + Rf
    elif min_storage < current_storage < max_storage and use == 'irrigation':
        release = Ri
    elif min_storage < current_storage < max_storage and use != 'irrigation':
        release = Rh
    elif current_storage < min_storage:
        active_release = Ri if use == 'irrigation' else Rh
        if active_release < env_flow and current_storage - env_flow > 0:
            release = env_flow
    return release
        
def initial_hydro_release(current_release, demand, current_storage, max_storage, min_storage, bankfull_number=BANKFULL_NUMBER):

    #equation 7
    
    release = current_release

    RF = reduction_factor(current_storage, min_storage, max_storage)

    if current_release < demand:
        release = demand*RF/bankfull_number
    elif current_release > demand:
        release = current_release
    return release

def hydropower_release(current_storage, min_storage, max_storage, current_release, demand):

    #equation 8

    release = 0

    Rhi = initial_hydro_release(current_release, demand, current_storage, max_storage, min_storage, bankfull_number=BANKFULL_NUMBER)
    
    if min_storage < current_storage < max_storage and current_storage - Rhi > min_storage:
        release = current_storage - Rhi
    elif current_storage - Rhi < min_storage:
        release = max(current_storage-min_storage, 0)
    return release

def irrigation_release(min_storage, max_storage, current_storage, current_release, demand, storage_capacity):

    #equation 9

    release = 0

    RF = reduction_factor(current_storage, min_storage, max_storage)

    if min_storage < current_storage < max_storage and current_release > demand:
        release = current_release
    elif min_storage < current_storage < max_storage and current_release < demand:
        release = RF*demand
    if current_storage - current_release < 0.1*storage_capacity:
        release = max(current_storage-0.1*storage_capacity, 0)
    return release     


def main():
    storagelist = []
    timesteps = np.arange(0, 100, 1)
    min_storage, max_storage, current_storage, avg_outflow = 10, 75, 50, 1
    storage_capacity, bankfull_discharge = 100, 2.3

    for t in timesteps:
        release = generic_release(current_storage, avg_outflow, storage_capacity, bankfull_discharge)
        current_storage = current_storage - release  # simplified water balance
        storagelist.append(current_storage)

    plt.plot(timesteps, storagelist)
    plt.show()
    
if __name__ == '__main__':
    main()
