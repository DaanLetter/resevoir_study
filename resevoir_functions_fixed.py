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

def new_storage(release, current_storage, inflow=0, precipitation=0, evaporation=0, storage_capacity=None):

    #equation 2
    # FIX: added optional storage_capacity cap — any volume above physical capacity spills immediately
    Sn = current_storage + inflow + precipitation - evaporation - release

    if storage_capacity is not None:
        Sn = min(Sn, storage_capacity)

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


def starfit_release(current_storage, storage_capacity, max_storage, min_storage, avg_outflow, env_flow, demand, avg_discharge, current_release=0, use='hydropower'):

    #equation 6.
    # FIX: added surplus-zone branch for max_storage <= S < storage_capacity.
    # Previously this gap caused release=0 whenever storage was above the flood bound
    # but below full capacity, allowing modelled_storage to exceed physical capacity.

    release = 0

    ID = initial_discharge(current_storage, max_storage, min_storage, avg_outflow, avg_discharge)
    Ri = irrigation_release(min_storage, max_storage, current_storage, current_release, demand, storage_capacity)
    Rh = hydropower_release(current_storage, min_storage, max_storage, current_release, demand)
    Rf = flood_release(current_storage, current_release, max_storage)

    if current_storage - current_release >= storage_capacity:
        # At or above full capacity: release initial discharge plus all excess
        release = ID + Rf
    elif current_storage >= max_storage:
        # FIX: surplus zone — above flood bound but below capacity.
        # Release the excess above max_storage so storage is brought back to the flood bound.
        release = current_storage - max_storage
    elif min_storage < current_storage < max_storage and use != 'hydropower':
        release = Ri
    elif min_storage < current_storage < max_storage and use == 'hydropower':
        release = Rh
    elif current_storage < min_storage:
        active_release = Ri if use != 'hydropower' else Rh
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
    # FIX: was returning current_storage - Rhi, which drained the reservoir in one step.
    # Correct behaviour: release Rhi (the scaled demand), capped so storage stays above min.

    release = 0

    Rhi = initial_hydro_release(current_release, demand, current_storage, max_storage, min_storage, bankfull_number=BANKFULL_NUMBER)

    if min_storage < current_storage < max_storage and current_storage - Rhi > min_storage:
        release = Rhi
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
