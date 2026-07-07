from datetime import datetime
print("\n-- Run initiated | ", datetime.now())

# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Important notes
# --------------------------------------------------------------------------------------------------------------------------------------------------------------

# Activate "M3GNet" conda environment to use this program

# python3 -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
# python3 -m pip install dgl -f https://data.dgl.ai/wheels/cu124/repo.html
# python3 -m pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html
# python3 -m pip install matgl
# 
# add matgl.set_backend('DGL') to the file matgl/ext/ase.py after 'import matgl' line

# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Import the MLIP potentials
# --------------------------------------------------------------------------------------------------------------------------------------------------------------

import os
os.environ['PYTHONNOUSERSITE'] = '1'

import matgl
from matgl.ext.ase import PESCalculator
matgl.set_backend('DGL')

# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Read the Dataset and get the molecules to run calculations for

import pandas as pd
import numpy as np

# Read the Dataset
dataset = pd.read_csv("dataset/mols-energies.csv", index_col=3)
ids = dataset.index.to_list()

print("\n-- Parameters " + "-"*100)
print(f"-- Ids : {', '.join([str(i) for i in ids])}")
print("-"*100+"\n")

# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Running Abcluster for molecules
# --------------------------------------------------------------------------------------------------------------------------------------------------------------

from tqdm import tqdm
import os
from abcluster_functions import *

# parameters
n_calcs = 500

abcluster_ls = os.listdir('./abcluster/')
if len(abcluster_ls) != 1:
    print("previous abcluster-generated results were found in the abcluster folder.")

abcluster_input = input("Do you want to run the abcluster part of the test (create inputs --> run abcluster --> filter configurations)? (Yes / No) ")

if abcluster_input.lower() == 'yes':

    base_folder = '/home/sakengali/Desktop/research/SIB-additives/interaction-energies/side-proj-E-int/MLIPs'

    n_fit_arr = []
    for id in tqdm(ids, desc="Molecule count", position=0, ascii=True, leave=True): # tqdm(ids, desc='molecule'):

        write_abcluster_input(id, n_calcs)
        run_abcluster_input(base_folder, id, n_calcs)
        print(f'configurations were created for id{id}_{n_calcs}')

        n_fit = filter_configurations(id, n_calcs, z_low = 1.0, z_upp = 2.0)
        print(f'{id} filtering done: {n_fit} configurations fit constraints')

        n_fit_arr.append(n_fit)

    print(f"Average number of configurations: {round(np.mean(n_fit_arr), 1)}")
        


# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Setting up calculation trackers
# --------------------------------------------------------------------------------------------------------------------------------------------------------------

import signal

# Setting up the MLIP potential
mlip_name = "M3GNet"

# overtime handler
max_time = 180 # 1200 seconds = 20 minutes
def overtime_handler(signum, frame):
    print(f'optimization running more than {max_time} seconds')
    raise TimeoutError('killing it and moving on ...')

# register function handler
signal.signal(signal.SIGALRM, overtime_handler)



# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Running Interaction Energy Calculations
# --------------------------------------------------------------------------------------------------------------------------------------------------------------

from energy_functions import *
from helper_functions import *

mlip_input = input(f"\nDo you want to initiate the calculation of interaction energies using the {mlip_name} potential? (Yes / No) ")

if mlip_input.lower() == 'yes':

    # loading the potential to calculate slab energy
    model = matgl.load_model("M3GNet-MP-2021.2.8-PES")
    mlip_pot = PESCalculator(potential=model)

    # energy of the slab
    slab_energy = calculate_slab_energy(mlip_name=mlip_name, mlip_pot=mlip_pot)
    for id in tqdm(ids, desc="Molecule count", position=0, ascii=True, leave=True):

        model = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        mlip_pot = PESCalculator(potential=model)

        skip_outer_loop = False

        os.makedirs(f'./{mlip_name}/opt-configurations/id{id}/', exist_ok=True)

        id_energies = []
        n_converged = 0

        # energy of the molecule
        molecule_energy = calculate_molecule_energy(mlip_name=mlip_name, mlip_pot=mlip_pot, id=id)

        xyz_files = np.array(glob.glob(f'abcluster/id{id}_{n_calcs}/*xyz'))
        i_calcs = sorted([file.split('/')[2].split('.')[0] for file in xyz_files])

        for i, i_calc in tqdm(enumerate(i_calcs), desc=f"i_calc for {id}", position=1, ascii=True, leave=False):

            tqdm.write(f"calculating energy of slab + molecule {id} | i_calc = {i_calc}")

            # set timeout for the energy calculation function
            signal.alarm(max_time)
            try:

                # energy of slab + molecule
                slab_plus_molecule_energy = calculate_slab_plus_molecule_energy(mlip_name=mlip_name, mlip_pot=mlip_pot, id=id, n_calcs=n_calcs, i_calc=i_calc)

                id_energies.append(
                    (i_calc, slab_plus_molecule_energy - molecule_energy - slab_energy)
                )
                
                n_converged += 1
                
            except TimeoutError as e:
                print(e)

                if i == 2 and n_converged == 0:
                    skip_outer_loop = True
                    with open("convergence_failure_log.txt", "a+") as f:
                        f.write(f"id{id} convergence failed for 3 configurations in a row\n")
                    break

            # reset timeout if the functionr returned before timeout
            signal.alarm(0)

        if skip_outer_loop:
            skip_outer_loop = False
            continue

        id_energies_sorted = sorted(id_energies, key=lambda x: x[1])

        id_energies_df = pd.DataFrame({
            "i_calc" : [item[0] for item in id_energies_sorted],
            "E_interaction" : [item[1] for item in id_energies_sorted]
        })

        #write the info for configurations with the lowest energies
        id_energies_df.to_csv(f"{mlip_name}/interaction-energy-values/id{id}_interaction_energies.csv", index=False)

        print('-'*100)
        print(f"id {id}-- done.")
        print('-'*100)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Applying constraints
# --------------------------------------------------------------------------------------------------------------------------------------------------------------

from constraint_functions import *

constraint_input = input(f"\nDo you want to apply the post-MLIP constraints to the optimized structures? (Yes / No) ")

if constraint_input.lower() == 'yes':
    print('-'*100)

    with open(f"{mlip_name}/constraints.log", "a+") as f:
       f.write("id\tinit\tpass\tperc\n")

    for id in tqdm(ids, desc="Molecule count", position=0, ascii=True, leave=True):

        optConfigs = os.listdir(f'{mlip_name}/opt-configurations/id{id}/')

        print('-'*120)
        print(f"Applying constraints to id{id} | Initial number of configurations: {len(optConfigs)}")

        # Constraint 1: No bond-breaking for adsorbates
        print("Constraint 1: No bond-breaking for adsorbates ")
        filteredConfigs_step1 = adsorbateFilter(optConfigs, id, mlip_name)
        print(f"{len(filteredConfigs_step1)} configurations pass")

        # Constraint 2: No molecule flying away
        print("Constraint 2: No molecule flying away")
        filteredConfigs_step2 = flyFilter(filteredConfigs_step1, id, mlip_name)
        print(f"{len(filteredConfigs_step2)} configurations pass")

        # Constraint 3: No significant changes in the slab structure
        print("Constraint 2: No significant changes in the slab structure")
        filteredConfigs_step3 = cushionFilter(filteredConfigs_step2, id, mlip_name)
        print(f"{len(filteredConfigs_step3)} configurations pass")

        # discard the structures that did not pass the filter -- was abandoned in case there was a mistake when filtering
        # discardFiltered(id, filteredConfigs_step3, optConfigs, mlip_name)
        # print(f"Configurations that did not satisfy constraints were moved to {mlip_name}/opt-configurations/discarded/ folder")

        updateEnergies(id, filteredConfigs_step3, optConfigs, mlip_name)
        print(f"Energies for id{id} were updated")
        print('-'*120)

        with open(f"{mlip_name}/constraints.log", "a+") as f:
            f.write(f"{id}\t{len(optConfigs)}\t\t{len(filteredConfigs_step3)}\t\t{len(filteredConfigs_step3)/len(optConfigs) * 100}%\n")

    print('-'*120)



# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Distincting top configurations
# --------------------------------------------------------------------------------------------------------------------------------------------------------------

Nfinal = 5
coef = 0.8
distincting_input = input(f"\nDo you want to apply the distinction algorithm and obtain {Nfinal} distinct configurations (RMSD bar coefficient = {coef})? (Yes / No) ")

if distincting_input.lower() == "yes":

    from distincting_functions import getRMSD, getNumberOfAtoms, getRangeRMSDMeans
    
    fewDistinct = {}

    rangeRMSDMeans = getRangeRMSDMeans(mlip_name)
    
    for id in tqdm(ids, desc="Molecule count", position=0, ascii=True, leave=True):
        # print(id)
        configsDF = pd.read_csv(f"{mlip_name}/filtered-interaction-energy-values/id{id}_interaction_energies.csv", index_col=0)
        configs = configsDF.index
        distinctConfigs = []

        nAtoms = getNumberOfAtoms(id)
        for range, rmsdMean in rangeRMSDMeans.items():
            a, b = (int(k) for k in range.split('-'))
            if a <= nAtoms < b:
                RMSDBar = rmsdMean * coef
                break

        for i, config in enumerate(configs):
            
            if i == 0:
                distinctConfigs.append(config)
                continue
            
            RMSDarr = []
            for acceptedConfig in distinctConfigs:
                RMSD = getRMSD(mlip_name, id, acceptedConfig, config)
                # print(acceptedConfig, config, RMSD)
                RMSDarr.append(RMSD)

            if all([rmsd > RMSDBar for rmsd in RMSDarr]):
                distinctConfigs.append(config)

            if len(distinctConfigs) == Nfinal:
                updatedDF = configsDF.loc[distinctConfigs, :]
                updatedDF.to_csv(f"{mlip_name}/distinct-top-configurations/id{id}_interaction_energies.csv")
                break

        if len(distinctConfigs) < Nfinal:
            fewDistinct[id] = len(distinctConfigs)
            updatedDF = configsDF.loc[distinctConfigs, :]
            updatedDF.to_csv(f"{mlip_name}/distinct-top-configurations/id{id}_interaction_energies.csv")

    print(f"\nFollowing molecules had less than {Nfinal} groups (N = {len(fewDistinct)}): {', '.join([f'{i} ({k})' for i, k in fewDistinct.items()])}")
