from ase import Atoms
from ase.io import write, read
from ase.neighborlist import neighbor_list
from ase.data import covalent_radii
from rdkit import Chem
import numpy as np
import shutil
import pandas as pd

# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# No bond-breaking for adsorbates
# --------------------------------------------------------------------------------------------------------------------------------------------------------------

def getConnectivitymatrix(atoms, coef=1.1):

    pos = atoms.positions
    atomNumbers = atoms.numbers
    n = len(atoms)
    conn = np.zeros((n,n), dtype=int)

    for i in range(n):
        for j in range(i+1, n):

            d = np.linalg.norm(pos[i] - pos[j])

            #print(i,j,d, covalent_radii[atomNumbers[i]] + covalent_radii[atomNumbers[j]])
            
            if d < coef*(covalent_radii[atomNumbers[i]] + covalent_radii[atomNumbers[j]]):
                conn[i,j] = conn[j,i] = 1

    return conn


def adsorbateFilter(optConfigs, id, mlip_name):

    if id == '1221':
        coef = 1.3
    else:
        coef = 1.1

    filteredConfigs = []

    initConfigAtoms = read(f"xyz-s/all/{id}.xyz")

    initConfigConn = getConnectivitymatrix(initConfigAtoms, coef=coef)
    
    for config in optConfigs:
        symbols = []
        coords = []
        with open(f'{mlip_name}/opt-configurations/id{id}/{config}', 'r') as f:
            lines = f.readlines()

        linesMol = [line for line in lines[2:] if not 'Zn' in line]
        
        for line in linesMol:
            symbols.append(line.split()[0])
            coords.append([float(i) for i in line.split()[1:4]])

        optConfigAtoms = Atoms(symbols=symbols, positions=coords)
    
        optConfigConn = getConnectivitymatrix(optConfigAtoms, coef=coef)

        if np.array_equal(initConfigConn, optConfigConn):
            filteredConfigs.append(config)

    return filteredConfigs            



# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# No molecule flying away
# --------------------------------------------------------------------------------------------------------------------------------------------------------------

def get_min_z_dist(config, id, mlip_name):
    """ Input: coordinates of Molecule (linesMol) and Slab (linesSlab)
        Output: minimum distance of molecule's atoms to the slab in z-direction
    """

    with open(f'{mlip_name}/opt-configurations/id{id}/{config}', 'r') as f:
        lines = f.readlines()
        linesMol = [line for line in lines[2:] if not 'Zn' in line]
        linesSlab = [line for line in lines[2:] if 'Zn' in line]

    if len(linesSlab[0].split()) <= 4:
        slab_z = np.max([float(line.split()[-1]) for line in linesSlab])
        z_coords = [float(line.split()[-1]) for line in linesMol]
    else:
        slab_z = np.max([float(line.split()[3]) for line in linesSlab])
        z_coords = [float(line.split()[3]) for line in linesMol]

    min_z_coord = np.min(z_coords)

    return min_z_coord - slab_z


def flyFilter(configs, id, mlip_name):
    
    filteredConfigs = []
    
    for config in configs:
        
        z_min_dist = get_min_z_dist(config, id, mlip_name)

        if z_min_dist <= 4.0:
            filteredConfigs.append(config)

    return filteredConfigs

# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# No bond-breaking or bond-formation in Molecule and Slab
# --------------------------------------------------------------------------------------------------------------------------------------------------------------

def getConnectivityArray(atoms, cushion):

    pos = atoms.positions
    atomNumbers = atoms.numbers
    n = len(atoms)
    conn_arr = set()

    for i in range(n):
        for j in range(i+1, n):

            d = np.linalg.norm(pos[i] - pos[j])
            # print(cushion*(covalent_radii[atomNumbers[i]] + covalent_radii[atomNumbers[j]]))
            
            if d < cushion*(covalent_radii[atomNumbers[i]] + covalent_radii[atomNumbers[j]]):
                conn_arr.add(f"{i}-{j}")

    return conn_arr

def cushionFilter(configs, id, mlip_name):
    
    filteredConfigs = []

    # Connectivity array of Relaxed Slab
    slab = read(f'{mlip_name}/opt-slabs/zn_opt.xyz')

    optSlabConnArrWithoutCushion = getConnectivityArray(slab, cushion=1.2)
    optSlabConnArrWithCushion = getConnectivityArray(slab, cushion=1.8)
    
    # Connectivity array of Relaxed Slab + Adsorbate
    for config in configs:
        symbols = []
        coords = []
        with open(f'{mlip_name}/opt-configurations/id{id}/{config}', 'r') as f:
            lines = f.readlines()

        linesSlab = [line for line in lines[2:] if 'Zn' in line]
        
        for line in linesSlab:
            symbols.append(line.split()[0])
            coords.append([float(i) for i in line.split()[1:4]])

        optConfigAtoms = Atoms(symbols=symbols, positions=coords)
    
        optConfigConnArrWithoutCushion = getConnectivityArray(optConfigAtoms, cushion=1.2)
        optConfigConnArrWithCushion = getConnectivityArray(optConfigAtoms, cushion=1.8)
        # print(len(optConfigConnArrWithoutCushion), len(optSlabConnArrWithCushion))

        subConstraint1 = optConfigConnArrWithoutCushion.issubset(optSlabConnArrWithCushion)
        subConstraint2 = optSlabConnArrWithoutCushion.issubset(optConfigConnArrWithCushion)

        if subConstraint1 and subConstraint2:
            filteredConfigs.append(config)

    return filteredConfigs
    
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Discard filtered configurations
# --------------------------------------------------------------------------------------------------------------------------------------------------------------

def discardFiltered(id, filteredConfigs, optConfigs, mlip_name):
    """ move configurations that do not fit constraints to opt-configurations/discarded folder """

    discardConfigs = [config for config in optConfigs if config not in filteredConfigs]

    # move the xyz files
    for config in discardConfigs:
        shutil.move(f'{mlip_name}/opt-configurations/id{id}/{config}', f'{mlip_name}/opt-configurations/discarded/{config}')

    return None

def updateEnergies(id, filteredConfigs, optConfigs, mlip_name):
    """ update the energy csv """

    discardConfigs = [config for config in optConfigs if config not in filteredConfigs]

    discardConfigsICalcs = [int(config.split('-')[1].split('.')[0]) for config in discardConfigs]
    energies_df = pd.read_csv(f"{mlip_name}/interaction-energy-values/id{id}_interaction_energies.csv", index_col=0)
    discardConfigsICalcs = [ICalc for ICalc in discardConfigsICalcs if ICalc in energies_df.index.tolist()]            # filter structures that did not converge and thus their E_int values were not saved in interaction-values.csv file
    energies_df.drop(discardConfigsICalcs, inplace=True)
    energies_df.to_csv(f"{mlip_name}/cushion-test-filtered-interaction-energy-values/id{id}_interaction_energies.csv")

    return None
