import pandas as pd
from ase.io import read, write
from ase import Atoms
import numpy as np
from spyrmsd import rmsd

def getCutStructure(path, Nslab=48):

    """
        Get path to the xyz file -> return ase Mol file with the bottom slab layers cut
    """

    with open(path) as f:
        lines = f.readlines()

    #number of atoms to cut
    Ncut = int(Nslab/2)

    # cut the lines for the bottom half of the slab
    del lines[2:Ncut+2]

    newN = int(float(lines[0]) - Ncut)
    
    symbols = []
    positions = np.empty((newN, 3), dtype=float)

    if lines[-1] == "\n":
        posLines = lines[2:-1]
    else:
        posLines = lines[2:]

    for i, line in enumerate(posLines):
        comps = line.split()
        symbols.append(comps[0])
        positions[i, :] = [float(x) for x in comps[1:4]]

    return Atoms(symbols=symbols, positions=positions)



def getRMSD(mlip_name, id, config1, config2, Nslab = 48):
    """
        Function to read the zys files of the MLIP-optimized configurations and return their rmsd
    """

    atoms1 = getCutStructure(f"{mlip_name}/opt-configurations/id{id}/id{id}-{config1}.xyz")
    atoms2 = getCutStructure(f"{mlip_name}/opt-configurations/id{id}/id{id}-{config2}.xyz")

    config1Coords = atoms1.get_positions()
    config1Numbers = atoms1.get_atomic_numbers()
    config2Coords = atoms2.get_positions()
    config2Numbers = atoms2.get_atomic_numbers()

    min_rmsd = rmsd.rmsd(config1Coords, config2Coords, config1Numbers, config2Numbers, minimize=True)

    return min_rmsd


def getNumberOfAtoms(id):

    atoms = read(f"xyz-s/all/{id}.xyz")

    return len(atoms)


def getRangeRMSDMeans(mlip_name):
    
    # Read the Dataset
    dataset = pd.read_csv("dataset/mols-energies.csv", index_col=3)
    ids = dataset.index.to_list()

    # print(np.max([getNumberOfAtoms(id) for id in ids]))
    
    ranges = np.arange(4, 22, 2)

    rangeRMSDs = defaultdict(list)
    for id in ids:
        configsDF = pd.read_csv(f"{mlip_name}/filtered-interaction-energy-values/id{id}_interaction_energies.csv", index_col=0)
        configs = configsDF.index
        nAtoms = getNumberOfAtoms(id)

        if len(configs) != 0:
            ref = configs[0]
        else:
            continue

        id_rmsds = []
        for config in configs[1:]:
            id_rmsds.append(getRMSD(mlip_name, id, ref, config))
        
        for i, range in enumerate(ranges[:-1]):
            if range <= nAtoms < ranges[i+1]:
                rangeRMSDs[f"{range}-{ranges[i+1]}"].append(np.mean(id_rmsds))
        
    
    rangeRMSDMeans = {key : np.mean(value) for key, value in rangeRMSDs.items()}

    # check
    # print(np.sum([len(val) for val in rangeRMSDs.values()]))

    return rangeRMSDMeans



from collections import Counter, defaultdict
import matplotlib.pyplot as plt

def main():

    rangeRMSDMeans = {
        '4 - 6': 0.6907337302137839, 
        '6 - 8': 0.9520833898404294,
        '8 - 10': 1.1985823901588906, 
        '10 - 12': 1.4322660079927494, 
        '12 - 14': 1.649444218631121, 
        '14 - 16': 1.856585195537642, 
        '16 - 18': 2.0219186589359475,
        '18 - 20': 2.2761476313109057, 
    }

    mlip_name = "UMA"

    # Read the Dataset
    dataset = pd.read_csv("dataset/mols-energies.csv", index_col=3)
    ids = dataset.index.to_list()
    
    
    Nfinal = 5
    coef = 0.8

    fewDistinct = {}
    goodDistinct = {}

    for id in ids:
        
        configsDF = pd.read_csv(f"{mlip_name}/filtered-interaction-energy-values/id{id}_interaction_energies.csv", index_col=0)
        configs = configsDF.index

        distinctConfigs = []

        nAtoms = getNumberOfAtoms(id)

        for range, rmsdMean in rangeRMSDMeans.items():
            a, b = (int(k) for k in range.split('-'))
            if a <= nAtoms < b:
                RMSDBar = rmsdMean * coef
                print(f"id: {id}, bar: {RMSDBar}")
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
                goodDistinct[id] = len(distinctConfigs)
                # updatedDF = configsDF.loc[distinctConfigs, :]
                # print(updatedDF)
                # inp = input('oooop')
                # updatedDF.to_csv(f"{mlip_name}/distinct-top-configurations/id{id}_interaction_energies.csv")
                break

        if len(distinctConfigs) < Nfinal:
            fewDistinct[id] = (len(distinctConfigs), nAtoms)
            # updatedDF = configsDF.loc[distinctConfigs, :]
            # updatedDF.to_csv(f"{mlip_name}/distinct-top-configurations/id{id}_interaction_energies.csv")

    print(goodDistinct)
    print(fewDistinct, len(fewDistinct))

# main()




























def getRangeRMSDMeansOld():

    mlip_name = "UMA"

    # Read the Dataset
    dataset = pd.read_csv("dataset/mols-energies.csv", index_col=3)
    ids = dataset.index.to_list()

    nAtoms = [getNumberOfAtoms(id) for id in ids]

    print(Counter(nAtoms))

    points = np.arange(4,22,2)
    
    ranges = defaultdict(list)
    for i, p in enumerate(points):
        if i == 0:
            continue
        
        for id in ids:
            if points[i-1] <= getNumberOfAtoms(id) < p:
                ranges[f"{points[i-1]} - {p}"].append(id)

    rangeRMSDs = defaultdict(list)

    for range, idlist in ranges.items():
        for id in idlist:
            configsDF = pd.read_csv(f"{mlip_name}/filtered-interaction-energy-values/id{id}_interaction_energies.csv", index_col=0)
            configs = configsDF.index

            RMSDarr = []
            for i, config in enumerate(configs):
            
                if i == 0:
                    refConfig = config
                    continue

                RMSD = getRMSD(mlip_name, id, refConfig, config)
                RMSDarr.append(RMSD)

            RMSDmean = np.mean(RMSDarr)
            rangeRMSDs[range].append(RMSDmean)

    rangeRMSDMean = {range: np.mean(rmsds) for range, rmsds in rangeRMSDs.items()}

    print(rangeRMSDMean)
    print(sum([len(vals) for keys, vals in rangeRMSDs.items()]))

    return rangeRMSDMean


# getRangeRMSDMeans()
