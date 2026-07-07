import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import seaborn as sns

groupsDict = {
    'amides' : 'Amides',
    'amidines' : 'Amidines',
    'aromatics' : 'Aromatics',
    'carbamate_esters' : 'Carbamate esters',
    'group2' : 'Group2',
    'group3N' : 'Group3N',
    'group3S' : 'Group3S',
    'group4' : 'Group4',
    'oximes' : 'Oximes',
}

# ROW 1: dE -----------------------------------------------------------------------------------------------------------------------------------------

def dE(ax, pipe='DFT_SPC'):

    for i, mlip_name in enumerate(["M3GNet", "eSCN", "UMA", "Mattersim"]):
        MLIPDF = pd.read_csv(f"../{mlip_name}/AllEnergies.csv")

        if mlip_name == "eSCN" and pipe == "DFT_SPC":
            MLIPDF = MLIPDF[MLIPDF[f'{pipe} minEInt'] > -4]

        MLIPDF["group"] = MLIPDF["group"].apply(lambda x: "group2" if x == "group2b" else x )
        MLIPDF["group"] = MLIPDF["group"].apply(lambda x: groupsDict[x] )

        method1 = "DFT_OPT E_int"
        method2 = "DFT_SPC minEInt" if pipe == "DFT_SPC" else "XTB minEInt"
    
        MLIPDF["dE"] = MLIPDF[method2] - MLIPDF[method1]

        if i == 0:
            
            AllDF = pd.DataFrame(MLIPDF['dE'])
            AllDF["MLIP"] = mlip_name

        else:
            tempDF = pd.DataFrame(MLIPDF['dE'])
            tempDF["MLIP"] = mlip_name
            AllDF = pd.concat([AllDF, tempDF], axis=0)

    AllDF.reset_index(inplace=True)
    print(AllDF)    
        
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.3)

    sns.stripplot(data=AllDF, x='MLIP', y=f"dE", ax=ax, hue='MLIP', alpha=0.3, zorder=0)    
    sns.boxplot(data=AllDF, x='MLIP', y='dE', ax=ax, hue='MLIP',  boxprops=dict(alpha=0.5), saturation=1.0,  showmeans=True, meanprops={'marker':'*','markerfacecolor':'red','markeredgecolor':'black','markersize':'10'}) #, palette=pastelDict, order=groupOrder, legend="")
    
    # ax.set_title("dE", fontsize=16)
    #ax.set_ylabel(r'$E_{ML-SPC} - E_{DFT-OPT}$, eV')
    ax.set_xticklabels([]); ax.set_xticks([]) 
    ax.set_ylabel(fr'$\Delta E$ (eV)', fontsize=18)
    ax.tick_params(which='major', labelsize=16)
    ax.set_xlabel('', fontsize=12)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_ylim([-1.2, 2.7])

# ROW 2: dS -----------------------------------------------------------------------------------------------------------------------------------------

from distincting_functions_side_proj import getCutStructure
from spyrmsd import rmsd
from irmsd import get_irmsd_ase
from scipy.optimize import linear_sum_assignment

def getRMSD(id, mlip_name, type):

    enerVals = pd.read_csv(f"../{mlip_name}/dft-energy-values/{id}.csv", index_col=0)
    
    # in case no configurations passed the filters and for none a dft-spc was run
    if len(enerVals) == 0:
        return np.nan
    
    minEnerConfig = enerVals.index[0]
    # print(id, minEnerConfig)

    atoms1 = getCutStructure(f"../dft-opt-structures/zn-{id}.xyz", type=type)
    atoms1.set_cell([9.405, 9.405, 20., 90, 90, 60])
    atoms1.pbc=[1, 1, 1]
    config1Coords = atoms1.get_positions()

    atoms2 = getCutStructure(f"../{mlip_name}/opt-configurations/id{id}/{minEnerConfig}.xyz", type=type)
    atoms2.set_cell([9.405, 9.405, 20., 90, 90, 60])
    atoms2.pbc=[1,1,1]
    config2Coords = atoms2.get_positions()

    inv_rmsd, _, _ = get_irmsd_ase(atoms1, atoms2)

    return inv_rmsd

def dS(ax, type="Mol"):
    
    for i, mlip_name in enumerate(["M3GNet", "eSCN", "UMA", "Mattersim"]):

        MLIPDF = pd.read_csv(f"../{mlip_name}/AllEnergies.csv", index_col=0)

        MLIPDF = MLIPDF.drop(['8a03', '8a02', '8a01', '66S1', '67N1', '7801'])

        if mlip_name == "eSCN":
            MLIPDF = MLIPDF[MLIPDF['DFT_SPC minEInt'] > -4]
        
        for ind in MLIPDF.index:
            MLIPDF.loc[ind, "dS"] = getRMSD(id=ind, mlip_name=mlip_name, type=type)

        print(f"{mlip_name} RMSD mean, {np.mean(MLIPDF['dS'])}")

        if mlip_name == "M3GNet":
            print(mlip_name, sum(MLIPDF["dS"] > 1.13))
        elif mlip_name == "eSCN":
            print(mlip_name, sum(MLIPDF["dS"] > 0.59))
        elif mlip_name == "UMA":
            print(mlip_name, sum(MLIPDF["dS"] > 0.62))
        elif mlip_name == "Mattersim":
            print(mlip_name, sum(MLIPDF["dS"] > 0.93))

        if i == 0:
            
            AllDF = pd.DataFrame(MLIPDF['dS'])
            AllDF["MLIP"] = mlip_name

        else:
            tempDF = pd.DataFrame(MLIPDF['dS'])
            tempDF["MLIP"] = mlip_name
            AllDF = pd.concat([AllDF, tempDF], axis=0)

    AllDF.reset_index(inplace=True)
    # print(AllDF)    

        
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.3)

    sns.stripplot(data=AllDF, x='MLIP', y="dS", ax=ax, hue='MLIP', alpha=0.3, zorder=0)
    sns.boxplot(data=AllDF, x='MLIP', y='dS', ax=ax, hue='MLIP', boxprops=dict(alpha=0.5), saturation=1.0,  showmeans=True, meanprops={'marker':'*','markerfacecolor':'red','markeredgecolor':'black','markersize':'10'}, showfliers=False) #, palette=pastelDict, order=groupOrder, legend="")

    # tick outliers

    ax.set_xticklabels([]); ax.set_xticks([])
    ax.tick_params(which='major', labelsize=16)
    ax.set_ylabel(r"Mol RMSD ($\AA$)", fontsize=18)
    # ax.set_title(r"Mol RMSD ($\AA$)", fontsize=16)
    ax.set_xlabel('', fontsize=12)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_ylim([-1.2, 2.7])

    # if mlip_name groups== "Mattersim":
    #     ax.legend(handles = [mpatches.Patch(color=cc, label=gg) for gg, cc in pastelDict.items()], bbox_to_anchor=(0.95, 0, 0.5, 1), loc='center')


# Row 3: d [Min Distance] -----------------------------------------------------------------------------------------------------------------------------------------

import ase
def closest_atom_zn_dist(engine, id):
    
    if engine == "DFT":
        xyz_path = f"../dft-opt-structures/zn-{id}.xyz"
    
    else:
        enerVals = pd.read_csv(f"../{engine}/dft-energy-values/{id}.csv", index_col=0)

        # in case no configurations passed the filters and for none a dft-spc was run
        if len(enerVals) == 0:
            return np.nan, np.nan
        minEnerConfig = enerVals.index[0]

        xyz_path = f"../{engine}/opt-configurations/id{id}/{minEnerConfig}.xyz"

    atoms = ase.io.read(xyz_path)
    atoms.set_cell([9.405, 9.405, 20., 90, 90, 60])
    atoms.pbc=[1, 1, 1]
    positions = atoms.get_positions(wrap=True)
    symbols = atoms.symbols

    slab_coords = [pos for i, pos in enumerate(positions) if symbols[i] == "Zn"]
    mol_coords = [pos for i, pos in enumerate(positions) if symbols[i] != "Zn"]
    mol_symbols = [sym for sym in symbols if sym != "Zn"]
    
    # position of molecule's atom with the lowest z value
    mol_z_dims = [line[2] for line in mol_coords]
    min_ind = np.argmin(mol_z_dims)

    min_z_pos = mol_coords[min_ind]
    min_z_sym = mol_symbols[min_ind]

    dist_arr = []
    for slab_atom_pos in slab_coords:
        dist = np.linalg.norm(np.array(min_z_pos) - np.array(slab_atom_pos))

        dist_arr.append(dist)

    min_dist = np.min(dist_arr)

    # if min_dist > 5:
    #     print(engine, id, minEnerConfig)
    #     print(dist_arr)
    #     print(min_z_pos)
    #     print(slab_coords)

    return min_z_sym, min_dist

def dMinDistance(ax):

    # DFT Reference

    engine = "DFT"
    DFTDF = pd.read_csv(f"../Mattersim/AllEnergies.csv", index_col=0)
    DFTDF = DFTDF.drop(['8a03', '8a02', '8a01', '66S1', '67N1', '7801'])
    
    for ind in DFTDF.index:
        el, dist = closest_atom_zn_dist(engine, ind)

        DFTDF.loc[ind, "Dist"] = dist
        DFTDF.loc[ind, "Element"] = el

    DFTDF.sort_index(inplace=True)

    # MLIPs

    for i, engine in enumerate([ "M3GNet", "eSCN", "UMA", "Mattersim"]):
  
        MLIPDF = pd.read_csv(f"../{engine}/AllEnergies.csv", index_col=0)

        MLIPDF = MLIPDF.drop(['8a03', '8a02', '8a01', '66S1', '67N1', '7801'])

        for ind in MLIPDF.index:
            el, dist = closest_atom_zn_dist(engine, ind)

            MLIPDF.loc[ind, "Dist"] = dist
            MLIPDF.loc[ind, "Element"] = el

        MLIPDF.sort_index(inplace=True)

        if i == 0:
            AllDF = pd.concat([MLIPDF["Dist"], MLIPDF["Element"]], axis=1)
            AllDF["dMinDistance"] = AllDF["Dist"] - DFTDF["Dist"]
            print(f"{engine} |delta d| mean, {np.mean(np.abs(AllDF['dMinDistance']))}")
            AllDF["same_binding_site"] = AllDF["Element"] == DFTDF["Element"]
            AllDF["MLIP"] = engine

        else:
            tempDF = pd.concat([MLIPDF["Dist"], MLIPDF["Element"]], axis=1)
            tempDF["dMinDistance"] = tempDF["Dist"] - DFTDF["Dist"]
            print(f"{engine} |delta d| mean, {np.mean(np.abs(tempDF['dMinDistance']))}")
            tempDF["same_binding_site"] = tempDF["Element"] == DFTDF["Element"]
            tempDF["MLIP"] = engine
            AllDF = pd.concat([AllDF, tempDF], axis=0)

    AllDF.reset_index(inplace=True)
    # print(AllDF)    

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.3)

    sns.stripplot(data=AllDF, x='MLIP', y=f"dMinDistance", ax=ax, hue='MLIP', alpha=0.3, zorder=0)
    sns.boxplot(data=AllDF, x='MLIP', y=f"dMinDistance", ax=ax, hue='MLIP', boxprops=dict(alpha=0.5), saturation=1.0, showmeans=True, meanprops={'marker':'*','markerfacecolor':'red','markeredgecolor':'black','markersize':'10'}, showfliers=False) #, palette=pastelDict, order=groupOrder, legend="")
    
    # ax.set_xticklabels([]); ax.set_xticks([])
    ax.tick_params(which='major', labelsize=16)
    ax.set_xlabel('')
    ax.set_ylabel(r"$\Delta d_{min}$ ($\AA$)", fontsize=18)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_ylim([-1.2, 2.7])

    ax2 = ax.twinx()

    # calculate binding site accuracy and line-plot
    bs_acc_dict = {}
    for i, mlip in enumerate([ "M3GNet", "eSCN", "UMA", "Mattersim"]):
        mlip_bs = AllDF[AllDF['MLIP'] == mlip][f"same_binding_site"]

        bs_acc_dict[mlip] = np.mean(mlip_bs)*100

    print(bs_acc_dict)
    sns.lineplot(x=bs_acc_dict.keys(), y=bs_acc_dict.values(), ax=ax2, marker='o', linestyle='--', color='blue', alpha=0.6)
    ax2.set_ylim([-45,100])
    ax2.set_xticklabels([]); ax2.set_xticks([])
    ax2.set_xlabel("")


    ax2.tick_params(which='both', labelsize=12, colors='blue')
    ax2.set_ylabel("Closest-adsorbate-atom accuracy, %", color='blue', fontsize=16)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])

# Plot -----------------------------------------------------------------------------------------------------------------------------------------

fig, axes = plt.subplots(1,2, figsize=(14,5))

#dE(axes[0])
dS(axes[0])
dMinDistance(axes[1])


pastelDict = {
    'M3G' : '#1f77b4',  
    'eSCN' : '#ff7f0e', 
    'UMA' : '#2ca02c',
    'MS' : '#d62728',
}

plt.subplots_adjust(hspace=0.1, wspace=0.3, bottom=0.2)#, left=0.05, right=0.965, top=0.95)
axes[1].legend(handles = [mpatches.Patch(color=cc, label=gg, alpha=0.5) for gg, cc in pastelDict.items()], bbox_to_anchor=(-0.18,-0.2), loc='lower center', ncol=9, fontsize=16)

s = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]
for i, a in enumerate(axes):
    a.text(0.05, 0.9, f"{s[i]}", transform=a.transAxes, fontweight='bold', fontsize=18)

plt.savefig("Figure_5_RMSD_decoupled_combined.png", dpi=300)
#plt.show()





