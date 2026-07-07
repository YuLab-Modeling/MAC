import numpy as np
import pandas as pd

import glob

# Read the Dataset
dataset = pd.read_csv("../dataset/mols-energies.csv", index_col=3)
ids = dataset.index.to_list()
n_calcs = 500

nConfigs = []
for id in ids:
    xyz_files = xyz_files = np.array(glob.glob(f'../abcluster/id{id}_{n_calcs}/*xyz'))
    nConfigs.append(len(xyz_files))

avgNConfigs = np.mean(nConfigs)
print(round(avgNConfigs))




# mlip time costs
for i, mlip_name in enumerate(["M3GNet", "UMA", "eSCN", "Mattersim"]):
    if i == 0:
        mlip_time_df = pd.read_csv(f"../{mlip_name}/{mlip_name}_opt_timecost.csv", index_col=0)
        mlip_time_df.index = [f"{mlip_name}_opt"]
    else:
        thisDF = pd.read_csv(f"../{mlip_name}/{mlip_name}_opt_timecost.csv", index_col=0)
        thisDF.index = [f"{mlip_name}_opt"]
        mlip_time_df = pd.concat([mlip_time_df, thisDF], axis=0)

mlip_time_df = mlip_time_df.T

#dft-spc time costs using OUTCAR files:

from glob import glob

mlip_name = "Mattersim"

def getTimeByVasp(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    
    cpu_line = lines[-11]

    return float(cpu_line.split()[-1])

vasp_time_dict = {}
dftspc_time_dict = {}

k = 5
for id in ids:

    mol_outcar = f'../{mlip_name}/dft-results/id{id}-mol/OUTCAR'
    mol_time = getTimeByVasp(mol_outcar)

    config_outcars = glob(f'../{mlip_name}/dft-results/id{id}-*/OUTCAR')
    config_outcars = [ff for ff in config_outcars if 'mol' not in ff]
    #print(config_outcars, len(config_outcars))

    this_config_times = []
    for config in config_outcars:
        this_config_times.append(getTimeByVasp(config))

    vasp_time_dict[id] = np.round(mol_time + k*np.mean(this_config_times), 3)
    dftspc_time_dict[id] = np.round(np.mean(this_config_times), 3)

vasp_time_df = pd.DataFrame(vasp_time_dict, index=[[0]])
vasp_time_df.index = ["VASP"]
vasp_time_df = vasp_time_df.T

dftspc_time_df = pd.DataFrame(dftspc_time_dict, index=[[0]])
dftspc_time_df.index = ["DFT_SPC"]
dftspc_time_df = dftspc_time_df.T

time_df = pd.concat([mlip_time_df, vasp_time_df, dftspc_time_df], axis=1)
print(time_df)

for i, mlip_name in enumerate(["M3GNet", "UMA", "eSCN", "Mattersim"]):
    time_df[f"{mlip_name}_opt_total"] = time_df[f"{mlip_name}_opt"]*avgNConfigs + time_df["VASP"]

# number of atoms
from ase.io import write, read

atom_numbers_dict = {}
for id in ids:
    atoms = read(f'../xyz-s/all/{id}.xyz')
    symbols = atoms.get_chemical_symbols()

    atom_numbers_dict[id] = len(symbols)

atom_numbers_df = pd.DataFrame(atom_numbers_dict, index=[[0]])
atom_numbers_df.index = ["Number of atoms"]

atom_numbers_df = atom_numbers_df.T

# all df
all_opt_df = pd.concat([time_df, atom_numbers_df], axis=1)

print(all_opt_df.columns)


# --------------------------------------------------------------------------------------------------------------------------------------------------
# plot mlip+dftspc speed plots
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# fig, ax = plt.subplots(figsize=(15,9))

# for i, mlip_name in enumerate(["M3GNet", "eSCN", "UMA", "Mattersim"]):
#     #sns.scatterplot(data=all_opt_df, x='Number of atoms', y=f'{mlip_name}_opt_total', ax=ax, label=mlip_name, alpha=0.2, color=colors[i])
#     sns.boxplot(data=all_opt_df, x='Number of atoms', y=f'{mlip_name}_opt_total', ax=ax, label=mlip_name, boxprops=dict(alpha=0.3), color=colors[i])

#     ax.set_ylabel("Time taken, sec", fontsize=14)
#     ax.set_xlabel("Number of atoms", fontsize=14)
#     ax.tick_params('both', labelsize=12)

# plt.legend(fontsize=12)
# plt.tight_layout()
# plt.savefig('Figure7_boxplot.png', dpi=300)

# plot all
# fig, axs = plt.subplots(1,2, figsize=(14,6))
# ax = axs[0]

fig = plt.figure(figsize=(12,7))
gs_master = gridspec.GridSpec(1,2, figure=fig, width_ratios=[2,1])

ax = fig.add_subplot(gs_master[0,0])

for i, mlip_name in enumerate(["M3GNet", "eSCN", "UMA", "Mattersim"]):
    # sns.scatterplot(data=all_opt_df, x='Number of atoms', y=f'{mlip_name}_opt_total', ax=ax, alpha=0.2, color=colors[i], label=mlip_name)

    sns.kdeplot(data=all_opt_df, x='Number of atoms', y=f'{mlip_name}_opt_total', ax=ax, label=mlip_name, color=colors[i], levels=[0.5, 0.9])

    print(mlip_name, np.mean(all_opt_df[f'{mlip_name}_opt_total']))

    ax.set_ylabel("Time taken, s", fontsize=14)
    ax.set_xlabel("Number of atoms", fontsize=14)
    ax.tick_params('both', labelsize=12)

ax.legend(handles = [mpatches.Patch(color=cc, label=gg) for cc, gg in zip(colors, ["M3G", "eSCN", "UMA", "MS"])], fontsize=12)
#ax.legend(fontsize=12)

ax.set_xlim([8, 20])
ax.set_ylim([200, 12500])



# --------------------------------------------------------------------------------------------------------------------------------------------------
# plot mlip_spc, xtb_spc and dft_spc plots
# ax = axs[1]
gs_sub = gridspec.GridSpecFromSubplotSpec(3,1, subplot_spec=gs_master[0,1], hspace=0.05)

ax0 = fig.add_subplot(gs_sub[0,0])
ax1 = fig.add_subplot(gs_sub[1,0])
ax2 = fig.add_subplot(gs_sub[2,0])

# mlip time spc costs
for i, mlip_name in enumerate(["M3GNet", "eSCN", "UMA", "Mattersim"]):
    if i == 0:
        mlip_time_df = pd.read_csv(f"../{mlip_name}/{mlip_name}_spc_timecost.csv", index_col=0)
        mlip_time_df.index = [f"{mlip_name}_spc"]
    else:
        thisDF = pd.read_csv(f"../{mlip_name}/{mlip_name}_spc_timecost.csv", index_col=0)
        thisDF.index = [f"{mlip_name}_spc"]
        mlip_time_df = pd.concat([mlip_time_df, thisDF], axis=0)

mlip_time_df = mlip_time_df.T

# load xtb spc data
xtb_spc_df = pd.read_csv(f"../xtb_timecost.csv", index_col=0)
xtb_spc_df.index = ["xTB_SPC"]
xtb_spc_df = xtb_spc_df.T

all_spc_df = pd.concat([mlip_time_df, xtb_spc_df, dftspc_time_df, atom_numbers_df], axis=1)

# print(all_spc_df)

# fig, ax = plt.subplots(figsize=(12,8))

#sns.scatterplot(data=all_spc_df, x='Number of atoms', y=f'Mattersim_spc', ax=ax, color='#9467bd', alpha=0.2)
sns.kdeplot(data=all_spc_df, x='Number of atoms', y=f'Mattersim_spc', ax=ax2, color='#9467bd', levels=[0.5, 0.9], log_scale=[False, True])
print("MS spc mean t: ", np.mean(all_spc_df["Mattersim_spc"]))

#sns.scatterplot(data=all_spc_df, x='Number of atoms', y=f'xTB_SPC', ax=ax, color='#bcbd22', alpha=0.2)
sns.kdeplot(data=all_spc_df, x='Number of atoms', y=f'xTB_SPC', ax=ax1, color='#bcbd22', levels=[0.5, 0.9], log_scale=[False, True])
print("xTB spc mean t: ", np.mean(all_spc_df["xTB_SPC"]))

#sns.scatterplot(data=all_spc_df, x='Number of atoms', y=f'DFT_SPC', ax=ax, color='#17becf', alpha=0.2)
sns.kdeplot(data=all_spc_df, x='Number of atoms', y=f'DFT_SPC', ax=ax0, color='#17becf', levels=[0.5, 0.9], log_scale=[False, True])
print("DFT spc mean t: ", np.mean(all_spc_df["DFT_SPC"]))

# limits ------------------------------------------------
ax2.set_ylim([4e-2, 6.5e-2])
ax1.set_ylim([2, 3e1])
ax0.set_ylim([1.5e2, 3e2])

ax2.set_xlim([9,18])
ax1.set_xlim([9,18])
ax0.set_xlim([9,18])

# ticks ------------------------------------------------

ax0.spines['bottom'].set_visible(False)
ax0.set_xticks([])

ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.set_xticks([])

ax2.spines['top'].set_visible(False)

# cosmetics ----------------------------------------------

ax0.set_xlabel(""); ax0.set_ylabel("")
ax0.tick_params('both', labelsize=12)

ax1.set_ylabel("Time taken, s", fontsize=14)
ax1.set_xlabel(""); ax0.set_ylabel("")
ax1.tick_params('both', labelsize=12)

ax2.set_xlabel("Number of atoms", fontsize=14); ax2.set_ylabel("")
ax2.tick_params('both', labelsize=12)

# draw the split ----------------------------------------------------------------------------------

d = .020

# ax0
kwargs = dict(linewidth=0.75, transform=ax0.transAxes, color='k', clip_on=False)
ax0.plot((1-d, 1+d), (-d, +d), **kwargs)        # bottom
ax0.plot((-d, +d), (-d, +d), **kwargs)          # bottom

# ax1
kwargs.update(transform=ax1.transAxes)
ax1.plot((-d, +d), (1-d, 1+d), **kwargs)        # top
ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)      # top
ax1.plot((1-d, 1+d), (-d, +d), **kwargs)        # bottom
ax1.plot((-d, +d), (-d, +d), **kwargs)          # bottom

# ax2
kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (1-d, 1+d), **kwargs)        # top
ax2.plot((1-d, 1+d), (1-d, 1+d), **kwargs)      # top


# yticks
ax0.set_yticks([2e2])
ax1.set_yticks([1e1])
ax2.set_yticks([5e-2])
ax0.yaxis.set_minor_locator(ticker.NullLocator())
ax1.yaxis.set_minor_locator(ticker.NullLocator())
ax2.yaxis.set_minor_locator(ticker.NullLocator())


lims = [8, 18]
ax0.set_xlim(lims)
ax1.set_xlim(lims)
ax2.set_xlim(lims)

colors_dict = {
    'MS SPC' : '#9467bd',
    'xTB SPC' : '#bcbd22', 
    'DFT SPC' : '#17becf'
}

ax2.legend(handles = [mpatches.Patch(color=cc, label=gg) for gg, cc in colors_dict.items()], fontsize=12, loc='lower left')

#plt.legend(fontsize=12)
plt.tight_layout(w_pad=2.0)

s = ["a", "b"]
ax.text(0.05, 0.9, f"a", transform=ax.transAxes, fontweight='bold', fontsize=16)
ax0.text(0.05, 0.8, f"b", transform=ax0.transAxes, fontweight='bold', fontsize=16)

#plt.show()
plt.savefig("Figure7_final_upd.png", dpi=300)