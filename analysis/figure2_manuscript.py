import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
import seaborn as sns
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

from sklearn.metrics import mean_absolute_error, root_mean_squared_error

groupOrder = ['amides', 'amidines', 'aromatics', 'carbamate_esters', 'group2', 'group3N', 'group3S', 'group4', 'oximes']

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

pastelDict = {
    'Amides' : '#a1c9f4', 
    'Amidines' : '#ffb482', 
    'Aromatics' : '#8de5a1',
    'Carbamate esters' : '#ff9f9b', 
    'Group2' : '#d0bbff', 
    'Group3N' : '#debb9b', 
    'Group3S' : '#fab0e4', 
    'Group4' : '#cfcfcf', 
    'Oximes' : '#fffea3'
}

mlip_colors = {
    "M3GNet" : "#1f77b4",
    "UMA" : "#2ca02c",
    "eSCN" : "#ff7f0e",
    "Mattersim" : "#d62728"
}

def DFTOPT_SPC(mlip_name, ax):
        
    AllDF = pd.read_csv(f"../{mlip_name}/AllEnergies.csv", index_col=0)

    AllDF = AllDF.drop(['8a03', '8a02', '8a01', '66S1', '67N1', '7801'])
    if mlip_name == "eSCN":
        AllDF = AllDF[AllDF['DFT_SPC minEInt'] > -4]

    # group2b to group2
    AllDF["group"] = AllDF["group"].apply(lambda x: "group2" if x == "group2b" else x )
    AllDF["group"] = AllDF["group"].apply(lambda x: groupsDict[x] )

    # E_int comparison plots
    method1 = "DFT_OPT E_int"
    method2 = "DFT_SPC minEInt" if engine == "DFT_SPC" else "XTB minEInt"

    print(mlip_name, np.nanmin(AllDF[method2]), np.nanmax(AllDF[method2]))

    # perfect line
    lowerBound = np.nanmin(AllDF[method1].values.tolist() + AllDF[method2].values.tolist()) - 0.1
    upperBound = np.nanmax(AllDF[method1].values.tolist() + AllDF[method2].values.tolist()) + 0.1
    ax.plot(np.arange(lowerBound, upperBound+0.2,0.1), np.arange(lowerBound, upperBound+0.2,0.1), '--', color='black', alpha=0.6, linewidth=1)
    
    # pair plot
    #sns.scatterplot(data=AllDF, x=method2, y=method1, ax=ax, color='#298c8c' if engine == "DFT_SPC" else "#a00000", alpha=0.75)
    sns.scatterplot(data=AllDF, x=method2, y=method1, ax=ax, color=mlip_colors[mlip_name] if engine == "DFT_SPC" else "#a00000", alpha=0.75)

    # MAE/RMSE
    row1 = AllDF[method1]
    row2 = AllDF[method2]
    
    mask = ~np.isnan(row1) & ~np.isnan(row2)

    mae = round(mean_absolute_error(row1[mask], row2[mask]), 2)
    rmse = round(root_mean_squared_error(row1[mask], row2[mask]), 2)
    ax.text(0.45, 0.12, f"RMSE = {rmse} eV", transform=ax.transAxes, fontsize=14)
    ax.text(0.45, 0.05, f"MAE   = {mae} eV", transform=ax.transAxes, fontsize=14)

    print(mlip_name, min(row2), max(row2), "DFT-OPT", min(row1), max(row1))


    # distribution
    divider = make_axes_locatable(ax)
    ax_top = divider.append_axes("top", size="20%", pad=0.1, sharex=ax)
    #sns.kdeplot(data=AllDF, x=method2, ax=ax_top, fill=True, alpha=0.6, color='#298c8c' if engine == "DFT_SPC" else "#a00000")
    sns.kdeplot(data=AllDF, x=method2, ax=ax_top, fill=True, alpha=0.6, color=mlip_colors[mlip_name] if engine == "DFT_SPC" else "#a00000", bw_adjust=0.5)
    ax_top.axis('off')

    if mlip_name == "UMA":
        ax_right = divider.append_axes("right", size="20%", pad=0.1, sharey=ax)
        sns.kdeplot(data=AllDF, y=method1, ax=ax_right, fill=True, alpha=0.6, color='#298c8c' if engine == "DFT_SPC" else "#a00000")
        ax_right.axis('off')
    
    # inset for m3gnet
    if mlip_name == "M3GNet":
        ax_inset = ax.inset_axes([-0.1, 0.6, 0.35, 0.35])
        ax_inset.plot(np.arange(lowerBound, upperBound+0.2,0.1), np.arange(lowerBound, upperBound+0.2,0.1), '--', color='black', alpha=0.6, linewidth=1)
        #sns.scatterplot(data=AllDF, x=method2, y=method1, ax=ax_inset, color='#298c8c' if engine == "DFT_SPC" else "#a00000", alpha=0.75, s=8)
        sns.scatterplot(data=AllDF, x=method2, y=method1, ax=ax_inset, color=mlip_colors[mlip_name] if engine == "DFT_SPC" else "#a00000", alpha=0.75, s=8)
        ax_inset.set_ylabel("", fontsize=10)
        ax_inset.set_xlabel("", fontsize=10)
        ax_inset.set_xlim([lowerBound, upperBound])
        ax_inset.set_ylim([lowerBound, upperBound])

        divider2 = make_axes_locatable(ax_inset)
        ax_top2 = divider2.append_axes("top", size="20%", pad=0.1, sharex=ax_inset)
        #sns.kdeplot(data=AllDF, x=method2, ax=ax_top2, fill=True, alpha=0.6, color='#298c8c' if engine == "DFT_SPC" else "#a00000")
        sns.kdeplot(data=AllDF, x=method2, ax=ax_top2, fill=True, alpha=0.6, color=mlip_colors[mlip_name] if engine == "DFT_SPC" else "#a00000")
        ax_top2.axis('off')
    
    # bounds
    ax.set_xlim([-1.5, 0.6])
    ax.set_ylim([-1.5, 0.6])
    ax.set_xticks([-1.5, -1.0, -0.5, 0, 0.5])
    ax.set_yticks([-1.5, -1.0, -0.5, 0, 0.5])

    # labels
    if mlip_name in ["M3GNet", "Mattersim"]:
        ax.set_ylabel(r"$E_{\mathrm{ads}}$ (DFT-OPT), eV", fontsize=14)

        if mlip_name == "M3GNet":
            mlip_ref_txt = "M3G"
        if mlip_name == "Mattersim":
            mlip_ref_txt = "MS"
    else:
        mlip_ref_txt = mlip_name
        ax.set_ylabel("", fontsize=14)

    ax.set_xlabel(rf"$E_{{\mathrm{{ads}}}}$ ($MAC_{{{mlip_ref_txt}}}^{{DFT}}$), eV" if engine == "DFT_SPC" else rf"$E_{{\mathrm{{ads}}}}$ ($MAC_{{{mlip_ref_txt}}}^{{XTB}}$), eV", fontsize=14)
    # ax.set_title(mlip_name, fontsize=16, fontweight='bold')
    

def getSuccessRate(AllDF, k=5, bar=0.2):

    method = "DFT_SPC" if engine == "DFT_SPC" else "XTB"
    
    kMinEnergyDict = {}
    for ind in AllDF.index:
        
        strArr = AllDF.loc[ind, method + " configEnergies"]
        
        if type(strArr) == str:
        
            enerVals = [float(x) for x in strArr[1:-1].split(",")]

            if np.any(~np.isnan(enerVals[:k])):
                kMinEnergyDict[ind] = np.nanmin(enerVals[:k])
            else:
                kMinEnergyDict[ind] = np.nan
        
    kMinEnergySeries = pd.Series(kMinEnergyDict)
    kMinEInt = kMinEnergySeries - AllDF[method + " molEnergy"] - AllDF[method + " slabEnergy"]
    # organize the indices in order
    kMinEInt = kMinEInt.loc[AllDF.index]

    isSuccess = []

    for methodVal, optVal in zip(kMinEInt.values, AllDF["DFT_OPT E_int"].values):

        if np.isnan(methodVal) or np.isnan(optVal):
            isSuccess.append("N/A")
            continue
        
        if method == "DFT_SPC" and methodVal < optVal:
            isSuccess.append("success")
            continue

        if abs(methodVal - optVal) < bar:
            isSuccess.append("success")
            continue
        
        isSuccess.append("fail")

    isSuccess = np.array(isSuccess)
    Ntot = sum(isSuccess == "success") + sum(isSuccess == "fail")
    Nsuccess = sum(isSuccess == "success")

    return round((Nsuccess * 100) / Ntot, 2 ) 

def plotSuccessRate(mlip_name, ax, ind, kMax=5):

    linestyles = [ '--.', '--s', '--o', '--*']

    AllDF = pd.read_csv(f"../{mlip_name}/AllEnergies.csv", index_col=0)

    AllDF = AllDF.drop(['8a03', '8a02', '8a01', '66S1', '67N1', '7801'])
    AllDF = AllDF[AllDF['DFT_SPC minEInt'] > -4]

    AllDF["group"] = AllDF["group"].apply(lambda x: "group2" if x == "group2b" else x )


    if mlip_name == "M3GNet":
        mlip_ref_txt = "M3G"
    elif mlip_name == "Mattersim":
        mlip_ref_txt = "MS"
    else:
        mlip_ref_txt = mlip_name

    ax.plot(np.arange(1,kMax+1,1), [getSuccessRate(AllDF=AllDF, k=k) for k in range(1,kMax+1)], linestyles[ind], label=mlip_ref_txt, alpha=0.8)
    ax.set_xlabel("k", fontsize=14, fontstyle='italic')
    ax.set_ylabel("Success rate, %", fontsize=14)
    ax.set_title("Success rate", fontsize=16, fontweight='bold')
    ax.set_ylim([0, 100])

    # ghost top axis for alignment
    divider = make_axes_locatable(ax)
    ax_top = divider.append_axes("top", size="20%", pad=0.1)
    ax_top.axis('off') # This 'squeezes' ax2 downward to align with ax1

    print(mlip_name, [(k, getSuccessRate(AllDF=AllDF, k=k)) for k in range(1,kMax+1)])

from scipy.stats import spearmanr, kendalltau
def getRankCoefs(data):
    
    # function to calculate the ranking coefficients: Spearman and Kendell

    method1="DFT_OPT E_int"
    method2 = "DFT_SPC minEInt" if engine == "DFT_SPC" else "XTB minEInt"

    data = data[[method1, method2]]
    #spearmanCoef = data.corr(method="spearman").values[0,1]

    vals1 = data[method1].values
    vals2 = data[method2].values

    indnan1 = ~np.isnan(vals1)
    indnan2 = ~np.isnan(vals2)

    indnan = indnan1 & indnan2

    vals1 = vals1[indnan]
    vals2 = vals2[indnan]

    spearmanCoef, _ = spearmanr(vals1, vals2)

    kendallCoef, _ = kendalltau(vals1, vals2)

    return round(spearmanCoef, 3), round(kendallCoef, 3)

def plotRankCoefs(mlipNames, ax):

    spearmanCoefsArr = []
    kendallCoefsArr = []

    for mlip_name in mlipNames:
        AllDF = pd.read_csv(f"../{mlip_name}/AllEnergies.csv", index_col=0)

        AllDF = AllDF.drop(['8a03', '8a02', '8a01', '66S1', '67N1', '7801'])
        AllDF = AllDF[AllDF['DFT_SPC minEInt'] > -4]

        AllDF["group"] = AllDF["group"].apply(lambda x: "group2" if x == "group2b" else x )

        spearmanCoef, kendallCoef = getRankCoefs(AllDF)
        spearmanCoefsArr.append(spearmanCoef)
        kendallCoefsArr.append(kendallCoef)

        print(mlip_name, 'p', spearmanCoef, 'tau', kendallCoef)

    x = np.arange(len(mlipNames))
    width = 0.35
    ax.bar(x - width/2, spearmanCoefsArr, width, label=r"$\rho$",  alpha=0.7, edgecolor='black')
    ax.bar(x + width/2, kendallCoefsArr, width, label=r"$\tau$",  alpha=0.7, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(["M3G", "eSCN", "UMA", "MS"])
    ax.set_ylim([-0.1, 1])
    ax.set_title("Ranking coefficients", fontsize=16, fontweight='bold')
    ax.tick_params('x', labelsize=12)
    # ax.set_ylabel(r"$\rho$/$\tau$")
    ax.legend()

    # ghost top axis for alignment
    divider = make_axes_locatable(ax)
    ax_top = divider.append_axes("top", size="20%", pad=0.1)
    ax_top.axis('off') # This 'squeezes' ax2 downward to align with ax1

    # ghost right axis for alignment
    ax_right = divider.append_axes("right", size="20%", pad=0.1)
    ax_right.axis('off') # This 'squeezes' ax2 downward to align with ax1
    


engine = "DFT_SPC"
#engine = "XTB"

# fig, axes = plt.subplots(2,3, figsize=(12,9))
# axs = axes.flatten()

fig = plt.figure(figsize=(13.5,9))
gs_master = fig.add_gridspec(2,3, figure=fig, width_ratios=[1,1,1.2])
axes = gs_master.subplots()
axs = axes.flatten()


s = ["a", "b", "c", "d", "e", "f"]

# pair plots
for i, mlip in enumerate(["M3GNet", "eSCN", "UMA", "Mattersim"]):
    DFTOPT_SPC(mlip_name=mlip, ax=axs[i])

# success rate
for i, mlip in enumerate(["M3GNet", "eSCN", "UMA", "Mattersim"]):
    plotSuccessRate(mlip_name=mlip, ax=axs[4], ind=i)

if engine == "DFT_SPC":
    axs[4].plot(np.arange(1,6,1), [2.49, 2.49, 2.49, 2.99, 3.98], '--^', label="Random", alpha=0.8, color='#326b77')

axs[4].legend()

# ranking coefficients
plotRankCoefs(["M3GNet", "eSCN", "UMA", "Mattersim"], ax=axs[5])

plt.subplots_adjust(hspace=0.15, wspace=0.25, bottom=0.075, left=0.075, right=0.975, top=0.95)

# legend for pair plots
# axs[3].legend(loc='lower center', bbox_to_anchor=(1.7,-0.43), ncol=7, fontsize=12)

s = ["a", "b", "c", "d", "e", "f", "g", "h"]
for i, a in enumerate(axs):
    a.text(0.05, 0.9, f"{s[i]}", transform=a.transAxes, fontweight='bold', fontsize=16)

#plt.show()

#plt.savefig("Figure_3.png") if engine == "DFT_SPC" else plt.savefig("Figure_4.png")

